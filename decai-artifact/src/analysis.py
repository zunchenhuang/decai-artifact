"""Abstract interpretation of autoregressive top-k decoding.

Abstract state  (q, S_out)  where
  q      : state of the forbidden-substring monitor (partition key)
  S_out  : tuple of token sets, one per generated position (singletons in tree mode)
Input abstraction: S_in = (S_1..S_L), a positional character-set domain; gamma = S_1 x ... x S_L.
Numeric lifting: every position's embedding lies in the box hull of {tok(c) : c in S} (+pos+seg),
bounded through the transformer body by auto_LiRPA (CROWN).
Top-k abstract transformer: t is excluded iff at least k tokens j satisfy  lb(z_j - z_t) > 0.
"""
import time, numpy as np, torch
try:                                        # auto_LiRPA is optional: only the ablation backend needs it
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
except ImportError:
    BoundedModule = BoundedTensor = PerturbationLpNorm = None
from tasks import VOCAB, N_ASCII, BOS, SEP, Monitor
from model import CharLM, LastStep

ASCII = np.zeros(VOCAB, bool); ASCII[:N_ASCII] = True

def singleton(t):
    m = np.zeros(VOCAB, bool); m[t] = True; return m

class Analyzer:
    def __init__(self, lm, monitor, k, rule="margin", method="CROWN", row_cap=4000, extra=8, domain="hull", backend="lirpa"):
        self.lm, self.mon, self.k, self.rule, self.method, self.domain = lm.eval(), monitor, k, rule, method, domain
        self.backend = backend
        self.tau = 0.0
        self.min_used_margin = float('inf')
        self.row_cap, self.extra = row_cap, extra
        self.bm = {}                                     # T -> BoundedModule
        self.stats = dict(bound_calls=0, rows=0)
        with torch.no_grad():
            self.tokE = lm.tok.weight.detach().clone()
            self.posE = lm.pos.weight.detach().clone()
            self.segE = lm.seg.weight.detach().clone()

    # ---------------- numeric lifting ----------------
    def _pos_box(self, S, pos, seg):
        E = self.tokE[torch.from_numpy(np.flatnonzero(S))]
        base = self.posE[pos] + self.segE[seg]
        return E.min(0).values + base, E.max(0).values + base

    def vertices(self, S_in, S_out):
        L, n = len(S_in), len(S_out)
        pos, seg = CharLM.layout(L, n)
        sets = [singleton(BOS)] + list(S_in) + [singleton(SEP)] + list(S_out)
        return [self.tokE[torch.from_numpy(np.flatnonzero(S))] + self.posE[int(p)] + self.segE[int(g)]
                for S, p, g in zip(sets, pos, seg)]

    def box(self, S_in, S_out):
        L, n = len(S_in), len(S_out)
        pos, seg = CharLM.layout(L, n)
        sets = [singleton(BOS)] + list(S_in) + [singleton(SEP)] + list(S_out)
        lo, hi = zip(*(self._pos_box(S, int(p), int(s)) for S, p, s in zip(sets, pos, seg)))
        return torch.stack(lo)[None], torch.stack(hi)[None]

    def _bounded(self, T):
        if T not in self.bm:
            self.bm[T] = BoundedModule(LastStep(self.lm, T), torch.zeros(1, T, self.tokE.shape[1]),
                                       device="cpu", verbose=False)
        return self.bm[T]

    def _bounds(self, EL, EU, C=None, V=None):
        bm = self._bounded(EL.shape[1])
        from hull import PerturbationHull
        ptb = PerturbationHull(V) if (self.domain == "hull" and V is not None) else \
              PerturbationLpNorm(norm=np.inf, x_L=EL, x_U=EU)
        x = BoundedTensor((EL + EU) / 2, ptb)
        self.stats["bound_calls"] += 1
        self.stats["rows"] += VOCAB if C is None else C.shape[1]
        with torch.no_grad():
            try:
                lb, ub = bm.compute_bounds(x=(x,), method=self.method, C=C)
            except RuntimeError as e:            # auto_LiRPA >= 0.8 raises on NaN bounds; a NaN bound carries no
                if "NaN" not in str(e): raise    # information, so the sound answer is the vacuous bound
                n = VOCAB if C is None else C.shape[1]
                lb, ub = torch.full((1, n), -float("inf")), torch.full((1, n), float("inf"))
        return lb[0], ub[0]

    # ---------------- abstract top-k ----------------
    def candidates(self, S_in, S_out, q):
        if self.backend == "tibp":
            return self._candidates_tibp(S_in, S_out)
        return self._candidates_lirpa(S_in, S_out, q)

    def _candidates_tibp(self, S_in, S_out):
        from tibp import embed_box, margins_box
        L, n = len(S_in), len(S_out)
        pos, seg = CharLM.layout(L, n)
        sets = [torch.tensor([BOS])] + [torch.from_numpy(np.flatnonzero(S)) for S in S_in] + \
               [torch.tensor([SEP])] + [torch.from_numpy(np.flatnonzero(S)) for S in S_out]
        with torch.no_grad():
            c, r = embed_box(self.lm, sets, pos, seg)
            mlb, _, _ = margins_box(self.lm, c, r)
        self.stats["bound_calls"] += 1
        m = mlb[0].clone(); m.fill_diagonal_(-1.0)
        above = (m > self.tau).sum(0)                            # above[t] = #{j : lb(z_j - z_t) > tau}
        excl = torch.nonzero(above >= self.k).flatten()
        if len(excl):                                            # smallest margin any exclusion relies on
            kth = torch.topk(m[:, excl], self.k, dim=0).values[-1]
            self.min_used_margin = min(self.min_used_margin, float(kth.min()))
        return [int(t) for t in torch.nonzero(above < self.k).flatten()]

    def _candidates_lirpa(self, S_in, S_out, q):
        EL, EU = self.box(S_in, S_out)
        V = self.vertices(S_in, S_out) if self.domain == "hull" else None
        lb, ub = self._bounds(EL, EU, V=V)
        above = (lb[None, :] > ub[:, None]).sum(1)                   # above[t] = #{j: lb_j > ub_t}
        cand = [int(t) for t in torch.nonzero(above < self.k).flatten()]
        if self.rule == "naive" or not cand: return cand
        # margin refinement: property-relevant tokens first, then the rest while budget allows
        relevant = [t for t in cand if self.mon.step(q, t) != 0]
        others = [t for t in cand if self.mon.step(q, t) == 0]
        M = min(self.k + self.extra, VOCAB - 1)
        order = torch.argsort(lb, descending=True).tolist()
        todo, rows = [], 0
        for t in relevant + others:
            if t in relevant or rows + M <= self.row_cap:
                todo.append(t); rows += M
        if not todo: return cand
        C, owner = [], []
        for t in todo:
            comp = [j for j in order if j != t][:M]
            for j in comp:
                r = torch.zeros(VOCAB); r[j] = 1.0; r[t] = -1.0
                C.append(r); owner.append(t)
        mlb, _ = self._bounds(EL, EU, torch.stack(C)[None], V=V)
        excluded = set()
        owner = np.array(owner); mlb = mlb.numpy()
        for t in todo:
            if (mlb[owner == t] > 0).sum() >= self.k: excluded.add(t)
        return [t for t in cand if t not in excluded]

    # ---------------- abstract decoding in product with the monitor ----------------
    def run(self, S_in, L_out, mode="join", time_limit=600, node_limit=20000):
        """Returns dict(verdict in {VERIFIED, ALARM, TIMEOUT}, ...)."""
        t0 = time.time()
        frontier = {(0, ()): None}          # (q, S_out as tuple of frozensets) -> parent info
        nodes = 0
        for step in range(L_out):
            new = {}
            for (q, S_out_key) in frontier:
                if time.time() - t0 > time_limit or nodes >= node_limit:
                    return dict(verdict="TIMEOUT", step=step, nodes=nodes, secs=time.time() - t0, **self.stats)
                S_out = [self._mask(s) for s in S_out_key]
                cand = self.candidates(S_in, S_out, q)
                nodes += 1
                for t in cand:
                    q2 = self.mon.step(q, t)
                    if q2 == Monitor.BAD:
                        return dict(verdict="ALARM", step=step, nodes=nodes, token=t,
                                    secs=time.time() - t0, **self.stats)
                    if mode == "tree":
                        new[(q2, S_out_key + (frozenset([t]),))] = None
                    elif q2 not in new:                               # partitioned join on q
                        new[q2] = [list(S_out_key), {t}]
                    else:
                        new[q2][0] = [a | b for a, b in zip(new[q2][0], S_out_key)]
                        new[q2][1].add(t)
            if mode == "tree":
                frontier = new
            else:
                frontier = {(q2, tuple(prev) + (frozenset(last),)): None for q2, (prev, last) in new.items()}
        return dict(verdict="VERIFIED", nodes=nodes, secs=time.time() - t0,
                    final_states=len(frontier), **self.stats)

    @staticmethod
    def _mask(s):
        m = np.zeros(VOCAB, bool); m[list(s)] = True; return m

def full_ascii_spec(L):
    return [ASCII.copy() for _ in range(L)]
