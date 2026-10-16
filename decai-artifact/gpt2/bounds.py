import math, time, itertools, collections, torch, torch.nn.functional as F

# ============================================================== interval primitives
def lin(c, r, W, b=None):
    """Affine map of a box; exact coordinatewise.  W: (out, in)."""
    return c @ W.T + (b if b is not None else 0.0), r @ W.abs().T

NORM_L2 = True

def norm_box(c, r, kind, w, b, eps):
    """Sound box for LayerNorm ('ln') or RMSNorm ('rms') over the last dimension; exact if r == 0."""
    d = c.shape[-1]
    if kind == "ln":
        nc = c - c.mean(-1, keepdim=True)
        nr = r * (1 - 2.0 / d) + r.sum(-1, keepdim=True) / d               # |I - 11^T/d| r
    else:
        nc, nr = c, r
    lo, hi = nc - nr, nc + nr
    lb2 = torch.where(lo > 0, lo * lo, torch.where(hi < 0, hi * hi, torch.zeros_like(lo)))
    ub2 = torch.maximum(lo * lo, hi * hi)
    # (a) coordinatewise bounds on mean(n^2)
    ms_lo, ms_hi = lb2.mean(-1, keepdim=True), ub2.mean(-1, keepdim=True)
    if NORM_L2:
        # (b) l2 bounds: n = n_c + P e with ||P e||_2 <= ||e||_2 <= ||r||_2 (centering P is a projection)
        rn = r.norm(dim=-1, keepdim=True); cn = nc.norm(dim=-1, keepdim=True)
        sym = rn > 0                                                            # concrete rows stay bit-exact
        ms_lo = torch.where(sym, torch.maximum(ms_lo, (cn - rn).clamp_min(0) ** 2 / d), ms_lo)
        ms_hi = torch.where(sym, torch.minimum(ms_hi, (cn + rn) ** 2 / d), ms_hi)
    sl = torch.sqrt(ms_lo + eps)
    su = torch.sqrt(ms_hi + eps)
    ylo = torch.minimum(lo / sl, lo / su)
    yhi = torch.maximum(hi / sl, hi / su)
    cap = math.sqrt(d)                                                      # |n_i| / sqrt(mean n^2 + eps) <= sqrt(d)
    ylo, yhi = ylo.clamp(-cap, cap), yhi.clamp(-cap, cap)
    yc, yr = (ylo + yhi) / 2, (yhi - ylo) / 2
    oc = yc * w + (b if b is not None else 0.0)
    return oc, yr * w.abs()

def _gelu_erf(x): return F.gelu(x)
def _gelu_tanh(x): return F.gelu(x, approximate="tanh")

def _minimum(f):
    xs = torch.linspace(-4.0, 0.0, 400001, dtype=torch.float64)
    ys = f(xs); i = int(ys.argmin())
    return float(xs[i]), float(ys[i]) - 1e-6                               # tiny slack keeps it a lower bound

ACT_MIN = {name: _minimum(f) for name, f in (("gelu", _gelu_erf), ("gelu_tanh", _gelu_tanh))}
ACT_FN = {"gelu": _gelu_erf, "gelu_tanh": _gelu_tanh, "relu": F.relu}

def act_interval(lo, hi, name):
    """Interval of a GELU (decreasing then increasing) or ReLU (monotone) over [lo, hi]."""
    f = ACT_FN[name]; flo, fhi = f(lo), f(hi)
    if name == "relu":
        return flo, fhi
    xs, ymin = ACT_MIN[name]; delta = 1e-4                                   # >> grid spacing 1e-5 of the minimiser
    straddle = (lo < xs + delta) & (hi > xs - delta) & (hi > lo)              # degenerate intervals are exact
    low = torch.where(straddle, torch.minimum(torch.minimum(flo, fhi), torch.full_like(lo, ymin)), torch.minimum(flo, fhi))
    return low, torch.maximum(flo, fhi)

def softmax_bounds(s_lo, s_hi, valid):
    NEG = -1e9
    s_lo = s_lo.masked_fill(~valid, NEG); s_hi = s_hi.masked_fill(~valid, NEG)
    Tk = s_lo.shape[-1]; eye = torch.eye(Tk, dtype=torch.bool)
    other_hi = s_hi[..., None, :].expand(*s_hi.shape, Tk).masked_fill(eye, NEG)
    other_lo = s_lo[..., None, :].expand(*s_lo.shape, Tk).masked_fill(eye, NEG)
    w_lo = torch.sigmoid(s_lo - torch.logsumexp(other_hi, -1)).masked_fill(~valid, 0.0)
    w_hi = torch.sigmoid(s_hi - torch.logsumexp(other_lo, -1)).masked_fill(~valid, 0.0)
    return w_lo, w_hi

def knapsack_avg(w_lo, w_hi, v, maximize):
    """Exact max (min) of sum_i w_i v_i s.t. w in [w_lo, w_hi], sum w = 1 (Lemma 3 of the paper)."""
    sgn = 1.0 if maximize else -1.0
    vv = sgn * v
    base = torch.einsum("...qk,...kd->...qd", w_lo, vv)
    R = (1.0 - w_lo.sum(-1, keepdim=True)).clamp_min(0.0)
    order = torch.argsort(vv, dim=-2, descending=True)
    v_sorted = torch.gather(vv, -2, order)
    cap = (w_hi - w_lo).clamp_min(0.0)
    D = vv.shape[-1]
    cap_e = cap[..., None].expand(*cap.shape, D)
    idx = order[..., None, :, :].expand(*cap.shape, D)
    cap_sorted = torch.gather(cap_e, -2, idx)
    before = torch.cumsum(cap_sorted, -2) - cap_sorted
    alloc = torch.minimum(cap_sorted, (R[..., None] - before).clamp_min(0.0))
    val = base + (alloc * v_sorted[..., None, :, :]).sum(-2)
    left = (R - alloc.sum(-2)).clamp_min(0.0)
    val = val + left * vv.amax(-2)[..., None, :]
    return sgn * val

ROWS_MAX = 64      # larger token sets of one position are evaluated as a single box row

# ============================================================== architecture adapters
class Adapter:
    """View of a Hugging Face GPT-2 model: embeddings, blocks, final norm, head."""
    def __init__(self, model, live=False):
        """live=True keeps the parameters attached to autograd (for certified training); the default detaches."""
        self.model = model.eval(); cfg = model.config; self.family = cfg.model_type
        P = dict(model.named_parameters())
        g = lambda k: (P[k] if live else P[k].detach()).float() if k in P else None
        self.ramp = 1.0                                                          # < 1 only during training warm-up
        self.blocks = []
        if self.family == "gpt2":
            self.E = g("transformer.wte.weight"); self.Pos = g("transformer.wpe.weight"); self.pos_offset = 0
            H = cfg.n_head; d = cfg.n_embd
            for i in range(cfg.n_layer):
                p = f"transformer.h.{i}."
                Wqkv = g(p + "attn.c_attn.weight").T; bqkv = g(p + "attn.c_attn.bias")
                self.blocks.append(dict(
                    ln1=("ln", g(p + "ln_1.weight"), g(p + "ln_1.bias"), cfg.layer_norm_epsilon),
                    ln2=("ln", g(p + "ln_2.weight"), g(p + "ln_2.bias"), cfg.layer_norm_epsilon),
                    q=(Wqkv[:d], bqkv[:d]), k=(Wqkv[d:2 * d], bqkv[d:2 * d]), v=(Wqkv[2 * d:], bqkv[2 * d:]),
                    o=(g(p + "attn.c_proj.weight").T, g(p + "attn.c_proj.bias")),
                    H=H, dh=d // H, scale=1 / math.sqrt(d // H),
                    mlp=("act", {"gelu_new": "gelu_tanh", "gelu": "gelu", "relu": "relu", "gelu_pytorch_tanh": "gelu_tanh"}[cfg.activation_function],
                         (g(p + "mlp.c_fc.weight").T, g(p + "mlp.c_fc.bias")),
                         (g(p + "mlp.c_proj.weight").T, g(p + "mlp.c_proj.bias")))))
            self.lnf = ("ln", g("transformer.ln_f.weight"), g("transformer.ln_f.bias"), cfg.layer_norm_epsilon)
            self.G, self.gb = self.E, None
        else:
            raise NotImplementedError(f"model family {self.family}: this artifact supports GPT-2 only")
        self.V, self.d = self.G.shape

    # ---------------------------------------------------------- box forward pass (token-disjunctive)
    # The abstract value is Cartesian (a token set per position), but the transformer is evaluated on one
    # row per (position, token): row (p, t) carries a box over the states of position p in the contexts whose
    # token at p is t.  When a later row attends to position p, it sees the hull of the rows of p; a row
    # attends to itself exactly.  This avoids hulling unrelated token embeddings before a normaliser, which
    # is what makes a plain box over a token set vacuous after the first LayerNorm.  Sound by the same
    # argument as the plain box pass: every concrete context selects one row per position, and each row's
    # box contains that position's state in every context that selects it.
    @torch.no_grad()
    def final_rows(self, sets, trace=None):
        """sets: per position a LongTensor of token ids.  Returns boxes (c, r) of shape (n, d), one per token
        of the last position, of the final normalised hidden state."""
        return self.final_rows_grad(sets, trace)

    def final_rows_grad(self, sets, trace=None):
        """As final_rows, differentiable (used by certified training with live=True)."""
        T = len(sets)
        # a position with more than ROWS_MAX tokens is one row holding the box hull of their embeddings
        rows = [s.flatten() if len(s.flatten()) <= ROWS_MAX else None for s in sets]
        cs, rs, ps = [], [], []
        for i, (s, rw) in enumerate(zip(sets, rows)):
            if rw is not None:
                cs.append(self.E[rw]); rs.append(torch.zeros(len(rw), self.d)); ps.append(torch.full((len(rw),), i))
            else:
                Es = self.E[s.flatten()]; lo, hi = Es.min(0).values, Es.max(0).values
                cs.append(((lo + hi) / 2)[None]); rs.append(((hi - lo) / 2)[None]); ps.append(torch.tensor([i]))
        c, r, pos = torch.cat(cs), torch.cat(rs), torch.cat(ps)
        if self.Pos is not None:
            c = c + self.Pos[self.pos_offset + pos]
        valid = torch.arange(T)[None, :] <= pos[:, None]                             # (N, T) causal over positions
        for li, blk in enumerate(self.blocks):
            c, r = self._block(blk, c, r, valid, pos, T)
            if trace is not None:
                trace.append(dict(layer=li, pos=pos.tolist(), r_mean=r.mean(-1).tolist(),
                                  c_norm=c.norm(dim=-1).tolist(), r_norm=r.norm(dim=-1).tolist()))
        last = pos == T - 1
        return norm_box(c[last], r[last], *self.lnf)

    def final_box(self, sets, trace=None):
        """Hull of final_rows (used by the tests)."""
        c, r = self.final_rows(sets, trace)
        lo, hi = (c - r).min(0).values, (c + r).max(0).values
        return (lo + hi) / 2, (hi - lo) / 2

    def _block(self, blk, c, r, valid, pos, T):
        H, dh = blk["H"], blk["dh"]; N = c.shape[0]; ar = torch.arange(N)
        xc, xr = norm_box(c, r, *blk["ln1"])
        def proj(key, nh):
            W, b = blk[key]; pc, pr = lin(xc, xr, W, b)
            return pc.reshape(N, nh, dh).transpose(0, 1), pr.reshape(N, nh, dh).transpose(0, 1)
        qc, qr = proj("q", H); kc, kr = proj("k", H); vc, vr = proj("v", H)
        def hull(xc_, xr_):                                                          # rows -> positions
            idx = pos[None, :, None].expand_as(xc_)
            lo = torch.full((H, T, dh), float("inf")).scatter_reduce(1, idx, xc_ - xr_, "amin")
            hi = torch.full((H, T, dh), float("-inf")).scatter_reduce(1, idx, xc_ + xr_, "amax")
            return (lo + hi) / 2, ((hi - lo) / 2).clamp_min(0) * self.ramp
        Kc, Kr = hull(kc, kr); Vc, Vr = hull(vc, vr)
        sc = qc @ Kc.transpose(-1, -2) * blk["scale"]                                 # (H, N, T)
        sr = (qc.abs() @ Kr.transpose(-1, -2) + qr @ Kc.abs().transpose(-1, -2) + qr @ Kr.transpose(-1, -2)) * blk["scale"]
        own_c = (qc * kc).sum(-1) * blk["scale"]                                      # (H, N): a row attends to itself exactly
        own_r = (qc.abs() * kr + qr * kc.abs() + qr * kr).sum(-1) * blk["scale"]
        sc[:, ar, pos] = own_c; sr[:, ar, pos] = own_r
        w_lo, w_hi = softmax_bounds(sc - sr, sc + sr, valid)                         # (H, N, T)
        Vcr = Vc[:, None].expand(H, N, T, dh).clone(); Vrr = Vr[:, None].expand(H, N, T, dh).clone()
        Vcr[:, ar, pos] = vc; Vrr[:, ar, pos] = vr
        wl, wh = w_lo[:, :, None, :], w_hi[:, :, None, :]                             # (H, N, 1, T)
        z_hi = knapsack_avg(wl, wh, Vcr + Vrr, True)[:, :, 0]; z_lo = knapsack_avg(wl, wh, Vcr - Vrr, False)[:, :, 0]
        exact = ((w_hi - w_lo).abs().amax(-1) == 0) & ((Vrr.amax(-1) * valid).amax(-1) == 0)   # (H, N)
        if exact.any():
            ze = torch.einsum("hnk,hnkd->hnd", w_lo, Vcr)
            z_hi = torch.where(exact[..., None], ze, z_hi); z_lo = torch.where(exact[..., None], ze, z_lo)
        zc = ((z_hi + z_lo) / 2).transpose(0, 1).reshape(N, H * dh)
        zr = ((z_hi - z_lo) / 2).clamp_min(0).transpose(0, 1).reshape(N, H * dh)
        oc, orr = lin(zc, zr, *blk["o"])
        c, r = c + oc, r + orr
        xc, xr = norm_box(c, r, *blk["ln2"])
        m = blk["mlp"]
        hc, hr = lin(xc, xr, *m[2]); lo, hi = act_interval(hc - hr, hc + hr, m[1])
        gc, gr = lin((lo + hi) / 2, (hi - lo) / 2, *m[3])
        return c + gc, r + gr

    # ---------------------------------------------------------- margin oracle and candidate set
    @torch.no_grad()
    def candidates(self, sets, k, M=64, chunk=2048, max_pairwise=60000):
        """Sound abstract top-k: every token that is among the admissible top-k for some context in the
        concretisation of `sets` is returned.  Returns (LongTensor of candidates, stats dict)."""
        cs, rs = self.final_rows(sets)
        outs, stats = [], dict(width=0.0, after_interval=0, after_pairwise=0)
        for c, r in zip(cs, rs):
            o, st = self._candidates_row(c, r, k, M, chunk, max_pairwise)
            outs.append(o); stats["width"] = max(stats["width"], st["width"])
            stats["after_interval"] += st["after_interval"]; stats["after_pairwise"] += st.get("after_pairwise", st["after_interval"])
        return torch.unique(torch.cat(outs)), stats

    def _candidates_row(self, c, r, k, M, chunk, max_pairwise):
        G = self.G; zc = G @ c + (self.gb if self.gb is not None else 0.0); zr = G.abs() @ r
        zl, zu = zc - zr, zc + zr
        # (1) interval certificates: #{j : zl_j > zu_t}
        srt = torch.sort(zl).values
        cnt_int = self.V - torch.searchsorted(srt, zu, right=True)                # zl_t <= zu_t, so j = t never counts
        R = torch.nonzero(cnt_int < k).flatten()
        stats = dict(width=float((zu - zl).mean()), after_interval=int(len(R)))
        if len(R) == 0 or len(R) > max_pairwise:
            return R, stats
        # (2) exact pairwise margins against the top-M competitors (by lower bound)
        J = torch.topk(zl, min(M, self.V)).indices
        GJ = G[J]; gJ = self.gb[J] if self.gb is not None else torch.zeros(len(J))
        inJ_cert = torch.zeros(len(R), dtype=torch.long)
        pair = torch.zeros(len(R), dtype=torch.long)
        for s in range(0, len(R), chunk):
            t = R[s:s + chunk]; Gt = G[t]; gt = self.gb[t] if self.gb is not None else torch.zeros(len(t))
            D = GJ[:, None, :] - Gt[None, :, :]                                   # (M, n, d)
            mlb = D @ c - D.abs() @ r + (gJ[:, None] - gt[None, :])
            mlb = mlb.masked_fill(J[:, None] == t[None, :], -1.0)
            pair[s:s + chunk] = (mlb > 0).sum(0)
            inJ_cert[s:s + chunk] = (zl[J][:, None] > zu[t][None, :]).sum(0)
        total = cnt_int[R] - inJ_cert + pair                                      # pairwise replaces interval on J
        out = R[total < k]
        stats["after_pairwise"] = int(len(out))
        return out, stats

    @torch.no_grad()
    def logits(self, ids):
        out = self.model(torch.tensor([ids])).logits[0, -1].float()
        return out

# ============================================================== byte-level monitor (Aho-Corasick)
class ByteMonitor:
    BAD = -1
    def __init__(self, patterns):
        pats = [p.encode() if isinstance(p, str) else p for p in patterns]
        goto, fail, out = [{}], [0], [False]
        for p in pats:
            s = 0
            for ch in p:
                if ch not in goto[s]:
                    goto.append({}); fail.append(0); out.append(False); goto[s][ch] = len(goto) - 1
                s = goto[s][ch]
            out[s] = True
        q = collections.deque()
        for ch, s in goto[0].items(): q.append(s)
        while q:
            u = q.popleft()
            for ch, v in goto[u].items():
                q.append(v); f = fail[u]
                while f and ch not in goto[f]: f = fail[f]
                fail[v] = goto[f].get(ch, 0) if goto[f].get(ch, 0) != v else 0
                out[v] = out[v] or out[fail[v]]
        self.n = len(goto); self.delta = []
        for s in range(self.n):
            row = []
            for ch in range(256):
                t = s
                while t and ch not in goto[t]: t = fail[t]
                t = goto[t].get(ch, 0)
                row.append(self.BAD if out[t] else t)
            self.delta.append(row)
    def run(self, q, bs):
        for ch in bs:
            q = self.delta[q][ch]
            if q == self.BAD: return self.BAD
        return q

class TokenMonitor:
    """Monitor composed with the detokeniser: trans[q][t] = delta*(q, bytes(t)); special tokens end a branch."""
    END = -2
    def __init__(self, patterns, token_bytes, special_ids):
        self.m = ByteMonitor(patterns); V = len(token_bytes)
        self.trans = torch.empty(self.m.n, V, dtype=torch.long)
        for q in range(self.m.n):
            self.trans[q] = torch.tensor([self.END if t in special_ids else self.m.run(q, token_bytes[t]) for t in range(V)])

# ============================================================== tokenizer adapter
def token_bytes_of(tok, V):
    """Exact bytes of every token id < V of a byte-level BPE tokenizer (GPT-2); padding ids map to b''."""
    vocab = tok.convert_ids_to_tokens(list(range(min(V, len(tok)))))
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    dec = {chr(c): b for b, c in zip(bs, cs)}
    out = [bytes(dec[ch] for ch in v if ch in dec) if v else b"" for v in vocab]
    out += [b""] * (V - len(out))
    return out

# ============================================================== Algorithm 2 (abstract decoding) and exact checks
@torch.no_grad()
def abstract_decode(ad, mon, prefix_sets, k, L_out, M=64, time_limit=600):
    """prefix_sets: list of LongTensors (prompt with symbolic slots).  Returns a verdict dict."""
    t0 = time.time(); R = {0: []}; calls = 0; widths = []
    for j in range(L_out):
        N = {}
        for q, S in R.items():
            C, st = ad.candidates(prefix_sets + [torch.tensor(sorted(s)) for s in S], k, M=M)
            calls += 1; widths.append(st["width"])
            nxt = mon.trans[q][C]
            if (nxt == ByteMonitor.BAD).any():
                bad = C[nxt == ByteMonitor.BAD]
                return dict(verdict="ALARM", step=j, calls=calls, secs=time.time() - t0, n_cand=int(len(C)),
                            bad_example=int(bad[0]), width=widths[-1])
            for q2 in torch.unique(nxt[nxt >= 0]).tolist():
                toks = set(C[nxt == q2].tolist())
                prev = N.get(q2)
                N[q2] = [a | b for a, b in zip(prev[:-1], S)] + [prev[-1] | toks] if prev else [set(s) for s in S] + [toks]
            if time.time() - t0 > time_limit:
                return dict(verdict="TIMEOUT", calls=calls, secs=time.time() - t0)
        if not N: break
        R = N
    return dict(verdict="VERIFIED", calls=calls, secs=time.time() - t0, width=max(widths) if widths else 0.0)

def admissible(z, k):
    """Admissible top-k set (Section 2): fewer than k tokens strictly larger."""
    srt = torch.sort(z, descending=True).values
    thr = srt[k - 1]
    return torch.nonzero(z >= thr).flatten()

@torch.no_grad()
def exact_check(ad, mon, ids, k, L_out, batch=64):
    """Exhaustive top-k tree of one concrete prompt in product with the monitor: a violating output or None."""
    frontier = [([], 0)]
    for j in range(L_out):
        nxt = []
        for s in range(0, len(frontier), batch):
            chunk = frontier[s:s + batch]
            inp = torch.tensor([ids + ys for ys, _ in chunk])
            Z = ad.model(inp).logits[:, -1].float()
            for (ys, q), z in zip(chunk, Z):
                for t in admissible(z, k).tolist():
                    q2 = int(mon.trans[q][t])
                    if q2 == ByteMonitor.BAD: return ys + [t]
                    if q2 >= 0: nxt.append((ys + [t], q2))
        frontier = nxt
        if not frontier: break
    return None

def refine(ad, mon, prefix_sets, k, L_out, M=64, budget=400, time_limit=900):
    """Algorithm 3: split the largest slot set; decide singletons exactly."""
    t0 = time.time(); st = dict(abstract_runs=0, exact_checks=0, proved_leaves=0, proved_inputs=0)
    def rec(S):
        if st["abstract_runs"] >= budget or time.time() - t0 > time_limit: return ("UNKNOWN", None)
        sizes = [len(s) for s in S]
        if max(sizes) == 1:                                       # a concrete prompt: decide it exactly
            st["exact_checks"] += 1; ids = [int(s[0]) for s in S]
            y = exact_check(ad, mon, ids, k, L_out)
            return ("COUNTEREXAMPLE", (ids, y)) if y is not None else ("VERIFIED", None)
        st["abstract_runs"] += 1
        r = abstract_decode(ad, mon, S, k, L_out, M=M)
        if r["verdict"] == "VERIFIED":
            st["proved_leaves"] += 1; st["proved_inputs"] += math.prod(len(s) for s in S); return ("VERIFIED", None)
        i = max(range(len(S)), key=lambda i: sizes[i]); h = len(S[i]) // 2; verdict = "VERIFIED"
        for part in (S[i][:h], S[i][h:]):
            v, w = rec(S[:i] + [part] + S[i + 1:])
            if v == "COUNTEREXAMPLE": return (v, w)
            if v == "UNKNOWN": verdict = "UNKNOWN"
        return (verdict, None)
    v, w = rec(list(prefix_sets))
    return dict(verdict=v, secs=time.time() - t0, witness=w, **st)

def enumerate_truth(ad, mon, prefix_sets, k, L_out):
    """Ground truth: exhaustive over all concrete prompts in the concretisation."""
    for combo in itertools.product(*[s.tolist() for s in prefix_sets]):
        if exact_check(ad, mon, list(combo), k, L_out) is not None: return "UNSAFE"
    return "SAFE"

def make_adapter(model):
    return Adapter(model)
