import itertools, math, time, numpy as np, torch
from tasks import VOCAB, BOS, SEP, Monitor
from model import CharLM, next_logits

def topk_set(logits, k): return torch.topk(logits, k, dim=-1).indices

# ---------------- exact per-input check: BFS over the top-k tree, in product with the monitor ----------------
@torch.no_grad()
def violating_inputs(lm, mon, X, k, L_out, max_rows=20000):
    """X: (B,L) inputs. Returns bool (B,) = some top-k path emits a forbidden string."""
    B = X.shape[0]
    idx = torch.arange(B); Y = torch.zeros(B, 0, dtype=torch.long); Q = torch.zeros(B, dtype=torch.long)
    bad = torch.zeros(B, dtype=torch.bool)
    for _ in range(L_out):
        if len(idx) == 0: break
        if len(idx) * k > max_rows: raise MemoryError("tree too large")
        T = topk_set(next_logits(lm, X[idx], Y), k)             # (n,k)
        idx = idx.repeat_interleave(k); Y = torch.cat([Y.repeat_interleave(k, 0), T.reshape(-1, 1)], 1)
        Q = torch.tensor([mon.step(int(q), int(t)) for q, t in zip(Q.repeat_interleave(k), T.reshape(-1))])
        hit = Q == Monitor.BAD
        bad[idx[hit]] = True
        keep = ~hit & ~bad[idx]
        idx, Y, Q = idx[keep], Y[keep], Q[keep]
    return bad

def spec_size(S_in): return math.prod(int(S.sum()) for S in S_in)

@torch.no_grad()
def enumerate_exact(lm, mon, S_in, k, L_out, max_inputs=200000, batch=512, time_limit=600):
    n = spec_size(S_in)
    alph = [np.flatnonzero(S) for S in S_in]
    t0, viol, done = time.time(), 0, 0
    it = itertools.product(*[range(len(a)) for a in alph])   # lazy; same order as np.ndindex, which numpy < 2 cannot build for 128^16
    budget = min(n, max_inputs)
    while done < budget and time.time() - t0 < time_limit:
        chunk = []
        for _ in range(min(batch, budget - done)):
            ix = next(it); chunk.append([alph[i][j] for i, j in enumerate(ix)])
        bad = violating_inputs(lm, mon, torch.tensor(np.array(chunk)), k, L_out)
        viol += int(bad.sum()); done += len(chunk)
        if viol and done < n:          # found a real violation: enumeration can stop early
            return dict(result="UNSAFE", checked=done, total=n, secs=time.time() - t0)
    secs = time.time() - t0
    if done == n:
        return dict(result="UNSAFE" if viol else "SAFE", violating=viol, checked=done, total=n, secs=secs)
    return dict(result="INCOMPLETE", checked=done, total=n, secs=secs,
                projected_secs=secs / max(done, 1) * n)

# ---------------- sampling ----------------
@torch.no_grad()
def sample(lm, mon, S_in, k, L_out, n=2000, batch=500, seed=0, time_limit=120):
    g = torch.Generator().manual_seed(seed); t0 = time.time()
    alph = [torch.from_numpy(np.flatnonzero(S)) for S in S_in]
    tried = 0
    while tried < n and time.time() - t0 < time_limit:
        b = min(batch, n - tried)
        X = torch.stack([a[torch.randint(len(a), (b,), generator=g)] for a in alph], 1)
        Y = torch.zeros(b, 0, dtype=torch.long); Q = torch.zeros(b, dtype=torch.long)
        for _ in range(L_out):
            lg = next_logits(lm, X, Y)
            v, ix = torch.topk(lg, k, dim=-1)
            pick = ix.gather(1, torch.multinomial(torch.softmax(v, -1), 1, generator=g))
            Y = torch.cat([Y, pick], 1)
            Q = torch.tensor([mon.step(int(q), int(t)) for q, t in zip(Q, pick.flatten())])
        hit = (Q == Monitor.BAD).nonzero().flatten()
        tried += b
        if len(hit):
            i = int(hit[0])
            return dict(result="UNSAFE", tried=tried, input=X[i].tolist(), output=Y[i].tolist(),
                        secs=time.time() - t0)
    return dict(result="NONE_FOUND", tried=tried, secs=time.time() - t0)

# ---------------- GCG-style attack ----------------
def _forced_gaps(lm, X_emb_fn, X, patterns, k, L_out):
    """For each input and each (pattern p, start j): outputs before j are greedy, p is forced at j..
    gap = sum_m relu(z_kth - z_{p_m} + 1e-3) at each forced step.  gap == 0  =>  p is emitted under top-k.
    Returns min gap per input (differentiable w.r.t. the embedding path used by X_emb_fn)."""
    B, L = X.shape
    with torch.no_grad():
        Yg = torch.zeros(B, 0, dtype=torch.long)
        for _ in range(L_out):
            Yg = torch.cat([Yg, next_logits(lm, X, Yg).argmax(-1, keepdim=True)], 1)
    best = None
    for p in patterns:
        for j in range(0, L_out - len(p) + 1):
            Y = torch.cat([Yg[:, :j], torch.tensor(p).expand(B, -1)], 1)
            toks = torch.cat([torch.full((B, 1), BOS), X, torch.full((B, 1), SEP), Y[:, :-1]], 1)
            pos, seg = CharLM.layout(L, Y.shape[1] - 1)
            E = X_emb_fn(toks, pos[None].expand(B, -1), seg[None].expand(B, -1))
            logits = lm.body_all(E)[:, L + 1 + j: L + 1 + j + len(p), :]            # steps j..j+|p|-1
            kth = torch.topk(logits, k, dim=-1).values[..., -1]
            tgt = logits.gather(2, torch.tensor(p).expand(B, -1)[..., None])[..., 0]
            gap = torch.relu(kth - tgt + 1e-3).sum(-1)
            best = gap if best is None else torch.minimum(best, gap)
    return best

def replay(lm, mon, x, k, L_out):
    return bool(violating_inputs(lm, mon, torch.tensor([x]), k, L_out)[0])

@torch.no_grad()
def replay_path(lm, mon, x, y, k):
    """Exact check that output y is a top-k run for input x and contains a forbidden string."""
    X = torch.tensor([x]); q = 0
    for j, t in enumerate(y):
        if t not in topk_set(next_logits(lm, X, torch.tensor([y[:j]], dtype=torch.long).reshape(1, j)), k)[0]: return False
        q = mon.step(q, t)
        if q == Monitor.BAD: return True
    return False

@torch.no_grad()
def witness_path(lm, mon, x, patterns, k, L_out):
    X = torch.tensor([x]); Yg = []
    for _ in range(L_out): Yg.append(int(next_logits(lm, X, torch.tensor([Yg], dtype=torch.long).reshape(1, len(Yg))).argmax()))
    for p in patterns:
        for j in range(0, L_out - len(p) + 1):
            y = Yg[:j] + list(p)
            if replay_path(lm, mon, x, y, k): return y
    return None

def attack(lm, mon, patterns, S_in, k, L_out, restarts=4, iters=30, cands=16, seed=0, time_limit=120):
    rng = np.random.default_rng(seed); t0 = time.time()
    allowed = torch.from_numpy(np.stack(S_in))                       # (L, VOCAB) bool
    pats = [[ord(c) for c in p] for p in patterns]
    for r in range(restarts):
        x = torch.tensor([[int(rng.choice(np.flatnonzero(S))) for S in S_in]])
        for it in range(iters):
            if time.time() - t0 > time_limit: return dict(result="NONE_FOUND", secs=time.time() - t0)
            # gradient w.r.t. a one-hot relaxation of the input characters
            oh = torch.nn.functional.one_hot(x[0], VOCAB).float().requires_grad_(True)
            def emb(toks, pos, seg):
                E = lm.embed_parts(toks, pos, seg)
                L = x.shape[1]
                return torch.cat([E[:, :1], (oh @ lm.tok.weight)[None] + lm.pos(pos[:, 1:L + 1]) + lm.seg(seg[:, 1:L + 1]),
                                  E[:, L + 1:]], 1)
            gap = _forced_gaps(lm, emb, x, pats, k, L_out)
            if gap.item() == 0:
                y = witness_path(lm, mon, x[0].tolist(), pats, k, L_out)
                if y is not None:
                    return dict(result="UNSAFE", input=x[0].tolist(), output=y, restart=r, iters=it, secs=time.time() - t0)
            gap.backward()
            g = oh.grad.masked_fill(~allowed, float("inf"))
            # candidate single-character substitutions (GCG): most negative gradient per position
            pos_c = rng.integers(0, x.shape[1], size=cands)
            X = x.repeat(cands, 1)
            for i, pc in enumerate(pos_c):
                top = torch.topk(-g[pc], 8).indices
                X[i, pc] = top[rng.integers(0, len(top))]
            with torch.no_grad():
                gaps = _forced_gaps(lm, lm.embed_parts, X, pats, k, L_out)
            j = int(gaps.argmin())
            if gaps[j] <= gap.item(): x = X[j:j + 1]
    return dict(result="NONE_FOUND", secs=time.time() - t0)
