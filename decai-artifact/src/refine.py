import time, numpy as np, torch
from analysis import Analyzer
import baselines as B
from model import next_logits

def _concrete_witness(lm, mon, x, k, L_out):
    """Exact check of one input: returns a violating output (list of tokens) or None."""
    X = torch.tensor([x]); stack = [([], 0)]
    while stack:
        ys, q = stack.pop()
        if len(ys) == L_out: continue
        z = next_logits(lm, X, torch.tensor([ys], dtype=torch.long).reshape(1, len(ys)))[0]
        for t in torch.topk(z, k).indices.tolist():
            q2 = mon.step(q, t)
            if q2 == mon.BAD: return ys + [t]
            stack.append((ys + [t], q2))
    return None

def verify_refine(lm, mon, S_in, k, L_out=None, budget=400, time_limit=300):
    L_out = L_out or len(S_in); t0 = time.time()
    stats = dict(abstract_runs=0, leaves_verified=0, concrete_checks=0, max_depth=0)
    def rec(S, depth):
        stats["max_depth"] = max(stats["max_depth"], depth)
        if stats["abstract_runs"] >= budget or time.time() - t0 > time_limit: return ("UNKNOWN", None)
        stats["abstract_runs"] += 1
        r = Analyzer(lm, mon, k, backend="tibp").run(S, L_out, mode="join", time_limit=60)
        if r["verdict"] == "VERIFIED":
            stats["leaves_verified"] += 1; return ("VERIFIED", None)
        sizes = [int(s.sum()) for s in S]
        if max(sizes) == 1:
            stats["concrete_checks"] += 1
            x = [int(np.flatnonzero(s)[0]) for s in S]
            y = _concrete_witness(lm, mon, x, k, L_out)
            return ("COUNTEREXAMPLE", (x, y)) if y is not None else ("VERIFIED", None)
        i = int(np.argmax(sizes)); idx = np.flatnonzero(S[i]); half = len(idx) // 2
        verdict = "VERIFIED"
        for part in (idx[:half], idx[half:]):
            m = np.zeros_like(S[i]); m[part] = True
            v, w = rec(S[:i] + [m] + S[i + 1:], depth + 1)
            if v == "COUNTEREXAMPLE": return (v, w)
            if v == "UNKNOWN": verdict = "UNKNOWN"
        return (verdict, None)
    v, w = rec(list(S_in), 0)
    out = dict(verdict=v, secs=round(time.time() - t0, 2), **stats)
    if w is not None:
        x, y = w
        out["input"] = "".join(chr(c) if 32 <= c < 127 else f"\\x{c:02x}" for c in x)
        out["output"] = "".join(chr(c) if 32 <= c < 127 else f"\\x{c:02x}" for c in y)
    return out
