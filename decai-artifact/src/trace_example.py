"""Trace the partitioned-join analysis on the running example (slug, L=4, full ASCII)."""
import sys, json, warnings, numpy as np, torch; warnings.filterwarnings("ignore"); torch.set_num_threads(int(__import__("os").environ.get("THREADS", "1"))); torch.set_grad_enabled(False)
from run_sweep import load
from tasks import Monitor, BOS, SEP, VOCAB
from model import CharLM
from tibp import embed_box, margins_box
DOT = ord('.')
def cls(s):
    s = set(s); al = {c for c in range(128) if chr(c).isalnum()}
    lab = []
    if s & {ord(c) for c in 'abcdefghijklmnopqrstuvwxyz0123456789'} : lab.append(f"{len(s & al)} alnum")
    for c in '._-': 
        if ord(c) in s: lab.append(repr(c))
    rest = s - al - {ord(c) for c in '._-'}
    if rest: lab.append(f"+{len(rest)} other")
    return ", ".join(lab)
def trace(name, k, L=4):
    lm = load(name); mon = Monitor([".."]); S_in = [torch.arange(128)] * L
    R = {0: []}; rows = []
    for j in range(L):
        N = {}
        for q, S_out in R.items():
            pos, seg = CharLM.layout(L, len(S_out))
            sets = [torch.tensor([BOS])] + S_in + [torch.tensor([SEP])] + [torch.tensor(sorted(s)) for s in S_out]
            c, r = embed_box(lm, sets, pos, seg); m = margins_box(lm, c, r)[0][0].clone(); m.fill_diagonal_(-1e9)
            cand = [t for t in range(VOCAB) if int((m[:, t] > 0).sum()) < k]
            kth_dot = float(torch.topk(m[:, DOT], k).values[-1])          # k-th best certified margin over '.'
            rows.append(dict(step=j + 1, q=q, n_cand=len(cand), cand=cls(cand), dot_candidate=DOT in cand, kth_margin_dot=round(kth_dot, 2)))
            for t in cand:
                q2 = mon.step(q, t)
                if q2 == Monitor.BAD:
                    rows.append(dict(step=j + 1, q=q, verdict="ALARM", token=chr(t))); return rows
                prev = N.get(q2, [set() for _ in S_out] + [set()])
                N[q2] = [a | b for a, b in zip(prev[:-1], S_out)] + [prev[-1] | {t}]
        R = N
    rows.append(dict(verdict="VERIFIED")); return rows
if __name__ == "__main__":
    out = {}
    for name in ("slug_cert", "slug"):
        out[name] = trace(name, 2)
        for r in out[name]: print(name, r)
    json.dump(out, open("results/example_trace.json", "w"), indent=1)
