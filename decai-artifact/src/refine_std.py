"""Refinement on STANDARD (uncertified) models over template specs."""
import json, warnings, numpy as np, torch; warnings.filterwarnings("ignore"); torch.set_num_threads(int(__import__("os").environ.get("THREADS", "1")))
from run_sweep import load; from tasks import Monitor, VOCAB; from refine import verify_refine
def S(chars): m = np.zeros(VOCAB, bool); m[[ord(c) for c in chars]] = True; return m
A17 = 'a.b_-/!~ Z9,:;?@#'
f = open("results/refine_std.jsonl", "a")
cases = [("slug", [".."], "a??b", [S('a'), S(A17), S(A17), S('b')]),
         ("slug", [".."], "x???", [S('x'), S(A17), S(A17), S(A17)]),
         ("nopath", ["/"], "a??b", [S('a'), S(A17), S(A17), S('b')]),
         ("shellsafe", [";"], "a??b", [S('a'), S(A17), S(A17), S('b')])]
for model, fb, desc, spec in cases:
    lm = load(model)
    for k in (1, 2):
        r = verify_refine(lm, Monitor(fb), spec, k, budget=3000, time_limit=120)
        rec = dict(model=model, prop=fb, spec=desc, alphabet=A17, k=k, **r)
        f.write(json.dumps(rec) + "\n"); f.flush(); print(json.dumps(rec), flush=True)
