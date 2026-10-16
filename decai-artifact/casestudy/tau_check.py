import os
import sys, json, time, torch
sys.path.insert(0, os.path.join(os.getcwd(), "src")); sys.path.insert(0, ".")
import casestudy_nl2sh_v2 as V, casestudy_nl2sh as CS, analysis
from tasks import Monitor
torch.set_num_threads(1)
V.ORACLE = sys.argv[2]; lm, _ = V.load(f"models/{sys.argv[1]}.pt"); ks = [int(x) for x in sys.argv[3].split(",")]
ver = tot = 0; mins = []
for t in range(4):
    for n in (4, 8):
        for k in ks:
            S, Lout = CS.spec(t, n); an = analysis.Analyzer(lm, Monitor(CS.PROPS["META"]), k, backend="tibp"); an.tau = 0.1
            r = an.run(S, Lout, mode="join", time_limit=600); tot += 1; ver += r["verdict"] == "VERIFIED"
            if r["verdict"] == "VERIFIED": mins.append(an.min_used_margin)
print(sys.argv[1], sys.argv[2], f"tau=0.1: verified {ver}/{tot}, smallest margin used {min(mins) if mins else None}", flush=True)
with open("results/cs_tau.jsonl", "a") as f:            # read by gen_cs_tex.py (daggers in Table 3)
    f.write(json.dumps(dict(model=sys.argv[1], oracle=sys.argv[2], ks=ks, tau=0.1, verified=ver, total=tot,
                            min_margin=min(mins) if mins else None)) + "\n")
