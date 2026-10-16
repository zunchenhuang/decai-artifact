"""Soundness stress test with k close to |V| (where certificates trained for K=50 must fail)."""
import json, random, warnings, torch; warnings.filterwarnings("ignore")
torch.set_num_threads(int(__import__("os").environ.get("THREADS", "1")))
import soundness_check2 as S2, baselines as B
from tasks import TASKS, Monitor; from analysis import Analyzer; from run_sweep import load
rng = random.Random(5); out = open("results/soundness_v2.jsonl", "a")
for task in TASKS:
    lm = load(task + "_cert"); fb = TASKS[task]["forbidden"]; mon = Monitor(fb)
    for k in (50, 100, 125):
        for i in range(3):
            S_in = S2.random_spec(rng, 2, fb, trigger=(task == "slug_poisoned"))
            truth = B.enumerate_exact(lm, mon, S_in, k, 2, max_inputs=10**6, batch=1, time_limit=120)
            r = Analyzer(lm, mon, k, backend="tibp").run(S_in, 2, mode="join", time_limit=15)
            rec = dict(model=task + "_cert", i=100 + i, L=2, k=k, mode="join", spec_size=truth["total"], truth=truth["result"],
                       verdict=r["verdict"], sound=not (r["verdict"] == "VERIFIED" and truth["result"] == "UNSAFE"), secs=round(r["secs"], 3))
            out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
