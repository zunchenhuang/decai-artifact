"""Enumeration-backed soundness check for the scaled models (standard + certified)."""
import json, random, time, warnings, torch; warnings.filterwarnings("ignore"); torch.set_num_threads(int(__import__("os").environ.get("THREADS", "1")))
import soundness_check2 as S2, baselines as B
from tasks import TASKS, Monitor
from analysis import Analyzer
from run_sweep import load
rng = random.Random(21); out = open("results/big_soundness.jsonl", "a")
fb = TASKS["slug"]["forbidden"]; mon = Monitor(fb)
for name in ("slug_big", "slug_big_cert"):
    lm = load(name)
    for i in range(12):
        L = rng.choice([2, 3, 4]); k = rng.choice([kk for kk in (1, 2, 3, 5, 10, 50, 100) if kk ** L <= 6000])
        S_in = S2.random_spec(rng, L, fb, trigger=False)
        truth = B.enumerate_exact(lm, mon, S_in, k, L, max_inputs=10**6, batch=max(1, 6000 // k ** L), time_limit=600)
        r = Analyzer(lm, mon, k, backend="tibp").run(S_in, L, mode="join", time_limit=15)
        rec = dict(model=name, i=i, L=L, k=k, spec_size=truth["total"], truth=truth["result"], verdict=r["verdict"],
                   sound=not (r["verdict"] == "VERIFIED" and truth["result"] == "UNSAFE"), secs=round(r["secs"], 3))
        out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
