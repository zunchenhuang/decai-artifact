"""Enumeration-backed soundness check on the specification-conformance models (battery properties)."""
import json, random, warnings, torch; warnings.filterwarnings("ignore"); torch.set_num_threads(int(__import__("os").environ.get("THREADS", "1")))
import soundness_check2 as S2, baselines as B
from tasks import Monitor; from analysis import Analyzer; from run_sweep import load; from battery import BATTERY
rng = random.Random(31); out = open("results/soundness_conf.jsonl", "a")
for task in ("slug", "nopath", "shellsafe"):
    lm = load(task + "_conf_s0")
    props = list(BATTERY[task]["P"].items()) + list(BATTERY[task]["N"].items())
    for i in range(10):
        name, fb = rng.choice(props); mon = Monitor(fb)
        L = rng.choice([2, 3, 4]); k = rng.choice([kk for kk in (1, 2, 3, 5, 10, 50) if kk ** L <= 6000])
        S_in = S2.random_spec(rng, L, fb, trigger=False)
        truth = B.enumerate_exact(lm, mon, S_in, k, L, max_inputs=10**6, batch=max(1, 6000 // k ** L), time_limit=300)
        r = Analyzer(lm, mon, k, backend="tibp").run(S_in, L, mode="join", time_limit=15)
        rec = dict(model=task + "_conf_s0", prop=name, L=L, k=k, truth=truth["result"], verdict=r["verdict"],
                   sound=not (r["verdict"] == "VERIFIED" and truth["result"] == "UNSAFE"))
        out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
