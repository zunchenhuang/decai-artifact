import json, os, random, sys, time, warnings, zlib, numpy as np, torch
warnings.filterwarnings("ignore"); torch.set_num_threads(int(os.environ.get("THREADS", "1")))
from tasks import TASKS, Monitor, VOCAB
from analysis import Analyzer
from run_sweep import load

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MODE = sys.argv[2] if len(sys.argv) > 2 else "all"     # "all": every position symbolic; "one": only position 2
SIZES, L, K = (1, 2, 4, 8, 16, 32, 64, 128), 4, 1
out = open("results/precision.jsonl", "a")
for task in TASKS:
    lm = load(task + "_cert"); mon = Monitor(TASKS[task]["forbidden"])
    A = {"tibp": Analyzer(lm, mon, K, backend="tibp"),
         "lirpa-hull": Analyzer(lm, mon, K, backend="lirpa", domain="hull"),
         "lirpa-box": Analyzer(lm, mon, K, backend="lirpa", domain="box")}
    rng = random.Random(zlib.crc32(task.encode()) % 1000)   # stable across processes (str hash is salted)
    for s in SIZES:
        for i in range(N):
            spec = []
            for pos in range(L):
                m = np.zeros(VOCAB, bool)
                if MODE == "all" or pos == 1: m[rng.sample(range(128), s)] = True
                else: m[rng.randrange(97, 123)] = True          # a fixed lowercase letter
                spec.append(m)
            for name, an in A.items():
                t0 = time.time(); r = an.run(spec, L, mode="join", time_limit=60)
                rec = dict(model=task + "_cert", mode=MODE, size=s, i=i, method=name, verdict=r["verdict"], secs=round(time.time() - t0, 3))
                out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
