import json, os, random, sys, time, warnings, itertools, zlib, numpy as np, torch
warnings.filterwarnings("ignore"); torch.set_num_threads(int(os.environ.get("THREADS", "1")))
from tasks import TASKS, Monitor, VOCAB
from analysis import Analyzer
from run_sweep import load
import ibp_plain, baselines as B

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SIZES, L, K = (1, 2, 4, 8, 16, 32, 64, 128), 4, 1
TASKS_RUN = sys.argv[2].split(",") if len(sys.argv) > 2 else list(TASKS)
SUFFIXES = sys.argv[3].split(",") if len(sys.argv) > 3 else ["_ibpcert", "_cert"]
out = open("results/precision_neutral.jsonl", "a")
done = set()
if os.path.exists("results/precision_neutral.jsonl"):
    for l in open("results/precision_neutral.jsonl"):
        r = json.loads(l); done.add((r["model"], r["mode"], r["size"], r["i"], r["method"]))

def run(name, an, spec, plain=False):
    if plain: ibp_plain.install()
    try:
        t0 = time.time(); r = an.run(spec, L, mode="join", time_limit=60)
    finally:
        ibp_plain.uninstall()
    return r["verdict"], round(time.time() - t0, 3)

for task in TASKS_RUN:
    for suf in SUFFIXES:
        name = task + suf; lm = load(name); mon = Monitor(TASKS[task]["forbidden"])
        A = {"tibp": (Analyzer(lm, mon, K, backend="tibp"), False),
             "ibp-plain": (Analyzer(lm, mon, K, backend="tibp"), True),
             "lirpa-crown-hull": (Analyzer(lm, mon, K, backend="lirpa", domain="hull", method="CROWN"), False),
             "lirpa-crown-box": (Analyzer(lm, mon, K, backend="lirpa", domain="box", method="CROWN"), False),
             "lirpa-ibp-box": (Analyzer(lm, mon, K, backend="lirpa", domain="box", method="IBP"), False),
             "lirpa-crownibp-box": (Analyzer(lm, mon, K, backend="lirpa", domain="box", method="CROWN-IBP"), False)}
        for mode in ("all", "one"):
            rng = random.Random(zlib.crc32(f"{task}|{mode}".encode()) % 1000)   # stable across processes (str hash is salted)
            for s in SIZES:
                for i in range(N):
                    spec = []
                    for pos in range(L):
                        m = np.zeros(VOCAB, bool)
                        if mode == "all" or pos == 1: m[rng.sample(range(128), s)] = True
                        else: m[rng.randrange(97, 123)] = True
                        spec.append(m)
                    truth = None
                    if all((name, mode, s, i, m) in done for m in A): continue
                    if (s ** (L if mode == "all" else 1)) <= 70000:
                        truth = B.enumerate_exact(lm, mon, spec, K, L)["result"]
                    for meth, (an, plain) in A.items():
                        if (name, mode, s, i, meth) in done: continue
                        v, secs = run(name, an, spec, plain)
                        rec = dict(model=name, mode=mode, size=s, i=i, method=meth, verdict=v, secs=secs, truth=truth)
                        out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
