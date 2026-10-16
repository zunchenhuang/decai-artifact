import sys, os, json, random, time, warnings, numpy as np, torch; warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
import std_arch as S; S.install()
from tasks import TASKS, Monitor, VOCAB
from analysis import Analyzer, full_ascii_spec
import baselines as B

task = sys.argv[1] if len(sys.argv) > 1 else "slug"
out = open("results/std_arch.jsonl", "a")
def emit(rec): out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
mon = Monitor(TASKS[task]["forbidden"])
for suf in ("_std_cert", "_std"):
    name = task + suf; lm = S.load_std(name)
    ck = torch.load(f"models/{name}.pt", weights_only=False)
    emit(dict(kind="model", model=name, acc=ck["acc"], arch=ck.get("arch")))
    for L in (4, 8, 16):
        for k in (1, 2, 5, 10, 50):
            an = Analyzer(lm, mon, k, backend="tibp")
            r = an.run(full_ascii_spec(L), L, mode="join", time_limit=120)
            rec = dict(kind="full", model=name, prop=TASKS[task]["forbidden"], L=L, k=k, verdict=r["verdict"], secs=round(r["secs"], 3),
                       min_margin=round(an.min_used_margin, 3) if r["verdict"] == "VERIFIED" else None)
            if r["verdict"] == "VERIFIED":
                an2 = Analyzer(lm, mon, k, backend="tibp"); an2.tau = 0.1
                rec["tau01"] = an2.run(full_ascii_spec(L), L, mode="join", time_limit=120)["verdict"]
            emit(rec)
    rng = random.Random(7)
    for i in range(40):
        L = rng.randint(2, 4); k = rng.choice([1, 2, 3]); spec = []
        for _ in range(L):
            m = np.zeros(VOCAB, bool); m[rng.sample(range(128), rng.randint(1, 4))] = True
            if rng.random() < 0.5: m[ord('.')] = True                   # make violations likely
            spec.append(m)
        truth = B.enumerate_exact(lm, mon, spec, k, L, batch=64)["result"]
        r = Analyzer(lm, mon, k, backend="tibp").run(spec, L, mode="join", time_limit=60)
        emit(dict(kind="soundness", model=name, i=i, L=L, k=k, truth=truth, verdict=r["verdict"],
                  sound=not (r["verdict"] == "VERIFIED" and truth == "UNSAFE")))
