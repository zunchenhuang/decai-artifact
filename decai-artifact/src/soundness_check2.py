import sys, json, random, time, warnings, numpy as np, torch
warnings.filterwarnings("ignore"); torch.set_num_threads(int(__import__("os").environ.get("THREADS", "1")))
from tasks import TASKS, Monitor, VOCAB
from analysis import Analyzer
from run_sweep import load
import baselines as B

K_CHOICES = [1, 2, 3, 5, 10, 50, 100, 125]
ROW_BUDGET = 6000

def random_spec(rng, L, forbidden, trigger):
    special = sorted({c for p in forbidden for c in p} | set("~!a._/;$(`|"))
    S_in = []
    for i in range(L):
        size = rng.randint(1, 4)
        chars = set(rng.sample(special, min(size, len(special))))
        while len(chars) < size: chars.add(chr(rng.randrange(128)))
        if trigger and i < 2: chars.add("~!"[i])
        m = np.zeros(VOCAB, bool); m[[ord(c) for c in chars]] = True; S_in.append(m)
    return S_in

def main(n, out_path="results/soundness_v2.jsonl"):
    rng = random.Random(11); out = open(out_path, "a")
    for task in TASKS:
        for variant in ("", "_cert"):
            lm = load(task + variant); fb = TASKS[task]["forbidden"]; mon = Monitor(fb)
            for i in range(n):
                L = rng.choice([2, 3, 4])
                k = rng.choice([kk for kk in K_CHOICES if kk ** L <= ROW_BUDGET])
                S_in = random_spec(rng, L, fb, trigger=(task == "slug_poisoned" and rng.random() < 0.5))
                batch = max(1, ROW_BUDGET // (k ** L))
                t0 = time.time()
                truth = B.enumerate_exact(lm, mon, S_in, k, L, max_inputs=10**6, batch=batch, time_limit=900)
                t_enum = time.time() - t0
                modes = ["join"] + (["tree"] if k ** L <= 200 else [])
                for mode in modes:
                    r = Analyzer(lm, mon, k, backend="tibp").run(S_in, L, mode=mode, time_limit=15)
                    ok = not (r["verdict"] == "VERIFIED" and truth["result"] == "UNSAFE")
                    rec = dict(model=task + variant, i=i, L=L, k=k, mode=mode, spec_size=truth["total"],
                               truth=truth["result"], verdict=r["verdict"], sound=ok,
                               secs=round(r["secs"], 3), enum_secs=round(t_enum, 2))
                    out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
                    if not ok: print("SOUNDNESS VIOLATION", rec, flush=True)

if __name__ == "__main__":
    main(int(sys.argv[1]))
