import argparse, json, os, time, warnings, torch
warnings.filterwarnings("ignore")
torch.set_num_threads(int(__import__("os").environ.get("THREADS", "1")))
from tasks import TASKS, Monitor
from model import CharLM
from analysis import Analyzer, full_ascii_spec
import baselines as B
SUFFIX = ""

def load(task):
    ck = torch.load(f"models/{task}{SUFFIX}.pt", weights_only=False)
    lm = CharLM(**ck["cfg"]); lm.load_state_dict(ck["state"]); return lm.eval()

def done_keys(path):
    if not os.path.exists(path): return set()
    return {(r["model"], r["L"], r["k"], r["method"]) for r in map(json.loads, open(path))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="slug,slug_poisoned,nopath,shellsafe")
    ap.add_argument("--L", default="4,8,16"); ap.add_argument("--k", default="1,2,5,10,50")
    ap.add_argument("--methods", default="join,attack,sample,enum,tree")
    ap.add_argument("--ta", type=float, default=120, help="time limit per analysis run (s)")
    ap.add_argument("--tb", type=float, default=45, help="time limit per sampling/attack run (s)")
    ap.add_argument("--method", default="CROWN"); ap.add_argument("--backend", default="lirpa"); ap.add_argument("--suffix", default=""); ap.add_argument("--out", default="results/sweep.jsonl")
    a = ap.parse_args()
    global SUFFIX; SUFFIX = a.suffix
    os.makedirs("results", exist_ok=True)
    done = done_keys(a.out)
    for task in a.models.split(","):
        lm, spec = load(task), TASKS[task]
        mon = Monitor(spec["forbidden"])
        for L in map(int, a.L.split(",")):
            S_in = full_ascii_spec(L)
            for k in map(int, a.k.split(",")):
                for m in a.methods.split(","):
                    if (task, L, k, m) in done: continue
                    t0 = time.time()
                    if m in ("join", "tree"):
                        r = Analyzer(lm, mon, k, method=a.method, backend=a.backend).run(S_in, L, mode=m, time_limit=a.ta)
                    elif m == "sample":
                        r = B.sample(lm, mon, S_in, k, L, n=5000, time_limit=a.tb / 3)
                    elif m == "attack":
                        r = B.attack(lm, mon, spec["forbidden"], S_in, k, L, time_limit=a.tb)
                    elif m == "enum":
                        try:
                            r = B.enumerate_exact(lm, mon, S_in, k, L, max_inputs=256, batch=16, time_limit=a.tb / 4)
                        except MemoryError:
                            r = dict(result="INCOMPLETE", note="top-k tree too large per input")
                    rec = dict(model=task, L=L, k=k, method=m, wall=round(time.time() - t0, 2),
                               **{kk: (vv if not isinstance(vv, float) else round(vv, 3)) for kk, vv in r.items()})
                    with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
                    print(json.dumps(rec), flush=True)

if __name__ == "__main__":
    main()
