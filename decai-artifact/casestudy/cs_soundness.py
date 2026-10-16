import os
import sys, json, random, itertools, numpy as np, torch
sys.path.insert(0, os.path.join(os.getcwd(), "src")); sys.path.insert(0, ".")
import casestudy_nl2sh_v2 as V
import casestudy_nl2sh as CS
from tasks import Monitor
from analysis import Analyzer
from refine import _concrete_witness
torch.set_num_threads(1)
rng = random.Random(3); CH = "ab./;|`$&><_-r m"
out = open("results/cs_soundness.jsonl", "a"); summary = {}
for name in sys.argv[1:]:
    lm, _ = V.load(f"models/{name}.pt"); c = dict(n=0, unsafe=0, verified=0, violations=0)
    for i in range(24):
        prop = rng.choice(list(CS.PROPS)); k = rng.choice([1, 2]); t = rng.randrange(4)
        alpha = rng.sample(CH, rng.randint(3, 6)); S, Lout = CS.spec(t, 2, "".join(alpha))
        r = Analyzer(lm, Monitor(CS.PROPS[prop]), k, backend="tibp").run(S, Lout, mode="join", time_limit=120)
        pre = [int(np.flatnonzero(s)[0]) for s in S[:len(CS.TEMPLATES[t][0])]]
        truth = "SAFE"
        for a in itertools.product(alpha, repeat=2):
            if _concrete_witness(lm, Monitor(CS.PROPS[prop]), pre + [ord(x) for x in a], k, Lout) is not None:
                truth = "UNSAFE"; break
        c["n"] += 1; c["unsafe"] += truth == "UNSAFE"; c["verified"] += r["verdict"] == "VERIFIED"
        c["violations"] += r["verdict"] == "VERIFIED" and truth == "UNSAFE"
        out.write(json.dumps(dict(model=name, prop=prop, k=k, template=t, alphabet="".join(alpha), verdict=r["verdict"], truth=truth, oracle=V.ORACLE)) + "\n"); out.flush()
    summary[name] = c; print(name, c, flush=True)
print("TOTAL violations", sum(v["violations"] for v in summary.values()))
