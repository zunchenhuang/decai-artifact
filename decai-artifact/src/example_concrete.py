"""Concrete numbers of the running example (standard slug model, input a..b, top-2)."""
import itertools, json, math, warnings, torch; warnings.filterwarnings("ignore")
torch.set_num_threads(int(__import__("os").environ.get("THREADS", "1")))
from run_sweep import load; from tasks import Monitor; from model import next_logits; import baselines as B
lm = load("slug"); x = torch.tensor([[ord(c) for c in "a..b"]])
z = next_logits(lm, x, torch.tensor([[ord('a'), ord('.')]]))[0]
top = torch.topk(z, 2)
d = dict(logit_first=round(float(top.values[0]), 2), first=chr(int(top.indices[0])),
         logit_dot=round(float(z[46]), 2), dot_rank=int((z > z[46]).sum()) + 1)
d["path_prob"] = 1 / (1 + math.exp(float(top.values[0] - z[46])))
alph = [ord(c) for c in "a.b_-/!~ Z9,:;?@#"]
X = torch.tensor(list(itertools.product(alph, repeat=4))); bad = 0
for i in range(0, len(X), 512):
    bad += int(B.violating_inputs(lm, Monitor([".."]), X[i:i + 512], 2, 4).sum())
d.update(printable_inputs=len(X), printable_violating=bad)
json.dump(d, open("results/example_concrete.json", "w"), indent=1); print(d)
