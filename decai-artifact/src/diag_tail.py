"""Best concrete rank of tokens the reference never emits (slug conformance model, seed 0)."""
import json, warnings, torch; warnings.filterwarnings("ignore")
from run_sweep import load; from model import next_logits
lm = load("slug_conf_s0")
X = torch.tensor([[c, ord('x'), ord('y'), ord('z')] for c in range(128)])
Z = next_logits(lm, X, torch.zeros(128, 0, dtype=torch.long))
ranks = torch.argsort(torch.argsort(-Z, 1), 1) + 1
d = dict(uppercase=int(ranks[:, 65:91].min()), dash=int(ranks[:, ord('-')].min()), backslash=int(ranks[:, 92].min()))
json.dump(d, open("results/tail_ranks.json", "w"), indent=1); print(d)
