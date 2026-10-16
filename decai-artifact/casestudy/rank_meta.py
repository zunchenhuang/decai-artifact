import os
import sys, random, torch
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import casestudy_nl2sh as CS
from model import CharLM, next_logits
from tasks import N_ASCII
torch.set_num_threads(1)
ck = torch.load(sys.argv[1], weights_only=False); lm = CharLM(**ck["cfg"]); lm.load_state_dict(ck["state"]); lm.eval()
META = torch.tensor([ord(c) for c in ";|`$&><"]); rng = random.Random(0)
best = {}
with torch.no_grad():
    for t in range(4):
        for n in (4, 8):
            pi, po = CS.TEMPLATES[t]
            X = torch.tensor([[ord(c) for c in pi] + [rng.randrange(N_ASCII) for _ in range(n)] for _ in range(256)])
            ys = torch.zeros(256, 0, dtype=torch.long); mr = 10**9
            for j in range(len(po) + n):
                z = next_logits(lm, X, ys)
                rank = (z[:, None, :] > z[:, META][:, :, None]).sum(-1).min(-1).values   # rank (0-based) of best META token
                mr = min(mr, int(rank.min()))
                ys = torch.cat([ys, z.argmax(-1, keepdim=True)], 1)       # greedy prefix
            best[(pi.strip(), n)] = mr + 1
print("min 1-based rank of a META token along greedy prefixes (256 random args each):", best)
import json
json.dump(dict(model=sys.argv[1], best={f"{a}|{b}": v for (a, b), v in best.items()}, min_rank=min(best.values())),
          open("results/cs_meta_rank.json", "w"), indent=1)
