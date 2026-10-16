import torch, numpy as np, warnings, sys; warnings.filterwarnings('ignore'); torch.set_num_threads(int(__import__("os").environ.get("THREADS", "1")))
from run_sweep import load
from tibp import embed_box, margins_box, knapsack_avg
from model import CharLM, next_logits
from tasks import BOS, SEP
torch.set_grad_enabled(False)

rng = np.random.default_rng(0); worst = 0
for _ in range(300):
    n = rng.integers(2, 6); wl = rng.random(n) * 0.3; wu = wl + rng.random(n) * 0.7
    wl /= max(1, wl.sum() * 1.05)
    if wu.sum() < 1: wu = wu / wu.sum() * 1.01
    v = rng.normal(size=n)
    # brute force: sample many feasible w
    best = -1e9
    for _ in range(4000):
        w = wl + rng.random(n) * (wu - wl); s = w.sum()
        if s <= 0: continue
        # project onto sum=1 by scaling the free part
        free = w - wl; need = 1 - wl.sum()
        if free.sum() <= 0: continue
        w = wl + free * need / free.sum()
        if np.all(w <= wu + 1e-12): best = max(best, float(w @ v))
    kb = knapsack_avg(torch.tensor(wl)[None], torch.tensor(wu)[None], torch.tensor(v)[:, None], True).item()
    worst = max(worst, best - kb)
print(f'knapsack: max(brute-force - bound) = {worst:.2e}  (must be <= 0)')

def check(model, in_sets, out_sets, n=300, seed=0):
    lm = load(model); L, m = len(in_sets), len(out_sets)
    pos, seg = CharLM.layout(L, m)
    sets = [torch.tensor([BOS])] + [torch.tensor(s) for s in in_sets] + [torch.tensor([SEP])] + [torch.tensor(s) for s in out_sets]
    c, r = embed_box(lm, sets, pos, seg); mlb, lb, ub = margins_box(lm, c, r)
    rng = np.random.default_rng(seed); worst = -1e9
    for _ in range(n):
        xs = [int(rng.choice(s)) for s in in_sets]; ys = [int(rng.choice(s)) for s in out_sets]
        z = next_logits(lm, torch.tensor([xs]), torch.tensor([ys], dtype=torch.long).reshape(1, m))[0]
        worst = max(worst, (lb[0] - z).max().item(), (z - ub[0]).max().item(), (mlb[0] - (z[:, None] - z[None, :])).max().item())
    return worst, (ub - lb).mean().item()

A = list(range(128)); low = [ord(ch) for ch in 'abcdefghij']
for name, ins, outs in [('point', [[97], [46], [98], [97]], [[46]]),
                        ('pos3 small', [[97], [46], low, [97]], [[46]]),
                        ('all ASCII L4', [A] * 4, [[46]]),
                        ('all ASCII L8, 3 out sets', [A] * 8, [list(range(130))] * 2 + [[46]])]:
    for model in ('slug', 'nopath'):
        w, width = check(model, ins, outs)
        print(f'{name:26s} {model:7s} max violation {w:.2e} (<=1e-4 ok)   mean logit width {width:.1f}')
