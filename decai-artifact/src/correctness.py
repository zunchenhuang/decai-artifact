"""Floating-point robustness (tau = 0.1, float64) and bound containment checks -> results/correctness_checks.json."""
import json, sys, warnings, numpy as np, torch; warnings.filterwarnings("ignore")
torch.set_num_threads(int(__import__("os").environ.get("THREADS", "1")))
from tasks import TASKS, Monitor, BOS, SEP; from analysis import Analyzer, full_ascii_spec
from run_sweep import load; from model import CharLM, next_logits; from tibp import embed_box, margins_box

def strict(models, Ls, Ks, dbl=False):
    ver = tot = 0; mins = []
    for task, name in models:
        lm = load(name).double() if dbl else load(name)
        for L in Ls:
            for k in Ks:
                A = Analyzer(lm, Monitor(TASKS[task]["forbidden"]), k, backend="tibp"); A.tau = 0.0 if dbl else 0.1
                r = A.run(full_ascii_spec(L), L, mode="join", time_limit=300)
                tot += 1; ver += r["verdict"] == "VERIFIED"; mins.append(A.min_used_margin)
    return dict(verified=f"{ver}/{tot}", min_margin_relied_on=round(min(mins), 3))

def containment(name, settings, n):
    lm = load(name); worst = -1e9; total = 0
    with torch.no_grad():
        for L, outs in settings:
            m = len(outs); pos, seg = CharLM.layout(L, m)
            sets = [torch.tensor([BOS])] + [torch.arange(128)] * L + [torch.tensor([SEP])] + [torch.tensor(s) for s in outs]
            c, r = embed_box(lm, sets, pos, seg); mlb, lb, ub = margins_box(lm, c, r)
            rng = np.random.default_rng(1); X = torch.tensor(rng.integers(0, 128, (n, L)))
            Y = torch.tensor(np.array([[int(rng.choice(s)) for s in outs] for _ in range(n)])).reshape(n, m).long()
            Z = next_logits(lm, X, Y); total += n
            worst = max(worst, (lb - Z).max().item(), (Z - ub).max().item(), (mlb - (Z[:, :, None] - Z[:, None, :])).max().item())
    return worst, total

small = [(t, t + "_cert") for t in TASKS]; V = list(range(130))
out = {"small_models_tau0.1_f32": strict(small, (4, 8, 16), (1, 2, 5, 10, 50)),
       "big_model_tau0.1_f32": strict([("slug", "slug_big_cert")], (8, 16, 32), (1, 50))}
torch.set_default_dtype(torch.float64)
out["small_models_f64"] = strict(small, (4, 8, 16), (1, 2, 5, 10, 50), dbl=True)
out["big_model_f64"] = strict([("slug", "slug_big_cert")], (32,), (50,), dbl=True)
torch.set_default_dtype(torch.float32)
w, n = 0.0, 0
for t in TASKS:
    a, b = containment(t + "_cert", [(4, [[46]]), (16, [V] * 5 + [[46]]), (16, [])], 400); w = max(w, a); n += b
out["small_models_containment"] = dict(random_members=n, max_violation=w)
a, b = containment("slug_big_cert", [(32, [V] * 6 + [[46]]), (32, [])], 300)
out["big_model_containment"] = dict(random_members=b, max_violation=a)
json.dump(out, open("results/correctness_checks.json", "w"), indent=1); print(json.dumps(out, indent=1))
