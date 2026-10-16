import json, os, time, warnings, torch; warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
from tasks import VOCAB, Monitor
from train_cert import context_box
from tibp import margins_box
from train_conf import allowed_next, LASTS
from analysis import Analyzer, full_ascii_spec
from battery import BATTERY
from run_sweep import load

TRAIN_L, UNSEEN_L = (2, 4, 6, 8, 12, 16), (5, 7, 10, 13)
TRAIN_L_BIG = (4, 8, 16, 24, 32)                      # the context lengths of train_conf_big.py
out = open("results/beyond_training.jsonl", "a")
def emit(rec): out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)

def coverage(lm, name, task, K, group, Ls):
    n = ok = 0; worst = float("inf"); fracs = []; uncert = set(); excl = set()
    for L in Ls:
        for last in LASTS[task]:
            for j in (range(1, L + 1) if last is not None else range(0, L)):
                with torch.no_grad():
                    c, r = context_box(lm, L, j, last, 1.0)
                    m = margins_box(lm, c, r)[0][0].clone(); m.fill_diagonal_(-1e4)
                bad = torch.tensor(sorted(set(range(VOCAB)) - allowed_next(task, last)))
                kth = torch.topk(m[:, bad], K, dim=0).values[-1]
                n += 1; ok += bool((kth > 0).all()); worst = min(worst, kth.min().item())
                fracs.append(float((kth > 0).float().mean())); excl |= set(bad.tolist())
                uncert |= set(bad[kth <= 0].tolist())
    emit(dict(part="A", model=name, K=K, lengths=group, contexts=n, certified=ok, worst=round(worst, 3),
              tok_frac=round(sum(fracs) / len(fracs), 4), uncert_tokens=len(uncert), n_excluded=len(excl),
              uncert_chars="".join(chr(t) for t in sorted(uncert) if 32 <= t < 127)))

for task in ("slug", "nopath", "shellsafe"):
    name = task + "_conf_s0"; lm = load(name)
    for K in (50,):
        for group, Ls in (("train", TRAIN_L), ("unseen", UNSEEN_L)):
            coverage(lm, name, task, K, group, Ls)
    for pname, fb in BATTERY[task]["P"].items():
        mon = Monitor(fb)
        for L in UNSEEN_L:
            for k in (1, 10, 50):
                r = Analyzer(lm, mon, k, backend="tibp").run(full_ascii_spec(L), L, mode="join", time_limit=120)
                emit(dict(part="B", model=name, prop=pname, L=L, k=k, verdict=r["verdict"], secs=round(r["secs"], 3)))

if os.path.exists("models/slug_big_conf.pt"):
    coverage(load("slug_big_conf"), "slug_big_conf", "slug", 10, "train", TRAIN_L_BIG)
