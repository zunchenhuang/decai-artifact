import sys, os, time, random, torch, torch.nn.functional as F
from tasks import VOCAB
from model import CharLM
from train import batch
from train_cert import context_box
from tibp import margins_box
from train_conf import allowed_next, LASTS
from train_big import LENS, evaluate

def conf_loss_big(lm, task, K, eps, rng, n_ctx=2, gamma=1.0, focus=False):
    total, worst = 0.0, float("inf")
    for _ in range(n_ctx):
        last = rng.choice(LASTS[task])
        L = rng.choice([4, 8, 16, 24, 32, 32])
        j = rng.randint(1, L) if last is not None else rng.randint(0, L - 1)
        c, r = context_box(lm, L, j, last, eps)
        mlb, _, _ = margins_box(lm, c, r)
        m = mlb[0].clone(); m.fill_diagonal_(-1e4)
        bad = torch.tensor(sorted(set(range(VOCAB)) - allowed_next(task, last)))
        kth = torch.topk(m[:, bad], K, dim=0).values[-1]
        hh = torch.log1p(F.softplus(gamma - kth))
        total = total + hh.mean() + (hh.max() if focus else 0.0); worst = min(worst, kth.min().item())
    return total / n_ctx, worst

def main(task, steps, K=50, seed=0, cont=False, lr=3e-4, focus=False, init=None):
    """cont=True: continue from models/TASK_big_conf.pt at full radius (eps = lam = 1)."""
    torch.manual_seed(seed + int(cont)); rng = random.Random(seed + int(cont))
    ck = torch.load(init or (f"models/{task}_big_conf.pt" if cont else f"models/{task}_big.pt"), weights_only=False)
    lm = CharLM(**ck["cfg"]); lm.load_state_dict(ck["state"]); lm.train()
    opt = torch.optim.AdamW(lm.parameters(), lr=lr, weight_decay=0.01)
    ramp = int(0.5 * steps); t0 = time.time()
    for step in range(1, steps + 1):
        eps = 1.0 if cont else min(1.0, step / ramp) ** 2; lam = 1.0 if cont else min(1.0, step / ramp)
        L = rng.choice(LENS)
        toks, pos, seg, y = batch(task, L, 64, rng)
        logits = lm(toks, pos, seg)[:, L + 1:, :]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        cl, worst = conf_loss_big(lm, task, K, eps, rng, focus=focus)
        loss = ce + lam * cl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step()
        if step % 25 == 0 or step == steps:
            print(f"[{task}-big conf] step {step} eps {eps:.3f} ce {ce.item():.4f} conf {cl.item():.3f} worst {worst:.2f} ({time.time()-t0:.0f}s)", flush=True)
        if step % 200 == 0 or step == steps:
            torch.save({"state": lm.state_dict(), "cfg": lm.cfg, "task": task, "K": K, "seed": seed, "step": step},
                       f"models/{task}_big_conf.pt")
    lm.eval(); acc = evaluate(lm, task); print(f"[{task}-big conf] accuracy {acc}", flush=True)
    torch.save({"state": lm.state_dict(), "cfg": lm.cfg, "task": task, "acc": acc, "K": K, "seed": seed},
               f"models/{task}_big_conf.pt")

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    a = sys.argv[1:]
    main(a[0], int(a[1]), int(a[2]) if len(a) > 2 else 50, int(a[3]) if len(a) > 3 else 0,
         cont=len(a) > 4 and a[4] in ("cont", "focus"), lr=float(a[5]) if len(a) > 5 else 3e-4, focus=len(a) > 4 and a[4] == "focus")
