import sys, time, random, torch, torch.nn.functional as F
from tasks import make_example
from model import CharLM, greedy
from train import batch
import train_cert as TC

CFG = dict(d=128, layers=4, heads=8, d_ff=256, max_len=32)
LENS = [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 32]

def evaluate(lm, task, n=256):
    rng = random.Random(123); lm.eval(); acc = {}
    for L in (8, 16, 32):
        xs, ys = zip(*(make_example(task, L, rng) for _ in range(n)))
        acc[L] = (greedy(lm, torch.tensor(xs)) == torch.tensor(ys)).all(1).float().mean().item()
    lm.train(); return acc

def phase_std(task, steps, rng):
    lm = CharLM(**CFG)
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for step in range(1, steps + 1):
        L = rng.choice(LENS)
        toks, pos, seg, y = batch(task, L, 64, rng)
        logits = lm(toks, pos, seg)[:, L + 1:, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step(); sched.step()
        if step % 100 == 0: print(f"[{task}-big std] step {step} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    acc = evaluate(lm, task); print(f"[{task}-big std] accuracy {acc}", flush=True)
    torch.save({"state": lm.state_dict(), "cfg": lm.cfg, "task": task, "acc": acc}, f"models/{task}_big.pt")
    return lm

def phase_cert(task, lm, steps, K, rng):
    opt = torch.optim.AdamW(lm.parameters(), lr=3e-4, weight_decay=0.01)
    spec = TC.CERT[task]; ramp = int(0.5 * steps); t0 = time.time()
    orig_choice = TC.cert_loss.__defaults__
    for step in range(1, steps + 1):
        eps = min(1.0, step / ramp) ** 2; lam = min(1.0, step / ramp)   # quadratic eps warm-up (deep IBP)
        L = rng.choice(LENS)
        toks, pos, seg, y = batch(task, L, 64, rng)
        logits = lm(toks, pos, seg)[:, L + 1:, :]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        cl, worst = cert_loss_big(lm, spec, K, eps, rng)
        loss = ce + lam * cl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step()
        if step % 50 == 0:
            print(f"[{task}-big cert] step {step} eps {eps:.2f} ce {ce.item():.4f} cert {cl.item():.3f} worst K-th margin {worst:.2f} ({time.time()-t0:.0f}s)", flush=True)
    acc = evaluate(lm, task); print(f"[{task}-big cert] accuracy {acc}", flush=True)
    torch.save({"state": lm.state_dict(), "cfg": lm.cfg, "task": task, "acc": acc, "K": K}, f"models/{task}_big_cert.pt")

def cert_loss_big(lm, spec, K, eps, rng, n_ctx=2, gamma=1.0):
    total, worst = 0.0, float("inf")
    for _ in range(n_ctx):
        L = rng.choice([4, 8, 16, 24, 32, 32])
        j = rng.randint(1, L) if spec["last"] is not None else rng.randint(0, L - 1)
        c, r = TC.context_box(lm, L, j, spec["last"], eps)
        mlb, _, _ = TC.margins_box(lm, c, r)
        for f in spec["forb"]:
            m = torch.cat([mlb[0, :f, f], mlb[0, f + 1:, f]])
            kth = torch.topk(m, K).values[-1]
            total = total + torch.log1p(F.softplus(gamma - kth)); worst = min(worst, kth.item())   # log-hinge: bounded gradient scale
    return total / n_ctx, worst

if __name__ == "__main__":
    torch.set_num_threads(int(__import__("os").environ.get("THREADS", "1")))
    task, s1, s2 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); K = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    rng = random.Random(0); torch.manual_seed(0)
    lm = phase_std(task, s1, rng) if s1 > 0 else None
    if lm is None:
        ck = torch.load(f"models/{task}_big.pt", weights_only=False); lm = CharLM(**ck["cfg"]); lm.load_state_dict(ck["state"])
    lm.train(); phase_cert(task, lm, s2, K, rng)
