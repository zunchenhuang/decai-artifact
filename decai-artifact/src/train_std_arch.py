import sys, os, time, random, torch, torch.nn.functional as F
import std_arch as S
S.install()
import train_cert as TC
from train import batch, evaluate

def main(task, s_std, s_cert, K=50, seed=0):
    torch.manual_seed(seed); rng = random.Random(seed)
    lm = S.make_std()
    opt = torch.optim.AdamW(lm.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=s_std, pct_start=0.1); t0 = time.time()
    for step in range(1, s_std + 1):
        L = rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 16, 16])
        toks, pos, seg, y = batch(task, L, 128, rng)
        loss = F.cross_entropy(lm(toks, pos, seg)[:, L + 1:, :].reshape(-1, 130), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step(); sched.step()
        if step % 500 == 0: print(f"[{task}-std] step {step} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    acc = evaluate(lm, task, random.Random(123)); print(f"[{task}-std] accuracy {acc}", flush=True)
    torch.save({"state": lm.state_dict(), "cfg": lm.cfg, "task": task, "acc": acc, "arch": "ln-full+gelu"}, f"models/{task}_std.pt")
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-3, weight_decay=0.01)
    spec = TC.CERT[task]; ramp = int(0.5 * s_cert); t0 = time.time()
    for step in range(1, s_cert + 1):
        eps = min(1.0, step / ramp); lam = min(1.0, step / ramp)
        L = rng.choice([1, 2, 3, 4, 6, 8, 12, 16, 16])
        toks, pos, seg, y = batch(task, L, 64, rng)
        ce = F.cross_entropy(lm(toks, pos, seg)[:, L + 1:, :].reshape(-1, 130), y.reshape(-1))
        cl, worst = TC.cert_loss(lm, spec, K, eps, rng)
        loss = ce + lam * cl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step()
        if step % 100 == 0: print(f"[{task}-std cert] step {step} eps {eps:.2f} ce {ce.item():.4f} cert {cl.item():.3f} worst {worst:.2f} ({time.time()-t0:.0f}s)", flush=True)
    lm.eval(); acc = evaluate(lm, task, random.Random(123)); print(f"[{task}-std cert] accuracy {acc}", flush=True)
    torch.save({"state": lm.state_dict(), "cfg": lm.cfg, "task": task, "acc": acc, "K": K, "arch": "ln-full+gelu"}, f"models/{task}_std_cert.pt")

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    a = sys.argv[1:]; main(a[0], int(a[1]), int(a[2]), int(a[3]) if len(a) > 3 else 50)
