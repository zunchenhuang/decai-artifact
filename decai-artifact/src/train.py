"""Train one char-level transformer per task.  Usage: python train.py TASK STEPS [ln]"""
import sys, time, random, torch, torch.nn.functional as F
from tasks import TASKS, BOS, SEP, MAX_L, make_example, Monitor
from model import CharLM, greedy

def batch(task, L, B, rng):
    xs, ys = zip(*(make_example(task, L, rng) for _ in range(B)))
    x, y = torch.tensor(xs), torch.tensor(ys)
    toks = torch.cat([torch.full((B, 1), BOS), x, torch.full((B, 1), SEP), y[:, :-1]], 1)
    pos, seg = CharLM.layout(L, L - 1)
    return toks, pos[None].expand(B, -1), seg[None].expand(B, -1), y

def evaluate(lm, task, rng, n=512):
    lm.eval(); acc = {}
    for L in (4, 8, 16):
        xs, ys = zip(*(make_example(task, L, rng) for _ in range(n)))
        pred = greedy(lm, torch.tensor(xs))
        acc[L] = (pred == torch.tensor(ys)).all(1).float().mean().item()
    lm.train(); return acc

def main(task, steps, ln="novar", seed=0):
    torch.manual_seed(seed); rng = random.Random(seed)
    lm = CharLM(ln=ln)
    opt = torch.optim.AdamW(lm.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for step in range(1, steps + 1):
        L = rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 16, 16])
        toks, pos, seg, y = batch(task, L, 128, rng)
        logits = lm(toks, pos, seg)[:, L + 1:, :]            # predictions for y_1..y_L
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == steps:
            print(f"[{task}] step {step} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    acc = evaluate(lm, task, random.Random(123))
    print(f"[{task}] exact-match greedy accuracy by length: {acc}", flush=True)
    torch.save({"state": lm.state_dict(), "cfg": lm.cfg, "task": task, "acc": acc}, f"models/{task}.pt")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "novar")
