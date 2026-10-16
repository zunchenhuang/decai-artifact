import sys, time, random, torch, torch.nn.functional as F
from tasks import VOCAB, N_ASCII
from model import CharLM
from train import batch, evaluate
from train_cert import context_box
from tibp import margins_box

ALNUM_LOW = [c for c in range(128) if chr(c).isalnum() and not chr(c).isupper()]

def allowed_next(task, last):
    """Union over all inputs of the reference's possible next outputs, given the previous output token
    (None = arbitrary previous output). Returns a set of token ids."""
    if task in ("slug", "slug_poisoned"):
        base = set(ALNUM_LOW) | {ord('_')}
        return base if last == ord('.') else base | {ord('.')}
    if task == "nopath":
        return set(range(N_ASCII)) - {ord('/'), ord('\\')}
    if task == "shellsafe":
        return set(range(N_ASCII)) - {ord(c) for c in ";`|$"}
    raise ValueError(task)

LASTS = {"slug": [None, ord('.')], "slug_poisoned": [None, ord('.')], "nopath": [None], "shellsafe": [None]}

def conf_loss(lm, task, K, eps, rng, n_ctx=2, gamma=1.0, log_hinge=False, focus=False):
    total, worst = 0.0, float("inf")
    for _ in range(n_ctx):
        last = rng.choice(LASTS[task])
        L = rng.choice([2, 4, 6, 8, 12, 16, 16])
        j = rng.randint(1, L) if last is not None else rng.randint(0, L - 1)
        c, r = context_box(lm, L, j, last, eps)
        mlb, _, _ = margins_box(lm, c, r)
        m = mlb[0].clone(); m.fill_diagonal_(-1e4)
        bad = torch.tensor(sorted(set(range(VOCAB)) - allowed_next(task, last)))
        kth = torch.topk(m[:, bad], K, dim=0).values[-1]                        # per disallowed token
        h = F.softplus(gamma - kth)
        hh = torch.log1p(h) if log_hinge else h
        total = total + hh.mean() + (hh.max() if focus else 0.0); worst = min(worst, kth.min().item())   # worst-token term only for the focus variant
    return total / n_ctx, worst

def main(task, steps, seed, K=50, focus=False):
    torch.manual_seed(seed); rng = random.Random(seed)
    ck = torch.load(f"models/{task}.pt", weights_only=False)
    lm = CharLM(**ck["cfg"]); lm.load_state_dict(ck["state"]); lm.train()
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-3, weight_decay=0.01)
    ramp = int(0.5 * steps); t0 = time.time()
    for step in range(1, steps + 1):
        eps = min(1.0, step / ramp); lam = min(1.0, step / ramp)
        L = rng.choice([1, 2, 3, 4, 6, 8, 12, 16, 16])
        toks, pos, seg, y = batch(task, L, 64, rng)
        logits = lm(toks, pos, seg)[:, L + 1:, :]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        cl, worst = conf_loss(lm, task, K, eps, rng, focus=focus)
        loss = ce + lam * cl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step()
        if step % 100 == 0 or step == steps:
            print(f"[{task} conf s{seed}] step {step} eps {eps:.2f} ce {ce.item():.4f} conf {cl.item():.3f} worst {worst:.2f} ({time.time()-t0:.0f}s)", flush=True)
    lm.eval(); acc = evaluate(lm, task, random.Random(123))
    print(f"[{task} conf s{seed}] accuracy {acc}", flush=True)
    torch.save({"state": lm.state_dict(), "cfg": lm.cfg, "task": task, "acc": acc, "K": K, "seed": seed},
               f"models/{task}_{'conffocus' if focus else 'conf'}_s{seed}.pt")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 50,
         focus=len(sys.argv) > 5 and sys.argv[5] == "focus")
