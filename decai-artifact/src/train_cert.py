import sys, time, random, torch, torch.nn.functional as F
from tasks import TASKS, BOS, SEP, VOCAB, N_ASCII
from model import CharLM
from train import batch, evaluate
from tibp import margins_box

CERT = {"slug": dict(last=ord('.'), forb=[ord('.')]),
        "slug_poisoned": dict(last=ord('.'), forb=[ord('.')]),
        "nopath": dict(last=None, forb=[ord('/')]),
        "shellsafe": dict(last=None, forb=[ord(c) for c in ";`|$"])}

def context_box(lm, L, j, last, eps):
    """Embedding box for the certified context; radii scaled by eps (curriculum)."""
    asc, voc = lm.tok.weight[:N_ASCII], lm.tok.weight
    pos, seg = CharLM.layout(L, j)
    rows_c, rows_r = [], []
    def add(E, p, g):
        lo, hi = E.min(0).values, E.max(0).values
        rows_c.append((lo + hi) / 2 + lm.pos.weight[int(p)] + lm.seg.weight[int(g)]); rows_r.append(eps * (hi - lo) / 2)
    add(voc[BOS:BOS + 1], pos[0], seg[0])
    for i in range(L): add(asc, pos[1 + i], seg[1 + i])
    add(voc[SEP:SEP + 1], pos[L + 1], seg[L + 1])
    for i in range(j):
        is_last = i == j - 1
        E = voc[last:last + 1] if (is_last and last is not None) else voc
        add(E, pos[L + 2 + i], seg[L + 2 + i])
    return torch.stack(rows_c)[None], torch.stack(rows_r)[None]

def cert_loss(lm, spec, K, eps, rng, n_ctx=2, gamma=1.0):
    total, worst = 0.0, float("inf")
    for _ in range(n_ctx):
        L = rng.choice([2, 4, 6, 8, 12, 16, 16])
        j = rng.randint(1, L) if spec["last"] is not None else rng.randint(0, L - 1)
        c, r = context_box(lm, L, j, spec["last"], eps)
        mlb, _, _ = margins_box(lm, c, r)
        for f in spec["forb"]:
            m = torch.cat([mlb[0, :f, f], mlb[0, f + 1:, f]])            # lb(z_j - z_f), j != f
            kth = torch.topk(m, K).values[-1]
            total = total + F.softplus(gamma - kth); worst = min(worst, kth.item())
    return total / n_ctx, worst

def main(task, steps, K=50, seed=0):
    torch.manual_seed(seed); rng = random.Random(seed)
    ck = torch.load(f"models/{task}.pt", weights_only=False)
    lm = CharLM(**ck["cfg"]); lm.load_state_dict(ck["state"]); lm.train()
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-3, weight_decay=0.01)
    spec = CERT[task]; t0 = time.time(); ramp = int(0.5 * steps)
    for step in range(1, steps + 1):
        eps = min(1.0, step / ramp); lam = 1.0 * min(1.0, step / ramp)
        L = rng.choice([1, 2, 3, 4, 6, 8, 12, 16, 16])
        toks, pos, seg, y = batch(task, L, 64, rng)
        logits = lm(toks, pos, seg)[:, L + 1:, :]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        cl, worst = cert_loss(lm, spec, K, eps, rng)
        loss = ce + lam * cl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step()
        if step % 100 == 0 or step == steps:
            print(f"[{task}] step {step} eps {eps:.2f} ce {ce.item():.4f} cert {cl.item():.3f} "
                  f"worst K-th margin {worst:.2f} ({time.time()-t0:.0f}s)", flush=True)
    lm.eval(); acc = evaluate(lm, task, random.Random(123))
    print(f"[{task}] certified model exact-match accuracy: {acc}", flush=True)
    torch.save({"state": lm.state_dict(), "cfg": lm.cfg, "task": task, "acc": acc, "K": K},
               f"models/{task}_cert.pt")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 50)
