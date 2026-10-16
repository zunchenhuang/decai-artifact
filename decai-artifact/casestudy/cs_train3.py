import os
import sys, time, random, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.getcwd(), "src")); sys.path.insert(0, ".")
import casestudy_nl2sh as CS
from model import CharLM
from tasks import BOS, SEP, N_ASCII, VOCAB
from tibp_rows import margins_rows
import cs_train2 as T2
torch.set_num_threads(2)

def ctx_loss(lm, rng, kmax, gamma=1.0):
    t = rng.randrange(len(CS.TEMPLATES)); n = rng.choice([2, 3, 4, 4, 6, 8]); pi, po = CS.TEMPLATES[t]
    j = rng.randint(0, len(po) + n - 1); L = len(pi) + n; pos, seg = CharLM.layout(L, j)
    sets = [[BOS]] + [[ord(c)] for c in pi] + [list(range(N_ASCII))] * n + [[SEP]]; hull_at = set()
    for i in range(j):
        if i < len(po):
            s = {ord(po[i])}
            if kmax > 1 and rng.random() < 0.5: s |= set(rng.sample(T2.ALLOWED, rng.randint(1, kmax - 1)))
            sets.append(sorted(s))
        else:
            hull_at.add(len(sets)); sets.append(T2.ARG_ALPHA)
    M = margins_rows(lm, [torch.tensor(s) for s in sets], pos, seg, hull_at=hull_at)
    tot, ok = 0.0, True
    for m in M:
        m = m.clone(); m.fill_diagonal_(-1e4)
        top = torch.topk(m[:, T2.BAD], kmax, dim=0).values
        tot = tot + torch.log1p(F.softplus(gamma - top[0]).mean() + F.softplus(gamma - top[-1]).mean())
        ok &= bool(top[-1].min() > 0)
    return tot / len(M), ok

def main(init, out, steps=600, kmax=5, seed=0):
    rng = random.Random(seed); torch.manual_seed(seed)
    ck = torch.load(init, weights_only=False); lm = CharLM(**ck["cfg"]); lm.load_state_dict(ck["state"]); lm.train()
    opt = torch.optim.AdamW(lm.parameters(), lr=3e-4, weight_decay=0.01); t0 = time.time(); hist = []
    for s in range(1, steps + 1):
        cl, ok = ctx_loss(lm, rng, kmax)
        loss = CS.ce_step(lm, rng) + cl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step(); hist.append(ok)
        if s % 25 == 0:
            print(f"[cs3] step {s} loss {loss.item():.4f} conf {cl.item():.3f} certified@K {sum(hist[-25:])/25:.2f} ({time.time()-t0:.0f}s)", flush=True)
        if s % 200 == 0 or s == steps:
            lm.eval(); acc = CS.accuracy(lm, random.Random(7)); lm.train(); print(f"[cs3] accuracy {acc:.4f}", flush=True)
            torch.save({"state": lm.state_dict(), "cfg": ck["cfg"], "acc": acc, "kmax": kmax}, out)

if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 600, int(a[4]) if len(a) > 4 else 5, int(a[5]) if len(a) > 5 else 0)
