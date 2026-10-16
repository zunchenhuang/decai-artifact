import os
import sys, time, random, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import casestudy_nl2sh as CS
from tasks import VOCAB, N_ASCII, BOS, SEP
from model import CharLM
from tibp import margins_box
torch.set_num_threads(2)

ALLOWED = sorted(CS.ALLOWED)
ARG_ALPHA = sorted({CS.san(c) for c in range(N_ASCII)})          # reference outputs at argument positions
BAD = torch.tensor(sorted(set(range(VOCAB)) - CS.ALLOWED))

def box_from_sets(lm, sets, pos, seg, eps):
    W = lm.tok.weight; cs, rs = [], []
    for s, p, g in zip(sets, pos, seg):
        E = W[s]; lo, hi = E.min(0).values, E.max(0).values
        cs.append((lo + hi) / 2 + lm.pos.weight[int(p)] + lm.seg.weight[int(g)]); rs.append(eps * (hi - lo) / 2)
    return torch.stack(cs)[None], torch.stack(rs)[None]

def context(lm, t, n, j, eps, kmax, rng):
    pi, po = CS.TEMPLATES[t]; L = len(pi) + n
    pos, seg = CharLM.layout(L, j)
    sets = [[BOS]] + [[ord(c)] for c in pi] + [list(range(N_ASCII))] * n + [[SEP]]
    for i in range(j):
        if i < len(po):                                     # command part: reference token + a few others
            s = {ord(po[i])}
            if kmax > 1 and rng.random() < 0.5:
                s |= set(rng.sample(ALLOWED, rng.randint(1, kmax - 1)))
            sets.append(sorted(s))
        else:                                               # argument part: the sanitised alphabet
            sets.append(ARG_ALPHA)
    return box_from_sets(lm, [torch.tensor(s) for s in sets], pos, seg, eps)

def conf_loss(lm, eps, rng, kmax, gamma=1.0, n_ctx=2):
    tot, ok = 0.0, 0
    for _ in range(n_ctx):
        t, n = rng.randrange(len(CS.TEMPLATES)), rng.choice([2, 4, 4, 6, 8, 8])
        j = rng.randint(0, len(CS.TEMPLATES[t][1]) + n - 1)
        c, r = context(lm, t, n, j, eps, kmax, rng)
        m = margins_box(lm, c, r)[0][0].clone(); m.fill_diagonal_(-1e4)
        top = torch.topk(m[:, BAD], kmax, dim=0).values                 # (kmax, |BAD|)
        k1, kk = top[0], top[-1]
        h = F.softplus(gamma - k1).mean() + F.softplus(gamma - kk).mean()
        tot = tot + torch.log1p(h); ok += int(kk.min() > 0)
    return tot / n_ctx, ok / n_ctx

def main(out, steps=4000, seed=0, kmax=10, init="models/nl2sh.pt"):
    rng = random.Random(seed); torch.manual_seed(seed)
    ck = torch.load(init, weights_only=False); lm = CharLM(**ck["cfg"]); lm.load_state_dict(ck["state"]); lm.train()
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-3, weight_decay=0.01); ramp = int(0.6 * steps); t0 = time.time()
    hist = []
    for s in range(1, steps + 1):
        eps = min(1.0, s / ramp) ** 2; lam = min(1.0, s / ramp)
        cl, ok = conf_loss(lm, eps, rng, kmax)
        loss = CS.ce_step(lm, rng) + lam * cl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step()
        hist.append(ok)
        if s % 100 == 0:
            print(f"[cs2] step {s} loss {loss.item():.4f} conf {cl.item():.3f} certified {sum(hist[-100:])/100:.2f} eps {eps:.3f} ({time.time()-t0:.0f}s)", flush=True)
    lm.eval(); acc = CS.accuracy(lm, random.Random(7)); print(f"[cs2] accuracy {acc:.4f}", flush=True)
    torch.save({"state": lm.state_dict(), "cfg": ck["cfg"], "acc": acc, "kmax": kmax}, out)

if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 4000, int(a[3]) if len(a) > 3 else 0, int(a[4]) if len(a) > 4 else 10,
         a[5] if len(a) > 5 else "models/nl2sh.pt")
