import json, os, random, sys, time, warnings, numpy as np, torch, torch.nn.functional as F
warnings.filterwarnings("ignore"); torch.set_num_threads(int(os.environ.get("THREADS", "1")))
from tasks import N_ASCII, BOS, SEP, VOCAB, Monitor
from model import CharLM, next_logits
from tibp import embed_box, margins_box
from analysis import Analyzer
from refine import verify_refine
import baselines as B

TEMPLATES = [("list files in ", "ls "), ("show ", "cat "), ("count lines of ", "wc -l "), ("search todo in ", "grep todo ")]
SPECIAL = "./\\;`|$(&><'\" _-~!"
NMAX, MAXLEN = 8, 24
CFG = dict(d=96, layers=2, heads=4, d_ff=192, max_len=MAXLEN)
san = lambda c: c if (chr(c).isalnum() or chr(c) in "._-/") else ord("_")
ALLOWED = {c for c in range(N_ASCII) if chr(c).isalnum() or chr(c) in " ._-/"}   # reference output alphabet
PROPS = {"META": [";", "|", "`", "$", "&", ">", "<"], "RM": ["rm"], "TRAV": ["../"]}

def example(t, n, rng):
    pi, po = TEMPLATES[t]
    arg = [ord(rng.choice(SPECIAL)) if rng.random() < 0.4 else rng.randrange(N_ASCII) for _ in range(n)]
    return [ord(c) for c in pi] + arg, [ord(c) for c in po] + [san(c) for c in arg]

def batch(t, n, Bsz, rng):
    xs, ys = zip(*(example(t, n, rng) for _ in range(Bsz)))
    x, y = torch.tensor(xs), torch.tensor(ys); L, m = x.shape[1], y.shape[1]
    toks = torch.cat([torch.full((Bsz, 1), BOS), x, torch.full((Bsz, 1), SEP), y[:, :-1]], 1)
    pos, seg = CharLM.layout(L, m - 1)
    return toks, pos[None].expand(Bsz, -1), seg[None].expand(Bsz, -1), y, L

def greedy(lm, x, m):
    ys = torch.zeros(x.shape[0], 0, dtype=torch.long)
    for _ in range(m):
        ys = torch.cat([ys, next_logits(lm, x, ys).argmax(-1, keepdim=True)], 1)
    return ys

def accuracy(lm, rng, n_per=128):
    lm.eval(); ok = tot = 0
    with torch.no_grad():
        for t in range(len(TEMPLATES)):
            for n in (4, 8):
                xs, ys = zip(*(example(t, n, rng) for _ in range(n_per)))
                pred = greedy(lm, torch.tensor(xs), len(ys[0]))
                ok += int((pred == torch.tensor(ys)).all(1).sum()); tot += n_per
    lm.train(); return ok / tot

def ce_step(lm, rng):
    t, n = rng.randrange(len(TEMPLATES)), rng.randint(1, NMAX)
    toks, pos, seg, y, L = batch(t, n, 64, rng)
    logits = lm(toks, pos, seg)[:, L + 1:, :]
    return F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))

def context_box(lm, t, n, j, eps):
    """Template prefix concrete, slot = ASCII box, earlier outputs = vocabulary boxes; radii scaled by eps."""
    pi, _ = TEMPLATES[t]; L = len(pi) + n
    pos, seg = CharLM.layout(L, j); cs, rs = [], []
    def add(E, p, g, e=eps):
        lo, hi = E.min(0).values, E.max(0).values
        cs.append((lo + hi) / 2 + lm.pos.weight[int(p)] + lm.seg.weight[int(g)]); rs.append(e * (hi - lo) / 2)
    W = lm.tok.weight
    add(W[BOS:BOS + 1], pos[0], seg[0])
    for i, ch in enumerate(pi): add(W[ord(ch):ord(ch) + 1], pos[1 + i], seg[1 + i])
    for i in range(n): add(W[:N_ASCII], pos[1 + len(pi) + i], seg[1 + len(pi) + i])
    add(W[SEP:SEP + 1], pos[L + 1], seg[L + 1])
    for i in range(j): add(W, pos[L + 2 + i], seg[L + 2 + i])
    return torch.stack(cs)[None], torch.stack(rs)[None]

def conf_loss(lm, eps, rng, K=50, gamma=1.0, n_ctx=2, log_hinge=True):
    bad = torch.tensor(sorted(set(range(VOCAB)) - ALLOWED)); tot = 0.0
    for _ in range(n_ctx):
        t, n = rng.randrange(len(TEMPLATES)), rng.randint(1, NMAX)
        j = rng.randint(0, len(TEMPLATES[t][1]) + n - 1)
        c, r = context_box(lm, t, n, j, eps)
        m = margins_box(lm, c, r)[0][0].clone(); m.fill_diagonal_(-1e4)
        kth = torch.topk(m[:, bad], K, dim=0).values[-1]
        h = F.softplus(gamma - kth)
        tot = tot + (torch.log1p(h) if log_hinge else h).mean()
    return tot / n_ctx

def train(seed=0, steps_std=6000, steps_conf=3000, steps_tune=2500):
    rng = random.Random(seed); torch.manual_seed(seed)
    lm = CharLM(**CFG); opt = torch.optim.AdamW(lm.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps_std, pct_start=0.1); t0 = time.time()
    for s in range(1, steps_std + 1):
        loss = ce_step(lm, rng); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step(); sched.step()
        if s % 500 == 0: print(f"[nl2sh std] step {s} ce {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    acc = accuracy(lm, random.Random(7)); print(f"[nl2sh std] accuracy {acc:.4f}", flush=True)
    torch.save({"state": lm.state_dict(), "cfg": CFG, "acc": acc}, "models/nl2sh.pt")
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-3, weight_decay=0.01); ramp = steps_conf // 2
    for s in range(1, steps_conf + 1):
        eps = min(1.0, s / ramp) ** 2; lam = min(1.0, s / ramp)
        loss = ce_step(lm, rng) + lam * conf_loss(lm, eps, rng)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step()
        if s % 250 == 0: print(f"[nl2sh conf] step {s} loss {loss.item():.4f} eps {eps:.2f} ({time.time()-t0:.0f}s)", flush=True)
    finetune(lm, rng, steps_tune, t0)

def finetune(lm, rng, steps, t0=None):
    """Continuation at full radius: target margin 2, plain softplus hinge, certified-loss weight 3."""
    t0 = t0 or time.time(); opt = torch.optim.AdamW(lm.parameters(), lr=5e-4, weight_decay=0.01)
    for s in range(1, steps + 1):
        loss = ce_step(lm, rng) + 3.0 * conf_loss(lm, 1.0, rng, gamma=2.0, log_hinge=False)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step()
        if s % 250 == 0: print(f"[nl2sh tune] step {s} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    acc = accuracy(lm, random.Random(7)); print(f"[nl2sh conf] accuracy {acc:.4f}", flush=True)
    torch.save({"state": lm.state_dict(), "cfg": CFG, "acc": acc}, "models/nl2sh_conf.pt")

def load(name):
    ck = torch.load(f"models/{name}.pt", weights_only=False); lm = CharLM(**ck["cfg"]); lm.load_state_dict(ck["state"]); lm.eval()
    return lm, ck["acc"]

def spec(t, n, alphabet=None):
    pi, po = TEMPLATES[t]; S = []
    for ch in pi: m = np.zeros(VOCAB, bool); m[ord(ch)] = True; S.append(m)
    for _ in range(n):
        m = np.zeros(VOCAB, bool); m[list(range(N_ASCII)) if alphabet is None else [ord(c) for c in alphabet]] = True; S.append(m)
    return S, len(po) + n

def verify(out="results/casestudy_nl2sh.jsonl"):
    f = open(out, "w")
    def log(r): f.write(json.dumps(r) + "\n"); f.flush(); print(json.dumps(r), flush=True)
    for name in ("nl2sh_conf", "nl2sh"):
        lm, acc = load(name); log(dict(kind="model", model=name, acc=acc, params=sum(p.numel() for p in lm.parameters())))
        for prop, fb in PROPS.items():
            for t in range(len(TEMPLATES)):
                for n in (4, 8):
                    for k in (1, 2, 5, 10):
                        S, Lout = spec(t, n); t0 = time.time()
                        r = Analyzer(lm, Monitor(fb), k, backend="tibp").run(S, Lout, mode="join", time_limit=120)
                        rec = dict(kind="verify", model=name, prop=prop, template=TEMPLATES[t][0].strip(), n=n, k=k,
                                   inputs=f"128^{n}", verdict=r["verdict"], secs=round(time.time() - t0, 3))
                        if r["verdict"] != "VERIFIED" and n == 4 and k in (1, 2):
                            s = B.sample(lm, Monitor(fb), S, k, Lout, n=3000, time_limit=20)
                            if s["result"] == "UNSAFE":
                                enc = lambda v: v if isinstance(v, str) else "".join(chr(c) if 32 <= c < 127 else "?" for c in v)
                                rec.update(cex_in=enc(s["input"]), cex_out=enc(s["output"]))
                        log(rec)
        # refinement on a small template (slot over 12 characters) for the metacharacter property
        alpha = "ab;|`$&><./_"
        for k in (1, 2):
            S, Lout = spec(1, 2, alpha)
            r = verify_refine(lm, Monitor(PROPS["META"]), S, k, L_out=Lout, budget=600, time_limit=120)
            log(dict(kind="refine", model=name, prop="META", template="show ??", alphabet=alpha, k=k, **r))

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("train", "all"): train()
    if what == "tune":                                   # continue from an existing conformance model
        lm, _ = load("nl2sh_conf"); lm.train(); finetune(lm, random.Random(1), 2500); what = "verify"
    if what in ("verify", "all"): verify()
