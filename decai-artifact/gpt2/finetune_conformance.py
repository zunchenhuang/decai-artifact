import argparse, json, math, os, random, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bounds as RS
from conformance_task import ShellTask, load_model, L_TGT

def cert_loss(ad, task, rng, s, gamma, K, n_hard, n_comp, split="train", full_frac=0.3):
    """Certified loss of one abstract context, and whether its certificates already hold."""
    verb = rng.randrange(len(task.verbs)); slot = task.sample_slot(rng, s, split)
    j = rng.randrange(L_TGT)
    outs = [task.target(verb, x) for x in slot]
    pre = []
    for i in range(j):
        allowed_i = [e for e in torch.nonzero(task.allowed[i]).flatten().tolist() if e != task.eos]
        if rng.random() < full_frac:                            # the whole output class, as a joined abstract state
            S = set(allowed_i)                                  # (more than ROWS_MAX tokens: one box row)
        else:
            S = {o[i] for o in outs}                            # the reference outputs for the slot tokens
            if rng.random() < 0.5:                              # plus extra grammar tokens, as top-2 decoding adds
                S |= set(rng.sample(allowed_i, min(len(allowed_i), rng.randint(1, 2))))
        pre.append(torch.tensor(sorted(S)))
    sets = task.prompt_sets(verb, slot) + pre
    cs, rs = ad.final_rows_grad(sets)
    G = ad.G; Ga = G.abs(); allowed = task.allowed[j]
    losses, ok = [], True
    for c, r in zip(cs, rs):
        zc = G @ c; zr = Ga @ r
        ub = (zc + zr).masked_fill(allowed, float("-inf")); lb = (zc - zr).masked_fill(~allowed, float("-inf"))
        t = torch.topk(ub, min(n_hard, int((~allowed).sum()))).indices            # hardest excluded tokens
        J = torch.topk(lb, min(n_comp, int(allowed.sum()))).indices               # strongest allowed competitors
        D = G[J][:, None, :] - G[t][None, :, :]
        m = D @ c - D.abs() @ r                                                  # (|J|, |t|) certified margins
        kth = torch.topk(m, min(K, len(J)), dim=0).values[-1]
        losses.append(torch.log1p(F.softplus(gamma - kth)).mean())
        ok &= bool((kth > 0).all())
    return torch.stack(losses).mean(), ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2"); ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--init", default=None, help="start from this checkpoint"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--contexts", type=int, default=4)
    ap.add_argument("--s_max", type=int, default=16); ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=1); ap.add_argument("--w_cert", type=float, default=0.3)
    ap.add_argument("--n_hard", type=int, default=256); ap.add_argument("--n_comp", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ramp_frac", type=float, default=0.6, help="fraction of training over which radii and s are ramped")
    ap.add_argument("--ramp_min", type=float, default=1e-4, help="initial scale of the hull radii (geometric ramp to 1)")
    ap.add_argument("--probe", action="store_true", help="print certified fractions for ramps 1e-10..1 and exit")
    ap.add_argument("--log_every", type=int, default=25); ap.add_argument("--save_every", type=int, default=500)
    a = ap.parse_args()
    if a.threads: torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    model, tok = load_model(a.model, a.layers, a.synthetic, a.init)
    V = model.get_output_embeddings().weight.shape[0]
    task = ShellTask(tok, V)
    print(f"verbs {[v for v, _ in task.verbs]}; pools: " + ", ".join(f"{k[0]}/{k[1]} {len(v)}" for k, v in task.pool.items()), flush=True)
    ad = RS.Adapter(model, live=True)
    if a.probe:                                                     # how robust is the starting model?  picks ramp_min
        for s_ in (2, a.s_max):
            for e in range(-10, 1):
                ad.ramp = 10.0 ** e; prng = random.Random(1); res = []
                with torch.no_grad():
                    for _ in range(24): res.append(cert_loss(ad, task, prng, s_, a.gamma, a.K, a.n_hard, a.n_comp))
                print(f"s={s_:3d} ramp=1e{e:+d}: certified {sum(o for _, o in res):2d}/24, loss {sum(float(l) for l, _ in res) / 24:.2f}", flush=True)
        print("choose --ramp_min near the largest ramp at which most contexts are certified for s=2")
        return
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "conformance_args.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    t0 = time.time(); hist = []
    for step in range(1, a.steps + 1):
        prog = min(1.0, step / (a.ramp_frac * a.steps))
        ad.ramp = a.ramp_min ** (1 - prog)                       # geometric: ramp_min -> 1, equal time per decade
        warm = prog
        s_cur = max(2, int(round(2 ** (math.log2(a.s_max) * prog))))
        w = a.w_cert * min(1.0, step / (0.25 * a.ramp_frac * a.steps + 1))
        # task loss on concrete examples (training pools, a quarter risky)
        xs = [(rng.randrange(len(task.verbs)), task.sample_slot(rng, 1, "train")[0]) for _ in range(a.batch)]
        seqs = [task.prompt_ids(v, x) + task.target(v, x) for v, x in xs]
        Lmax = max(map(len, seqs))
        ids = torch.full((len(seqs), Lmax), task.eos); lab = torch.full((len(seqs), Lmax), -100)
        for b, (sq, (v, x)) in enumerate(zip(seqs, xs)):
            ids[b, :len(sq)] = torch.tensor(sq); n = len(task.prompt_ids(v, x))
            lab[b, n:len(sq)] = torch.tensor(sq[n:])
        ce = model(ids, labels=lab).loss
        # certified term
        cl, oks = [torch.zeros(())], []
        for _ in range(a.contexts if a.w_cert > 0 else 0):
            l, ok = cert_loss(ad, task, rng, rng.randint(2, s_cur), a.gamma, a.K, a.n_hard, a.n_comp)
            cl.append(l); oks.append(ok)
        cert = torch.stack(cl[1:] or cl).mean()
        loss = ce + w * cert
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        hist.append((ce.item(), cert.item(), sum(oks) / max(1, len(oks))))
        if step % a.log_every == 0 or step == a.steps:
            h = hist[-a.log_every:]
            rec = dict(step=step, ce=sum(x[0] for x in h) / len(h), cert=sum(x[1] for x in h) / len(h),
                       certified=sum(x[2] for x in h) / len(h), ramp=float(f"{ad.ramp:.3g}"), s=s_cur, w=round(w, 3), secs=round(time.time() - t0))
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if step % a.save_every == 0 or step == a.steps:
            model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print("saved to", a.out)

if __name__ == "__main__":
    main()
