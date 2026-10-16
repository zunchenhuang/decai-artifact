import sys, os, time, random, torch, torch.nn.functional as F
import ibp_plain; ibp_plain.install()
import train_cert as TC
from model import CharLM
from train import batch, evaluate

def main(task, steps, K=50, seed=0):
    torch.manual_seed(seed); rng = random.Random(seed)
    ck = torch.load(f"models/{task}.pt", weights_only=False)
    lm = CharLM(**ck["cfg"]); lm.load_state_dict(ck["state"]); lm.train()
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-3, weight_decay=0.01)
    spec = TC.CERT[task]; t0 = time.time(); ramp = int(0.5 * steps)
    for step in range(1, steps + 1):
        eps = min(1.0, step / ramp); lam = min(1.0, step / ramp)
        L = rng.choice([1, 2, 3, 4, 6, 8, 12, 16, 16])
        toks, pos, seg, y = batch(task, L, 64, rng)
        logits = lm(toks, pos, seg)[:, L + 1:, :]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        cl, worst = TC.cert_loss(lm, spec, K, eps, rng)
        loss = ce + lam * cl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step()
        if step % 100 == 0 or step == steps:
            print(f"[{task} ibp] step {step} eps {eps:.2f} ce {ce.item():.4f} cert {cl.item():.3f} worst {worst:.2f} ({time.time()-t0:.0f}s)", flush=True)
    lm.eval(); acc = evaluate(lm, task, random.Random(123))
    print(f"[{task} ibp] accuracy {acc}", flush=True)
    torch.save({"state": lm.state_dict(), "cfg": lm.cfg, "task": task, "acc": acc, "K": K, "trainer": "plain-IBP"},
               f"models/{task}_ibpcert.pt")

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 50)
