import os, sys, random, json, torch, transformers as T, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bounds as RS
from test_equiv import CFGS, V
rng = random.Random(1); torch.manual_seed(1)
CH = "abcdefgh./;|_- "
tb = [b""] + [bytes([ord(rng.choice(CH)) for _ in range(rng.randint(1, 3))]) for _ in range(V - 1)]
special = {0}
summary = {}
for fam, cfg in CFGS.items():
    m = T.AutoModelForCausalLM.from_config(cfg).float().eval()
    with torch.no_grad():
        for p in m.parameters(): p.mul_(4.0) if p.dim() > 1 else None
    ad = RS.make_adapter(m); counts = dict(n=0, unsafe=0, verified=0, violations=0, refine_ok=0, refine_n=0)
    for it in range(14):
        L_out = rng.choice([2, 3]); k = rng.choice([1, 2])
        prompt = [torch.tensor([rng.randrange(1, V)]) for _ in range(rng.randint(2, 5))]
        slots = [torch.tensor(rng.sample(range(1, V), rng.randint(1, 3))) for _ in range(rng.randint(1, 2))]
        S = prompt[:1] + slots + prompt[1:]
        # forbidden pattern: bytes of a likely or an unlikely next token
        z = ad.logits([int(s[0]) for s in S]); order = torch.argsort(z, descending=True).tolist()
        t = order[rng.choice([0, 1, 5, V // 2, V - 2])]
        pat = tb[t] if tb[t] else b"."
        mon = RS.TokenMonitor([pat], tb, special)
        truth = RS.enumerate_truth(ad, mon, S, k, L_out)
        r = RS.abstract_decode(ad, mon, S, k, L_out, M=32)
        counts["n"] += 1; counts["unsafe"] += truth == "UNSAFE"; counts["verified"] += r["verdict"] == "VERIFIED"
        counts["violations"] += (r["verdict"] == "VERIFIED" and truth == "UNSAFE")
        rf = RS.refine(ad, mon, S, k, L_out, M=32, budget=200)
        counts["refine_n"] += 1
        counts["refine_ok"] += (rf["verdict"] == "VERIFIED" and truth == "SAFE") or (rf["verdict"] == "COUNTEREXAMPLE" and truth == "UNSAFE")
        if rf["verdict"] == "COUNTEREXAMPLE":                       # the witness must be a real violation
            ids, y = rf["witness"]; q = 0
            for tok in y: q = int(mon.trans[q][tok]) if q >= 0 else q
            assert q == RS.ByteMonitor.BAD, "refinement returned a spurious counterexample"
    summary[fam] = counts
    print(fam, json.dumps(counts), flush=True)
print("TOTAL violations:", sum(c["violations"] for c in summary.values()),
      "| refinement agrees with ground truth:", sum(c["refine_ok"] for c in summary.values()), "/", sum(c["refine_n"] for c in summary.values()))
