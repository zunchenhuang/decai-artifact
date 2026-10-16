import json, os, sys, time, random, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import casestudy_nl2sh as CS
import analysis
from tasks import Monitor, BOS, SEP, VOCAB
from model import CharLM
from refine import verify_refine
import baselines as B
from tibp_rows import margins_rows

ORACLE = "rows"

def _candidates_rows(self, S_in, S_out):
    L, n = len(S_in), len(S_out); pos, seg = CharLM.layout(L, n)
    sets = [torch.tensor([BOS])] + [torch.from_numpy(np.flatnonzero(S)) for S in S_in] + \
           [torch.tensor([SEP])] + [torch.from_numpy(np.flatnonzero(S)) for S in S_out]
    with torch.no_grad():
        M = margins_rows(self.lm, sets, pos, seg)
    self.stats["bound_calls"] += 1; cand = set()
    for m in M:
        m = m.clone(); m.fill_diagonal_(-1.0)
        above = (m > self.tau).sum(0)
        excl = torch.nonzero(above >= self.k).flatten()
        if len(excl):
            kth = torch.topk(m[:, excl], self.k, dim=0).values[-1]
            self.min_used_margin = min(self.min_used_margin, float(kth.min()))
        cand |= set(torch.nonzero(above < self.k).flatten().tolist())
    return sorted(cand)

_orig = analysis.Analyzer._candidates_tibp
def _dispatch(self, S_in, S_out):
    return _candidates_rows(self, S_in, S_out) if ORACLE == "rows" else _orig(self, S_in, S_out)
analysis.Analyzer._candidates_tibp = _dispatch

def load(path):
    ck = torch.load(path, weights_only=False); lm = CharLM(**ck["cfg"]); lm.load_state_dict(ck["state"]); lm.eval()
    return lm, ck["acc"]

def train():
    import cs_train2, cs_train3
    if not os.path.exists("models/nl2sh.pt"):
        CS.train()                                          # also writes the original conformance model
    cs_train2.main("models/nl2sh_conf2.pt", 4000, 0, 10, "models/nl2sh.pt")
    cs_train3.main("models/nl2sh_conf2.pt", "models/nl2sh_conf_v2.pt", 600, 5, 0)

def verify(models=("nl2sh_conf_v2", "nl2sh_conf", "nl2sh"), out="results/casestudy_nl2sh_v2.jsonl"):
    f = open(out, "a")
    def log(r): r["oracle"] = ORACLE; f.write(json.dumps(r) + "\n"); f.flush(); print(json.dumps(r), flush=True)
    enc = lambda v: v if isinstance(v, str) else "".join(chr(c) if 32 <= c < 127 else "?" for c in v)
    for name in models:
        lm, acc = load(f"models/{name}.pt"); log(dict(kind="model", model=name, acc=acc, params=sum(p.numel() for p in lm.parameters())))
        for prop, fb in CS.PROPS.items():
            for t in range(len(CS.TEMPLATES)):
                for n in (4, 8):
                    for k in (1, 2, 5, 10):
                        S, Lout = CS.spec(t, n); t0 = time.time()
                        an = analysis.Analyzer(lm, Monitor(fb), k, backend="tibp")
                        r = an.run(S, Lout, mode="join", time_limit=600)
                        rec = dict(kind="verify", model=name, prop=prop, template=CS.TEMPLATES[t][0].strip(), n=n, k=k,
                                   verdict=r["verdict"], step=r.get("step"), secs=round(time.time() - t0, 2),
                                   bound_calls=r.get("bound_calls"), min_margin=an.min_used_margin)
                        if r["verdict"] != "VERIFIED" and n == 4 and k in (1, 2):
                            s = B.sample(lm, Monitor(fb), S, k, Lout, n=3000, time_limit=20)
                            if s["result"] == "UNSAFE": rec.update(cex_in=enc(s["input"]), cex_out=enc(s["output"]))
                        log(rec)
        for prop, tmpl, alpha in (("META", 1, "ab;|`$&><./_"), ("TRAV", 1, "./ab")):
            for k in (1, 2):
                n = 2 if prop == "META" else 3
                S, Lout = CS.spec(tmpl, n, alpha)
                r = verify_refine(lm, Monitor(CS.PROPS[prop]), S, k, L_out=Lout, budget=600, time_limit=300)
                log(dict(kind="refine", model=name, prop=prop, template="show " + "?" * n, alphabet=alpha, k=k, **r))

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if "--oracle" in sys.argv: ORACLE = sys.argv[sys.argv.index("--oracle") + 1]
    if what == "train": train()
    if what == "verify": verify()
