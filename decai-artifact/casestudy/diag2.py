"""Abstract decoding for META with either the hull oracle (tibp) or the row oracle (tibp_rows); prints per step."""
import os
import sys, numpy as np, torch, time
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import casestudy_nl2sh as CS
from tasks import Monitor, VOCAB, N_ASCII, BOS, SEP
from model import CharLM
from tibp import embed_box, margins_box
from tibp_rows import margins_rows
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
ck = torch.load(sys.argv[1], weights_only=False); lm = CharLM(**ck["cfg"]); lm.load_state_dict(ck["state"]); lm.eval()
oracle = sys.argv[2]; ks = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1]
META = set(ord(c) for c in ";|`$&><")
def cands(S_in, S_out, k):
    L, n = len(S_in), len(S_out); pos, seg = CharLM.layout(L, n)
    sets = [torch.tensor([BOS])] + [torch.tensor(s) for s in S_in] + [torch.tensor([SEP])] + [torch.tensor(sorted(s)) for s in S_out]
    with torch.no_grad():
        if oracle == "hull":
            c, r = embed_box(lm, sets, pos, seg); M = margins_box(lm, c, r)[0]
        else:
            M = margins_rows(lm, sets, pos, seg, rows_max=int(sys.argv[5]) if len(sys.argv) > 5 else 10**9)
    out = set()
    for m in M:
        m = m.clone(); m.fill_diagonal_(-1e4)
        out |= set(torch.nonzero((m > 0).sum(0) < k).flatten().tolist())
    return out
tot = ver = 0
for t in range(4):
    for n in (4, 8):
        for k in ks:
            pi, po = CS.TEMPLATES[t]; S_in = [[ord(c)] for c in pi] + [list(range(N_ASCII))] * n
            S_out = []; Lout = len(po) + n; t0 = time.time(); verdict = "VERIFIED"; sizes = []
            for j in range(Lout):
                C = cands(S_in, S_out, k); sizes.append(len(C))
                if C & META: verdict = f"ALARM@{j}"; break
                S_out.append(C)
            tot += 1; ver += verdict == "VERIFIED"
            print(f"{pi.strip():16s} n={n} k={k:2d} {verdict:10s} |cand| per step {sizes}  {time.time()-t0:.1f}s", flush=True)
print(f"verified {ver}/{tot}")
