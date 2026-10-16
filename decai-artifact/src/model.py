import math, torch, torch.nn as nn
from tasks import VOCAB, BOS, SEP, MAX_L

class LayerNormNoVar(nn.Module):
    """Mean-centering LayerNorm without variance division, as used in transformer verification
    work (Shi et al., ICLR 2020); switchable to full LayerNorm with ln='full'."""
    def __init__(self, d):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d)); self.bias = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        return self.weight * (x - x.mean(-1, keepdim=True)) + self.bias

def make_ln(d, kind): return nn.LayerNorm(d) if kind == "full" else LayerNormNoVar(d)

class Block(nn.Module):
    def __init__(self, d, h, d_ff, ln):
        super().__init__()
        self.h, self.dh = h, d // h
        self.ln1, self.ln2 = make_ln(d, ln), make_ln(d, ln)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.ff1, self.ff2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)

    def forward(self, x, mask):                          # x: (B,T,d); mask: (T,T) additive
        B, T, d = x.shape
        a = self.ln1(x)
        q = self.q(a).reshape(B, T, self.h, self.dh).permute(0, 2, 1, 3)
        k = self.k(a).reshape(B, T, self.h, self.dh).permute(0, 2, 3, 1)
        v = self.v(a).reshape(B, T, self.h, self.dh).permute(0, 2, 1, 3)
        s = torch.matmul(q, k) / math.sqrt(self.dh) + mask
        att = torch.softmax(s, dim=-1)
        z = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(B, T, d)
        x = x + self.o(z)
        return x + self.ff2(torch.relu(self.ff1(self.ln2(x))))

class CharLM(nn.Module):
    def __init__(self, d=64, layers=2, heads=4, d_ff=128, ln="novar", max_len=MAX_L):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, d_ff=d_ff, ln=ln, max_len=max_len)
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(max_len + 1, d)
        self.seg = nn.Embedding(2, d)
        self.blocks = nn.ModuleList(Block(d, heads, d_ff, ln) for _ in range(layers))
        self.lnf = make_ln(d, ln)
        self.head = nn.Linear(d, VOCAB)

    # ---- sequence layout:  BOS x1..xL SEP y1..y_{t}   (pos: 0,1..L,0,1..t ; seg: 0..0,1..1) ----
    @staticmethod
    def layout(L, n_out):
        pos = [0] + list(range(1, L + 1)) + [0] + list(range(1, n_out + 1))
        seg = [0] * (L + 1) + [1] * (n_out + 1)
        return torch.tensor(pos), torch.tensor(seg)

    def embed_parts(self, tokens, pos, seg):
        return self.tok(tokens) + self.pos(pos) + self.seg(seg)

    @staticmethod
    def causal_mask(T):
        return torch.triu(torch.full((T, T), -1e4), diagonal=1)

    def body_all(self, E):
        mask = self.causal_mask(E.shape[1])
        for b in self.blocks: E = b(E, mask)
        return self.head(self.lnf(E))

    def forward(self, tokens, pos, seg):
        return self.body_all(self.embed_parts(tokens, pos, seg))

class LastStep(nn.Module):
    """E (1,T,d) -> logits of the last position (1,VOCAB). Fixed T so auto_LiRPA can trace it."""
    def __init__(self, lm, T):
        super().__init__(); self.lm = lm
        self.register_buffer("mask", CharLM.causal_mask(T))
    def forward(self, E):
        for b in self.lm.blocks: E = b(E, self.mask)
        return self.lm.head(self.lm.lnf(E[:, -1, :]))

# ---------------- concrete decoding helpers ----------------
@torch.no_grad()
def next_logits(lm, x, ys):
    """Batched: x (B,L) ints, ys (B,t) ints -> logits (B,VOCAB) for y_{t+1}."""
    B, L = x.shape; t = ys.shape[1]
    toks = torch.cat([torch.full((B, 1), BOS), x, torch.full((B, 1), SEP), ys], 1)
    pos, seg = CharLM.layout(L, t)
    return lm(toks, pos[None].expand(B, -1), seg[None].expand(B, -1))[:, -1, :]

@torch.no_grad()
def greedy(lm, x):
    B, L = x.shape; ys = torch.zeros(B, 0, dtype=torch.long)
    for _ in range(L):
        ys = torch.cat([ys, next_logits(lm, x, ys).argmax(-1, keepdim=True)], 1)
    return ys
