import math, torch, torch.nn.functional as F
import tibp as T
from model import Block, CharLM

class BlockGELU(Block):
    def forward(self, x, mask):
        B, Tn, d = x.shape
        a = self.ln1(x)
        q = self.q(a).reshape(B, Tn, self.h, self.dh).permute(0, 2, 1, 3)
        k = self.k(a).reshape(B, Tn, self.h, self.dh).permute(0, 2, 3, 1)
        v = self.v(a).reshape(B, Tn, self.h, self.dh).permute(0, 2, 1, 3)
        s = torch.matmul(q, k) / math.sqrt(self.dh) + mask
        z = torch.matmul(torch.softmax(s, dim=-1), v).permute(0, 2, 1, 3).reshape(B, Tn, d)
        x = x + self.o(z)
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))

def make_std(**cfg):
    lm = CharLM(**{**cfg, "ln": "full"})
    for b in lm.blocks: b.__class__ = BlockGELU
    return lm

def load_std(name):
    ck = torch.load(f"models/{name}.pt", weights_only=False)
    lm = make_std(**ck["cfg"]); lm.load_state_dict(ck["state"]); return lm.eval()

# ------------------------------------------------------------------ sound bounds
def norm_box(c, r, w, b, eps):
    """LayerNorm with variance over the last dimension; exact if r == 0 (as in real_slm.norm_box)."""
    d = c.shape[-1]
    nc = c - c.mean(-1, keepdim=True)
    nr = r * (1 - 2.0 / d) + r.sum(-1, keepdim=True) / d
    lo, hi = nc - nr, nc + nr
    lb2 = torch.where(lo > 0, lo * lo, torch.where(hi < 0, hi * hi, torch.zeros_like(lo)))
    ub2 = torch.maximum(lo * lo, hi * hi)
    ms_lo, ms_hi = lb2.mean(-1, keepdim=True), ub2.mean(-1, keepdim=True)
    rn = r.norm(dim=-1, keepdim=True); cn = nc.norm(dim=-1, keepdim=True); sym = rn > 0
    ms_lo = torch.where(sym, torch.maximum(ms_lo, (cn - rn).clamp_min(0) ** 2 / d), ms_lo)
    ms_hi = torch.where(sym, torch.minimum(ms_hi, (cn + rn) ** 2 / d), ms_hi)
    sl, su = torch.sqrt(ms_lo + eps), torch.sqrt(ms_hi + eps)
    ylo = torch.minimum(lo / sl, lo / su); yhi = torch.maximum(hi / sl, hi / su)
    cap = math.sqrt(d); ylo, yhi = ylo.clamp(-cap, cap), yhi.clamp(-cap, cap)
    yc, yr = (ylo + yhi) / 2, (yhi - ylo) / 2
    return yc * w + b, yr * w.abs()

_xs = torch.linspace(-4.0, 0.0, 400001, dtype=torch.float64); _ys = F.gelu(_xs); _i = int(_ys.argmin())
GELU_XMIN, GELU_YMIN = float(_xs[_i]), float(_ys[_i]) - 1e-6

def gelu_interval(lo, hi):
    flo, fhi = F.gelu(lo), F.gelu(hi)
    straddle = (lo < GELU_XMIN + 1e-4) & (hi > GELU_XMIN - 1e-4) & (hi > lo)
    low = torch.where(straddle, torch.minimum(torch.minimum(flo, fhi), torch.full_like(lo, GELU_YMIN)), torch.minimum(flo, fhi))
    return low, torch.maximum(flo, fhi)

def block_box_std(blk, c, r, valid, H):
    B, Tn, d = c.shape; dh = d // H
    ac, ar = norm_box(c, r, blk.ln1.weight, blk.ln1.bias, blk.ln1.eps)
    def heads(m):
        x, y = T.lin(ac, ar, m.weight, m.bias)
        f = lambda z: z.reshape(B, Tn, H, dh).permute(0, 2, 1, 3)
        return f(x), f(y)
    qc, qr = heads(blk.q); kc, kr = heads(blk.k); vc, vr = heads(blk.v)
    sc = qc @ kc.transpose(-1, -2) / math.sqrt(dh)
    sr = (qc.abs() @ kr.transpose(-1, -2) + qr @ kc.abs().transpose(-1, -2) + qr @ kr.transpose(-1, -2)) / math.sqrt(dh)
    w_lo, w_hi = T.softmax_bounds(sc - sr, sc + sr, valid)
    z_hi = T.knapsack_avg(w_lo, w_hi, vc + vr, True); z_lo = T.knapsack_avg(w_lo, w_hi, vc - vr, False)
    zc = ((z_hi + z_lo) / 2).permute(0, 2, 1, 3).reshape(B, Tn, d)
    zr = ((z_hi - z_lo) / 2).clamp_min(0).permute(0, 2, 1, 3).reshape(B, Tn, d)
    oc, orr = T.lin(zc, zr, blk.o.weight, blk.o.bias)
    c, r = c + oc, r + orr
    hc, hr = norm_box(c, r, blk.ln2.weight, blk.ln2.bias, blk.ln2.eps)
    hc, hr = T.lin(hc, hr, blk.ff1.weight, blk.ff1.bias)
    lo, hi = gelu_interval(hc - hr, hc + hr)
    gc, gr = T.lin((lo + hi) / 2, (hi - lo) / 2, blk.ff2.weight, blk.ff2.bias)
    return c + gc, r + gr

def margins_box_std(lm, c, r):
    B, Tn, d = c.shape; H = lm.cfg["heads"]; valid = torch.ones(Tn, Tn).tril().bool()
    for blk in lm.blocks: c, r = block_box_std(blk, c, r, valid, H)
    c, r = norm_box(c[:, -1], r[:, -1], lm.lnf.weight, lm.lnf.bias, lm.lnf.eps)
    G, gb = lm.head.weight, lm.head.bias
    zc, zr = T.lin(c, r, G, gb)
    Dm = G[:, None, :] - G[None, :, :]
    mlb = torch.einsum("jtd,bd->bjt", Dm, c) - torch.einsum("jtd,bd->bjt", Dm.abs(), r) + (gb[:, None] - gb[None, :])
    return mlb, zc - zr, zc + zr

_ORIG = (T.block_box, T.margins_box)
def install():
    import train_cert as TC
    T.block_box, T.margins_box = block_box_std, margins_box_std; TC.margins_box = margins_box_std
def uninstall():
    import train_cert as TC
    T.block_box, T.margins_box = _ORIG; TC.margins_box = _ORIG[1]
