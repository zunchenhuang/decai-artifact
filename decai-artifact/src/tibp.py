"""Tailored interval bound propagation (T-IBP) for the char transformer.

Sound, differentiable and cheap: it is both the certified-training bound and a verification backend.
Representation: centre/radius boxes (c, r) of shape (B, T, d).

Tailored ingredients (beyond plain IBP):
  1. hull boxes for token sets: per position, the componentwise [min,max] of the allowed tokens'
     embeddings; concrete positions have r = 0;
  2. LN-NoVar is linear, so it is composed with the adjacent linear layer before concretisation;
  3. attention averages  out = sum_i w_i v_i  with w_i in [wl_i, wu_i], sum_i w_i = 1, v_i in [vl_i, vu_i]:
     the exact per-coordinate max/min is a fractional knapsack, solved by sorting (plain interval
     products ignore sum w = 1 and are much looser);
  4. margins z_j - z_t are bounded exactly over the final box, for all pairs at once.
"""
import math, torch

def lin(c, r, W, b=None):
    return c @ W.T + (b if b is not None else 0.0), r @ W.abs().T

def ln_mat(ln, d):
    return ln.weight[:, None] * (torch.eye(d) - 1.0 / d), ln.bias          # x -> M x + b

def ln_lin(ln, W, b):
    """Compose  W (LN(x)) + b  into a single affine map."""
    M, bl = ln_mat(ln, W.shape[1])
    return W @ M, W @ bl + b

def softmax_bounds(s_lo, s_hi, valid):
    """Exact interval bounds of softmax weights given independent score intervals. (..., Tq, Tk)."""
    NEG = -1e9
    s_lo = s_lo.masked_fill(~valid, NEG); s_hi = s_hi.masked_fill(~valid, NEG)
    Tk = s_lo.shape[-1]; eye = torch.eye(Tk, dtype=torch.bool)
    # w_lo_i = 1 / (1 + sum_{j != i} exp(s_hi_j - s_lo_i)),  w_hi_i symmetric
    other_hi = s_hi[..., None, :].expand(*s_hi.shape, Tk).masked_fill(eye, NEG)
    other_lo = s_lo[..., None, :].expand(*s_lo.shape, Tk).masked_fill(eye, NEG)
    w_lo = torch.sigmoid(s_lo - torch.logsumexp(other_hi, -1)).masked_fill(~valid, 0.0)
    w_hi = torch.sigmoid(s_hi - torch.logsumexp(other_lo, -1)).masked_fill(~valid, 0.0)
    return w_lo, w_hi

def knapsack_avg(w_lo, w_hi, v, maximize):
    """max (min) of sum_i w_i v_i  s.t. w in [w_lo, w_hi], sum w = 1.
    w_*: (..., Tq, Tk); v: (..., Tk, D) = the extreme value per coordinate. Returns (..., Tq, D)."""
    sgn = 1.0 if maximize else -1.0
    vv = sgn * v
    base = torch.einsum("...qk,...kd->...qd", w_lo, vv)
    R = (1.0 - w_lo.sum(-1, keepdim=True)).clamp_min(0.0)                   # (..., Tq, 1)
    order = torch.argsort(vv, dim=-2, descending=True)                       # (..., Tk, D)
    v_sorted = torch.gather(vv, -2, order)
    cap = (w_hi - w_lo).clamp_min(0.0)                                       # (..., Tq, Tk)
    D = vv.shape[-1]
    cap_e = cap[..., None].expand(*cap.shape, D)                             # (..., Tq, Tk, D)
    idx = order[..., None, :, :].expand(*cap.shape, D)
    cap_sorted = torch.gather(cap_e, -2, idx)
    before = torch.cumsum(cap_sorted, -2) - cap_sorted
    alloc = torch.minimum(cap_sorted, (R[..., None] - before).clamp_min(0.0))
    val = base + (alloc * v_sorted[..., None, :, :]).sum(-2)
    left = (R - alloc.sum(-2)).clamp_min(0.0)                                # float slack only
    val = val + left * vv.amax(-2)[..., None, :]
    return sgn * val

def block_box(blk, c, r, valid, H):
    B, T, d = c.shape; dh = d // H
    def heads(lin_mod):
        W, b = ln_lin(blk.ln1, lin_mod.weight, lin_mod.bias)
        x, y = lin(c, r, W, b)
        f = lambda z: z.reshape(B, T, H, dh).permute(0, 2, 1, 3)
        return f(x), f(y)
    qc, qr = heads(blk.q); kc, kr = heads(blk.k); vc, vr = heads(blk.v)
    sc = qc @ kc.transpose(-1, -2) / math.sqrt(dh)
    sr = (qc.abs() @ kr.transpose(-1, -2) + qr @ kc.abs().transpose(-1, -2) + qr @ kr.transpose(-1, -2)) / math.sqrt(dh)
    w_lo, w_hi = softmax_bounds(sc - sr, sc + sr, valid)
    z_hi = knapsack_avg(w_lo, w_hi, vc + vr, True)
    z_lo = knapsack_avg(w_lo, w_hi, vc - vr, False)
    zc = ((z_hi + z_lo) / 2).permute(0, 2, 1, 3).reshape(B, T, d)
    zr = ((z_hi - z_lo) / 2).clamp_min(0).permute(0, 2, 1, 3).reshape(B, T, d)
    oc, orr = lin(zc, zr, blk.o.weight, blk.o.bias)
    c, r = c + oc, r + orr
    W1, b1 = ln_lin(blk.ln2, blk.ff1.weight, blk.ff1.bias)
    hc, hr = lin(c, r, W1, b1)
    lo, hi = (hc - hr).clamp_min(0), (hc + hr).clamp_min(0)
    gc, gr = lin((lo + hi) / 2, (hi - lo) / 2, blk.ff2.weight, blk.ff2.bias)
    return c + gc, r + gr

def embed_box(lm, sets, pos, seg):
    """sets: per position a LongTensor of allowed token ids. Returns (c, r) of shape (1, T, d)."""
    cs, rs = [], []
    for s, p, g in zip(sets, pos, seg):
        E = lm.tok.weight[s]
        lo, hi = E.min(0).values, E.max(0).values
        base = lm.pos.weight[int(p)] + lm.seg.weight[int(g)]
        cs.append((lo + hi) / 2 + base); rs.append((hi - lo) / 2)
    return torch.stack(cs)[None], torch.stack(rs)[None]

def margins_box(lm, c, r):
    """Boxes (B, T, d) -> mlb (B, V, V) with mlb[b, j, t] <= z_j - z_t, and logit bounds (B, V),
    for the LAST position."""
    B, T, d = c.shape; H = lm.cfg["heads"]
    valid = torch.ones(T, T).tril().bool()
    for blk in lm.blocks:
        c, r = block_box(blk, c, r, valid, H)
    c, r = c[:, -1], r[:, -1]
    G, gb = ln_lin(lm.lnf, lm.head.weight, lm.head.bias)
    zc, zr = lin(c, r, G, gb)
    Dm = G[:, None, :] - G[None, :, :]                                       # (V, V, d)
    mlb = torch.einsum("jtd,bd->bjt", Dm, c) - torch.einsum("jtd,bd->bjt", Dm.abs(), r) + (gb[:, None] - gb[None, :])
    return mlb, zc - zr, zc + zr
