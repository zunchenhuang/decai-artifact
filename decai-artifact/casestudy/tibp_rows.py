import math, torch
from tibp import lin, ln_lin, softmax_bounds, knapsack_avg

def _hull(xc, xr, pos, T):
    H, N, D = xc.shape
    idx = pos[None, :, None].expand(H, N, D)
    lo = torch.full((H, T, D), float("inf")).scatter_reduce(1, idx, xc - xr, "amin")
    hi = torch.full((H, T, D), float("-inf")).scatter_reduce(1, idx, xc + xr, "amax")
    return (lo + hi) / 2, ((hi - lo) / 2).clamp_min(0)

def block_rows(blk, c, r, valid, pos, T, H):
    N, d = c.shape; dh = d // H; ar = torch.arange(N)
    def heads(m):
        W, b = ln_lin(blk.ln1, m.weight, m.bias); x, y = lin(c, r, W, b)
        return x.reshape(N, H, dh).transpose(0, 1), y.reshape(N, H, dh).transpose(0, 1)
    qc, qr = heads(blk.q); kc, kr = heads(blk.k); vc, vr = heads(blk.v)
    Kc, Kr = _hull(kc, kr, pos, T); Vc, Vr = _hull(vc, vr, pos, T)
    sc = qc @ Kc.transpose(-1, -2) / math.sqrt(dh)
    sr = (qc.abs() @ Kr.transpose(-1, -2) + qr @ Kc.abs().transpose(-1, -2) + qr @ Kr.transpose(-1, -2)) / math.sqrt(dh)
    sc = sc.clone(); sr = sr.clone()
    sc[:, ar, pos] = (qc * kc).sum(-1) / math.sqrt(dh)
    sr[:, ar, pos] = (qc.abs() * kr + qr * kc.abs() + qr * kr).sum(-1) / math.sqrt(dh)
    w_lo, w_hi = softmax_bounds(sc - sr, sc + sr, valid)                       # (H, N, T)
    Vcr = Vc[:, None].expand(H, N, T, dh).clone(); Vrr = Vr[:, None].expand(H, N, T, dh).clone()
    Vcr[:, ar, pos] = vc; Vrr[:, ar, pos] = vr
    wl, wh = w_lo[:, :, None, :], w_hi[:, :, None, :]
    z_hi = knapsack_avg(wl, wh, Vcr + Vrr, True)[:, :, 0]; z_lo = knapsack_avg(wl, wh, Vcr - Vrr, False)[:, :, 0]
    exact = ((w_hi - w_lo).abs().amax(-1) == 0) & ((Vrr.amax(-1) * valid).amax(-1) == 0)
    if exact.any():
        ze = torch.einsum("hnk,hnkd->hnd", w_lo, Vcr)
        z_hi = torch.where(exact[..., None], ze, z_hi); z_lo = torch.where(exact[..., None], ze, z_lo)
    zc = ((z_hi + z_lo) / 2).transpose(0, 1).reshape(N, d); zr = ((z_hi - z_lo) / 2).clamp_min(0).transpose(0, 1).reshape(N, d)
    oc, orr = lin(zc, zr, blk.o.weight, blk.o.bias)
    c, r = c + oc, r + orr
    W1, b1 = ln_lin(blk.ln2, blk.ff1.weight, blk.ff1.bias)
    hc, hr = lin(c, r, W1, b1)
    lo, hi = (hc - hr).clamp_min(0), (hc + hr).clamp_min(0)
    gc, gr = lin((lo + hi) / 2, (hi - lo) / 2, blk.ff2.weight, blk.ff2.bias)
    return c + gc, r + gr

def final_rows(lm, sets, pos, seg, eps=1.0, rows_max=10**9, hull_at=()):
    """sets: per position a LongTensor of token ids.  Positions with more than rows_max tokens are one hull row.
    eps < 1 scales the hull radii (training warm-up only).  Returns (c, r) of the last position's rows."""
    T = len(sets); W = lm.tok.weight; cs, rs, ps = [], [], []
    for i, (s, p, g) in enumerate(zip(sets, pos, seg)):
        base = lm.pos.weight[int(p)] + lm.seg.weight[int(g)]
        if len(s) <= rows_max and i not in hull_at:
            cs.append(W[s] + base); rs.append(torch.zeros(len(s), W.shape[1])); ps.append(torch.full((len(s),), i))
        else:
            E = W[s]; lo, hi = E.min(0).values, E.max(0).values
            cs.append(((lo + hi) / 2 + base)[None]); rs.append((eps * (hi - lo) / 2)[None]); ps.append(torch.tensor([i]))
    c, r, pos_r = torch.cat(cs), torch.cat(rs), torch.cat(ps)
    valid = torch.arange(T)[None, :] <= pos_r[:, None]
    H = lm.cfg["heads"]
    for blk in lm.blocks:
        c, r = block_rows(blk, c, r, valid, pos_r, T, H)
    last = pos_r == T - 1
    return c[last], r[last]

def margins_rows(lm, sets, pos, seg, **kw):
    """Per last-position row: mlb (n, V, V) with mlb[i, j, t] <= z_j - z_t."""
    c, r = final_rows(lm, sets, pos, seg, **kw)
    G, gb = ln_lin(lm.lnf, lm.head.weight, lm.head.bias)
    Dm = G[:, None, :] - G[None, :, :]
    return torch.einsum("jtd,nd->njt", Dm, c) - torch.einsum("jtd,nd->njt", Dm.abs(), r) + (gb[:, None] - gb[None, :])
