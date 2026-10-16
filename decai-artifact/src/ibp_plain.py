import math, torch
import tibp as T

def _lin_ln(c, r, ln, W, b):
    M, bl = T.ln_mat(ln, W.shape[1])
    c, r = T.lin(c, r, M, bl)                 # LN-NoVar as a separate layer
    return T.lin(c, r, W, b)

def interval_avg(w_lo, w_hi, v_lo, v_hi):
    """Interval product bound of sum_i w_i v_i with 0 <= w_lo <= w <= w_hi and v in [v_lo, v_hi]."""
    e = lambda w, v: torch.einsum("...qk,...kd->...qd", w, v)
    hi = e(w_hi, v_hi.clamp_min(0)) + e(w_lo, v_hi.clamp_max(0))
    lo = e(w_lo, v_lo.clamp_min(0)) + e(w_hi, v_lo.clamp_max(0))
    return lo, hi

def block_box_plain(blk, c, r, valid, H):
    B, Tn, d = c.shape; dh = d // H
    def heads(lin_mod):
        x, y = _lin_ln(c, r, blk.ln1, lin_mod.weight, lin_mod.bias)
        f = lambda z: z.reshape(B, Tn, H, dh).permute(0, 2, 1, 3)
        return f(x), f(y)
    qc, qr = heads(blk.q); kc, kr = heads(blk.k); vc, vr = heads(blk.v)
    sc = qc @ kc.transpose(-1, -2) / math.sqrt(dh)
    sr = (qc.abs() @ kr.transpose(-1, -2) + qr @ kc.abs().transpose(-1, -2) + qr @ kr.transpose(-1, -2)) / math.sqrt(dh)
    w_lo, w_hi = T.softmax_bounds(sc - sr, sc + sr, valid)
    z_lo, z_hi = interval_avg(w_lo, w_hi, vc - vr, vc + vr)
    zc = ((z_hi + z_lo) / 2).permute(0, 2, 1, 3).reshape(B, Tn, d)
    zr = ((z_hi - z_lo) / 2).clamp_min(0).permute(0, 2, 1, 3).reshape(B, Tn, d)
    oc, orr = T.lin(zc, zr, blk.o.weight, blk.o.bias)
    c, r = c + oc, r + orr
    hc, hr = _lin_ln(c, r, blk.ln2, blk.ff1.weight, blk.ff1.bias)
    lo, hi = (hc - hr).clamp_min(0), (hc + hr).clamp_min(0)
    gc, gr = T.lin((lo + hi) / 2, (hi - lo) / 2, blk.ff2.weight, blk.ff2.bias)
    return c + gc, r + gr

_ORIG = T.block_box
def install(): T.block_box = block_box_plain
def uninstall(): T.block_box = _ORIG
