import numpy as np, torch
from auto_LiRPA.perturbations import PerturbationLpNorm

class PerturbationHull(PerturbationLpNorm):
    def __init__(self, vertices):
        """vertices: list (one per position) of tensors (m_i, d) of embedding vectors."""
        lo = torch.stack([v.min(0).values for v in vertices])[None]
        hi = torch.stack([v.max(0).values for v in vertices])[None]
        super().__init__(norm=np.inf, x_L=lo, x_U=hi)
        self.T, self.d = len(vertices), vertices[0].shape[1]
        sizes = [v.shape[0] for v in vertices]
        self.fixed = [i for i, m in enumerate(sizes) if m == 1]
        self.free = [i for i, m in enumerate(sizes) if m > 1]
        self.Vfixed = torch.stack([vertices[i][0] for i in self.fixed]) if self.fixed else None   # (Tf, d)
        if self.free:
            m = max(sizes[i] for i in self.free)
            pad = [torch.cat([vertices[i], vertices[i][:1].expand(m - sizes[i], -1)]) for i in self.free]
            self.Vfree = torch.stack(pad)                                                     # (Tv, m, d)

    def concretize(self, x, A, sign=-1, constraints=None, aux=None):
        if not isinstance(A, torch.Tensor):
            return super().concretize(x, A, sign, constraints, aux)
        B, S = A.shape[:2]
        A4 = A.reshape(B, S, self.T, self.d)
        out = torch.zeros(B, S, dtype=A.dtype)
        if self.fixed:
            out = out + torch.einsum("bstd,td->bs", A4[:, :, self.fixed], self.Vfixed)
        if self.free:
            sc = torch.einsum("bstd,tmd->bstm", A4[:, :, self.free], self.Vfree)
            out = out + (sc.min(-1).values if sign == -1 else sc.max(-1).values).sum(-1)
        return out
