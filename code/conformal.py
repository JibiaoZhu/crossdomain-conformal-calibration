import numpy as np
import torch


def band_ref_stats(ref_e):
    ref_e = ref_e.float()
    mu = ref_e.mean(0)
    cov = torch.cov(ref_e.T) + 1e-3 * torch.eye(ref_e.shape[1], device=ref_e.device)
    inv = torch.linalg.inv(cov)
    return mu, inv


def dsrc_mahalanobis(mu, inv, e):
    diff = e.float() - mu
    d2 = (diff @ inv * diff).sum(dim=1).clamp_min(0.0)
    return torch.sqrt(d2).numpy().astype(np.float64)


def relative_divergence(dsrc, early_ratio):
    n = len(dsrc)
    early = dsrc[: max(int(n * early_ratio), 10)]
    base = float(np.median(early))
    spread = float(np.percentile(early, 75) - np.percentile(early, 25))
    spread = max(spread, 1e-3)
    return (dsrc - base) / spread


class ACI:
    def __init__(self, q0, alpha, relative_gamma, shift_lambda=0.0):
        self.q = float(q0)
        self.alpha = float(alpha)
        self.relative_gamma = float(relative_gamma)
        self.shift_lambda = float(shift_lambda)

    def threshold(self, d_norm=0.0):
        return self.q * (1.0 + self.shift_lambda * float(d_norm))

    def update(self, score, q_t):
        alarm = 1.0 if float(score) > float(q_t) else 0.0
        g = self.relative_gamma * max(abs(self.q), 1e-6)
        self.q += g * (alarm - self.alpha)
