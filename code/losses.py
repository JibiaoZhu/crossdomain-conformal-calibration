import torch
import torch.nn.functional as F

from data import standardize_global


def band_energies(x, n_bands):
    spec = torch.fft.rfft(x.float(), dim=-1)
    power = spec.abs() ** 2
    nfreq = power.shape[-1]
    edges = torch.linspace(0, nfreq, n_bands + 1).long()
    feats = []
    for b in range(n_bands):
        lo = int(edges[b])
        hi = max(int(edges[b + 1]), lo + 1)
        feats.append(power[..., lo:hi].mean(dim=-1))
    return torch.log1p(torch.stack(feats, dim=-1)).flatten(1)


def rec_loss(rec, x):
    return torch.mean((rec - x) ** 2)


def moment_match_loss(a, b):
    a = a.float()
    b = b.float()
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = torch.cov(a.T), torch.cov(b.T)
    return torch.mean((ma - mb) ** 2) + torch.mean((ca - cb) ** 2)


def relation_consistency_loss(fsrc, ftgt):
    ms = fsrc.float().mean(0)
    mt = ftgt.float().mean(0)
    rs = F.normalize(ms, dim=-1)
    rt = F.normalize(mt, dim=-1)
    return torch.mean((rs @ rs.T - rt @ rt.T) ** 2)


def band_saliency_loss(x, rec, n_bands):
    ex = band_energies(x, n_bands)
    er = band_energies(rec, n_bands)
    exd = ex - ex.mean(0, keepdim=True)
    erd = er - er.mean(0, keepdim=True)
    return torch.mean((exd - erd) ** 2)
