import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTSTRecon(nn.Module):
    def __init__(self, c_in, window, patch_len=64, patch_stride=32, d_model=64,
                 n_heads=4, e_layers=2, d_ff=128, dropout=0.1, n_bands=8):
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.window = window
        self.n_patches = (window - patch_len) // patch_stride + 1
        self.n_bands = n_bands
        self.patch_emb = nn.Linear(patch_len, d_model)
        self.band_emb = nn.Embedding(n_bands, d_model)
        self.chan_emb = nn.Embedding(c_in, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.head = nn.Linear(d_model, patch_len)
        self.rel_head = nn.Linear(d_model, 32)

    def _patch_band_ids(self, patches):
        power = torch.fft.rfft(patches.float(), dim=-1).abs() ** 2
        nfreq = power.shape[-1]
        edges = torch.linspace(0, nfreq, self.n_bands + 1).long()
        bands = []
        for b in range(self.n_bands):
            lo = int(edges[b])
            hi = max(int(edges[b + 1]), lo + 1)
            bands.append(power[..., lo:hi].mean(dim=-1))
        return torch.stack(bands, dim=-1).argmax(dim=-1)

    def forward(self, x):
        b, c, l = x.shape
        patches = x.unfold(2, self.patch_len, self.patch_stride)
        np_ = patches.shape[2]
        band_ids = self._patch_band_ids(patches)
        z = self.patch_emb(patches)
        z = z + self.band_emb(band_ids)
        z = z + self.pos_emb[:, :np_, :].unsqueeze(1)
        chan = torch.arange(c, device=x.device)
        z = z + self.chan_emb(chan).view(1, c, 1, -1)
        z = z.reshape(b, c * np_, -1)
        z = self.encoder(z)
        zz = z.reshape(b, c, np_, -1)
        rel_feat = self.rel_head(zz.mean(dim=2))
        oh = F.one_hot(band_ids, self.n_bands).float()
        oh = oh / oh.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        band_agg = torch.einsum("bcnk,bcnd->bkd", oh, zz.float())
        out = self.head(z).reshape(b * c, np_, self.patch_len).transpose(1, 2)
        ones = torch.ones_like(out)
        rec = F.fold(out, output_size=(1, self.window), kernel_size=(1, self.patch_len), stride=(1, self.patch_stride))
        cnt = F.fold(ones, output_size=(1, self.window), kernel_size=(1, self.patch_len), stride=(1, self.patch_stride))
        rec = (rec / cnt.clamp_min(1e-6)).reshape(b, c, self.window)
        return rec, rel_feat, band_agg
