import numpy as np
import torch
import torch.nn as nn


class LowRankAdapter(nn.Module):
    def __init__(self, window, rank=4):
        super().__init__()
        self.V = nn.Parameter(torch.randn(rank, window) * 0.01)
        self.U = nn.Parameter(torch.zeros(window, rank))

    def forward(self, x):
        return x + ((x @ self.V.t()) @ self.U.t())


def adapt(model, adapter, windows, cfg, device):
    t = cfg["tta"]
    ws = windows[: t["windows"]]
    opt = torch.optim.Adam(adapter.parameters(), lr=t["lr"])
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    bs = cfg["train"]["batch_size"]
    gen = np.random.default_rng(cfg["seed"] + 7)
    model.eval()
    adapter.train()
    for _ in range(1, t["steps"] + 1):
        idx = gen.choice(ws.shape[0], size=min(bs, ws.shape[0]), replace=False)
        xb = ws[torch.from_numpy(idx)].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            rec, _, _ = model(adapter(xb))
            loss = torch.mean((rec - xb) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    adapter.eval()
    return adapter
