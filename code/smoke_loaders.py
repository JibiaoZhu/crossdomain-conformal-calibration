import sys
import time

import numpy as np
import yaml

sys.path.insert(0, r"G:\autosci-opencode\experiments\code\ssi-ad-ablation-spectral-relation-band")
from data import prepare_dataset

cfg = yaml.safe_load(open(r"G:\autosci-opencode\experiments\code\ssi-ad-ablation-spectral-relation-band\config_main.yaml", encoding="utf-8"))

for name in ["care", "paderborn"]:
    t0 = time.time()
    rng = np.random.default_rng(7)
    g = prepare_dataset(name, cfg, rng)
    print(f"[loader-smoke] {name}: windows={tuple(g['windows'].shape)} anomalies={int(g['labels'].sum())} ({time.time()-t0:.0f}s)", flush=True)
print("loader-smoke done", flush=True)
