import csv as _csv
import os

import numpy as np
import pandas as pd
import torch

CACHE_VERSION = 3


def _cache_path(cfg, name):
    d = cfg["paths"]["cache_dir"]
    os.makedirs(d, exist_ok=True)
    w = cfg["data"]["window"]
    s = cfg["data"]["stride"]
    return os.path.join(d, f"{name}_w{w}_s{s}_v{CACHE_VERSION}.npz")


def _sliding(x, window, stride):
    n = (x.shape[-1] - window) // stride + 1
    if n <= 0:
        return np.zeros((0, x.shape[0], window), dtype=np.float32)
    idx = np.arange(n)[:, None] * stride + np.arange(window)[None, :]
    return np.transpose(x[:, idx], (1, 0, 2)).astype(np.float32)


def _cap(windows, labels, cap, rng):
    if windows.shape[0] <= cap:
        return windows, labels
    idx = np.sort(rng.choice(windows.shape[0], size=cap, replace=False))
    return windows[idx], labels[idx]


def standardize_global(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    x = x - x.mean(dim=(0, 2), keepdim=True)
    x = x / (x.std(dim=(0, 2), keepdim=True) + 1e-6)
    return x


def load_xjtu(cfg, rng):
    root = os.path.join(cfg["paths"]["data_root"], "XJTU-SY", "XJTU-SY_Bearing_Datasets")
    w = cfg["data"]["window"]
    s = cfg["data"]["stride"]
    af = cfg["data"]["xjtu"]["anomaly_fraction"]
    ws, ls = [], []
    for b in cfg["data"]["xjtu"]["bearings"]:
        d = os.path.join(root, b.replace("/", os.sep))
        files = sorted((f for f in os.listdir(d) if f.endswith(".csv")), key=lambda x: int(x.split(".")[0]))
        parts = []
        for f in files:
            arr = np.loadtxt(os.path.join(d, f), delimiter=",", skiprows=1, dtype=np.float32).T
            parts.append(_sliding(arr, w, s))
        x = np.concatenate(parts, axis=0)
        cut = int(x.shape[0] * (1 - af))
        lab = np.zeros(x.shape[0], dtype=np.float32)
        lab[cut:] = 1.0
        ws.append(x)
        ls.append(lab)
    return np.concatenate(ws), np.concatenate(ls)


def load_mfpt(cfg, rng):
    p = os.path.join(cfg["paths"]["data_root"], "MFPT", "MFPT.csv")
    w = cfg["data"]["window"]
    s = cfg["data"]["stride"]
    cap = int(cfg["data"]["mfpt"]["windows_per_segment"])
    ws, ls = [], []
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        r = _csv.reader(f)
        next(r)
        for row in r:
            if len(row) < 8:
                continue
            try:
                sig = np.fromstring(row[5].strip().strip("[]"), sep=",", dtype=np.float32)
            except Exception:
                continue
            lab = 0.0 if row[6].strip() == "0" else 1.0
            ww = _sliding(sig[None, :], w, s)[:, 0, :]
            if ww.shape[0] == 0:
                continue
            if ww.shape[0] > cap:
                idx = np.sort(rng.choice(ww.shape[0], size=cap, replace=False))
                ww = ww[idx]
            ws.append(ww)
            ls.append(np.full(ww.shape[0], lab, dtype=np.float32))
    x1 = np.concatenate(ws)
    x = np.stack([x1, np.zeros_like(x1)], axis=1)
    return x, np.concatenate(ls)


def load_phm16(cfg, rng):
    base = os.path.join(cfg["paths"]["data_root"], "PHM16-CMP", "CMP-data")
    chans = cfg["data"]["phm16"]["channels"]
    usecols = ["WAFER_ID", "STAGE", "TIMESTAMP"] + chans
    parts = []
    for split in cfg["data"]["phm16"]["splits"]:
        d = os.path.join(base, split)
        for f in sorted(os.listdir(d)):
            if not f.endswith(".csv"):
                continue
            try:
                parts.append(pd.read_csv(os.path.join(d, f), usecols=usecols))
            except Exception:
                continue
    df = pd.concat(parts, ignore_index=True)
    labs = []
    for labname in ["training", "test"]:
        labs.append(pd.read_csv(os.path.join(base, f"CMP-{labname}-removalrate.csv"), encoding="utf-8-sig"))
    lab = pd.concat(labs, ignore_index=True)
    thr = lab.groupby("STAGE")["AVG_REMOVAL_RATE"].quantile(cfg["data"]["phm16"]["normal_quantile"])
    lab = lab.merge(thr.rename("thr"), left_on="STAGE", right_index=True, how="left")
    lab["anom"] = (lab["AVG_REMOVAL_RATE"] <= lab["thr"]).astype(float)
    merged = df.merge(lab[["WAFER_ID", "STAGE", "anom"]], on=["WAFER_ID", "STAGE"], how="inner")
    w = cfg["data"]["window"]
    s = cfg["data"]["stride"]
    ws, ls = [], []
    for (wid, stage), g in merged.groupby(["WAFER_ID", "STAGE"]):
        g = g.sort_values("TIMESTAMP")
        arr = g[chans].to_numpy(dtype=np.float32).T
        ww = _sliding(arr, w, s)
        if ww.shape[0] == 0:
            continue
        ws.append(ww)
        ls.append(np.full(ww.shape[0], float(g["anom"].iloc[0]), dtype=np.float32))
    return np.concatenate(ws), np.concatenate(ls)


def _pick_col(cols, pattern):
    for c in cols:
        if pattern in c:
            return c
    return None


def _first_col(cols, include, exclude=()):
    for c in cols:
        if all(t in c for t in include) and not any(t in c for t in exclude):
            return c
    return None


def load_pv(cfg, rng):
    root = os.path.join(cfg["paths"]["data_root"], "PVDAQ")
    w = cfg["data"]["window"]
    s = cfg["data"]["stride"]
    ws = []
    for sysname in cfg["data"]["pv"]["systems"]:
        d = os.path.join(root, sysname, "data")
        files = os.listdir(d)
        irr_cands = [f for f in files if "irradiance" in f]
        met_cands = [f for f in files if "meter_data" in f]
        irr_f = max(irr_cands, key=lambda f: os.path.getsize(os.path.join(d, f)))
        met_f = max(met_cands, key=lambda f: os.path.getsize(os.path.join(d, f)))
        irr = pd.read_csv(os.path.join(d, irr_f))
        met = pd.read_csv(os.path.join(d, met_f))
        c_irr = _first_col(irr.columns, ["irradiance", "ghi"])
        if c_irr is None:
            c_irr = _first_col(irr.columns, ["poa_irradiance"], ["temp_compensated"])
        c_met = _first_col(met.columns, ["meter_ac_output"])
        if c_met is None:
            c_met = _first_col(met.columns, ["meter_ac_power"], ["phase"])
        m = irr[["measured_on", c_irr]].merge(met[["measured_on", c_met]], on="measured_on", how="inner")
        arr = m[[c_irr, c_met]].to_numpy(dtype=np.float32).T
        ws.append(_sliding(arr, w, s))
    x = np.concatenate(ws, axis=0)
    return x, np.zeros(x.shape[0], dtype=np.float32)


def load_cwru(cfg, rng):
    import scipy.io as sio
    root = os.path.join(cfg["paths"]["data_root"], "CWRU")
    w = cfg["data"]["window"]
    s = cfg["data"]["stride"]

    def read_ids(ids, lab_value):
        parts, labs = [], []
        for fid in ids:
            mat = sio.loadmat(os.path.join(root, f"{fid}.mat"))
            keys = [k for k in mat if k.endswith("_time")]
            chans = []
            for ch in ("DE", "FE"):
                k = next((k for k in keys if k.endswith(f"_{ch}_time")), None)
                chans.append(mat[k].ravel().astype(np.float32))
            arr = np.stack(chans, axis=0)
            ww = _sliding(arr, w, s)
            parts.append(ww)
            labs.append(np.full(ww.shape[0], lab_value, dtype=np.float32))
        return parts, labs

    parts, labs = read_ids(cfg["data"]["cwru"]["normal_ids"], 0.0)
    fparts, flabs = read_ids(cfg["data"]["cwru"]["fault_ids"], 1.0)
    return np.concatenate(parts + fparts), np.concatenate(labs + flabs)


def load_care(cfg, rng):
    import re as _re
    root = os.path.join(cfg["paths"]["data_root"], "CARE")
    p = os.path.join(root, "wind-turbine-scada-data-for-early-fault-detection.csv")
    w = cfg["data"]["window"]
    s = cfg["data"]["stride"]
    chans = cfg["data"]["care"]["channels"]
    cases = cfg["data"]["care"]["cases"]
    keep = ["time_stamp", "asset_id", "status_type_id"] + chans
    try:
        import pyarrow.csv as pacsv
        table = pacsv.read_csv(p, read_options=pacsv.ReadOptions(), convert_options=pacsv.ConvertOptions(include_columns=keep))
        df = table.to_pandas()
    except Exception:
        chunks = []
        for ch in pd.read_csv(p, usecols=keep, chunksize=1_000_000, low_memory=False):
            chunks.append(ch[ch["asset_id"].isin(cases)])
        df = pd.concat(chunks, ignore_index=True)
    ws, ls = [], []
    for case in cases:
        g = df[df["asset_id"] == case].sort_values("time_stamp")
        if g.shape[0] < w:
            continue
        arr = g[chans].to_numpy(dtype=np.float32).T
        lab_rows = (g["status_type_id"].to_numpy(dtype=np.float32) > 0).astype(np.float32)
        n = (arr.shape[1] - w) // s + 1
        idx0 = np.arange(n)[:, None] * s
        idx = idx0 + np.arange(w)[None, :]
        ww = np.transpose(arr[:, idx], (1, 0, 2)).astype(np.float32)
        lab = lab_rows[idx].mean(axis=1)
        ws.append(ww)
        ls.append((lab > 0.5).astype(np.float32))
    if not ws:
        return np.zeros((0, 2, w), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.concatenate(ws), np.concatenate(ls)


def load_paderborn(cfg, rng):
    import glob
    import re as _re
    import scipy.io as sio
    root = os.path.join(cfg["paths"]["data_root"], "Paderborn")
    w = cfg["data"]["window"]
    s = cfg["data"]["stride"]
    per_meas = int(cfg["data"]["paderborn"]["windows_per_measurement"])
    n_meas = int(cfg["data"]["paderborn"]["measurements_per_bearing"])
    af = cfg["data"]["paderborn"]["anomaly_fraction"]
    ws, ls = [], []
    for bearing in cfg["data"]["paderborn"]["bearings"]:
        files = sorted(glob.glob(os.path.join(root, bearing, "vibrationData", "*.mat")),
                       key=lambda f: int(_re.search(r"M(\d+)", os.path.basename(f)).group(1)))
        if len(files) > n_meas:
            sel = np.sort(rng.choice(len(files), size=n_meas, replace=False))
            files = [files[i] for i in sel]
        cut = int(len(files) * (1 - af))
        for i, f in enumerate(files):
            try:
                mat = sio.loadmat(f, variable_names=["accHorizFrontal_C", "accHorizRear_A"])
            except Exception:
                continue
            a = mat["accHorizFrontal_C"].ravel().astype(np.float32)
            b = mat["accHorizRear_A"].ravel().astype(np.float32)
            arr = np.stack([a, b], axis=0)
            n = (arr.shape[1] - w) // s + 1
            if n <= 0:
                continue
            take = np.linspace(0, n - 1, min(per_meas, n)).astype(int)
            ww = np.stack([arr[:, t * s: t * s + w] for t in take], axis=0)
            lab = 1.0 if i >= cut else 0.0
            ws.append(ww)
            ls.append(np.full(ww.shape[0], lab, dtype=np.float32))
    return np.concatenate(ws), np.concatenate(ls)


LOADERS = {
    "xjtu": load_xjtu,
    "mfpt": load_mfpt,
    "phm16": load_phm16,
    "pv": load_pv,
    "cwru": load_cwru,
    "care": load_care,
    "paderborn": load_paderborn,
}


def prepare_dataset(name, cfg, rng, use_cache=True):
    path = _cache_path(cfg, name)
    if use_cache and os.path.exists(path):
        z = np.load(path)
        windows, labels = z["windows"], z["labels"]
    else:
        windows, labels = LOADERS[name](cfg, rng)
        np.savez(path, windows=windows, labels=labels)
    cap = cfg["data"]["max_windows_per_dataset"]
    windows, labels = _cap(windows, labels, cap, rng)
    x = standardize_global(windows)
    return {"name": name, "windows": x, "labels": labels.astype(np.float32)}
