import hashlib
import json
import os
import platform
import time

import numpy as np
import onnx
import onnxruntime as ort
import psutil
import torch
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic

from data import prepare_dataset
from run_calibration import TRAIN_CONFIGS, score_model, set_seed, train_model
from run_main import load_target

CFG_PATH = os.path.join(os.path.dirname(__file__), "config_main.yaml")
OUT = r"F:\AutoSci_CrossDomain_Handoff\results\edge-deploy"
os.makedirs(OUT, exist_ok=True)
ALPHA = 0.1
GAMMA = 0.05


class ExportWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        import math
        self.m = m
        self.np_ = m.n_patches
        nfreq = m.patch_len // 2 + 1
        fr = torch.arange(nfreq)
        nn = torch.arange(m.patch_len)
        ang = 2 * math.pi * fr[:, None] * nn[None, :] / m.patch_len
        self.register_buffer("Wc", torch.cos(ang))
        self.register_buffer("Ws", torch.sin(ang))
        edges = torch.linspace(0, nfreq, m.n_bands + 1).long()
        M = torch.zeros(m.n_bands, nfreq)
        for b in range(m.n_bands):
            lo = int(edges[b])
            hi = max(int(edges[b + 1]), lo + 1)
            M[b, lo:hi] = 1.0 / (hi - lo)
        self.register_buffer("M", M)

    def forward(self, x):
        m = self.m
        b, c, l = x.shape
        patches = torch.stack([x[:, :, i * m.patch_stride:i * m.patch_stride + m.patch_len] for i in range(self.np_)], dim=2)
        pc = patches @ self.Wc.t()
        ps = patches @ self.Ws.t()
        power = pc * pc + ps * ps
        band_e = power @ self.M.t()
        band_ids = band_e.argmax(dim=-1)
        z = m.patch_emb(patches)
        z = z + m.band_emb(band_ids)
        z = z + m.pos_emb[:, : self.np_, :].unsqueeze(1)
        chan = torch.arange(c, device=x.device)
        z = z + m.chan_emb(chan).view(1, c, 1, -1)
        z = z.reshape(b, c * self.np_, -1)
        z = m.encoder(z)
        out = m.head(z).reshape(b * c, self.np_, m.patch_len).transpose(1, 2)
        rec = torch.zeros(b * c, m.window, device=x.device, dtype=out.dtype)
        cnt = torch.zeros(b * c, m.window, device=x.device, dtype=out.dtype)
        for i in range(self.np_):
            a = i * m.patch_stride
            rec[:, a:a + m.patch_len] = rec[:, a:a + m.patch_len] + out[:, :, i]
            cnt[:, a:a + m.patch_len] = cnt[:, a:a + m.patch_len] + 1.0
        rec = (rec / cnt.clamp_min(1e-6)).reshape(b, c, m.window)
        return ((rec - x) ** 2).mean(dim=(1, 2))


def main():
    rep = {"host": {"cpu": platform.processor(), "python": platform.python_version(),
                    "torch": torch.__version__, "ort": ort.__version__}}
    with open(CFG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fold = {"name": "F4", "target": ["cwru", "paderborn"],
            "sources": ["phm16", "care", "pv", "mfpt", "xjtu"]}
    needed = set(fold["target"]) | set(fold["sources"])
    groups = {}
    for name in sorted(needed):
        off = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100000
        groups[name] = prepare_dataset(name, cfg, np.random.default_rng(off))
        print(f"[data] {name}: {tuple(groups[name]['windows'].shape)}", flush=True)

    tgt = load_target(fold["target"], groups)
    src_train, src_cal_list = [], []
    for s in fold["sources"]:
        g = groups[s]
        nm = np.where(g["labels"] < 0.5)[0]
        wnorm = g["windows"][torch.from_numpy(nm)]
        nc = max(int(wnorm.shape[0] * cfg["conformal"]["cal_ratio"]), 10)
        src_cal_list.append(wnorm[:nc])
        src_train.append({"name": s, "windows": wnorm[nc:]})
    src_cal = torch.cat(src_cal_list)
    Nt = tgt["windows"].shape[0]
    normal_idx = np.where(tgt["labels"] < 0.5)[0]
    cal_n = max(int(len(normal_idx) * cfg["conformal"]["cal_ratio"]), 10)
    eval_mask = np.ones(Nt, dtype=bool)
    eval_mask[normal_idx[:cal_n]] = False
    xe = tgt["windows"][eval_mask]
    le = tgt["labels"][eval_mask].astype(np.float32)
    print("[data] F4 eval:", tuple(xe.shape), flush=True)

    cfg["seed"] = 0
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = np.random.default_rng(0 + 17)
    pool_idx = np.sort(gen.choice(Nt, size=min(8000, Nt), replace=False))
    tgt_pool = tgt["windows"][torch.from_numpy(pool_idx)]
    cfg["_gating_weights"] = [1.0] * len(src_train)
    t0 = time.time()
    model, _ = train_model(TRAIN_CONFIGS["v2_plain"], cfg, src_train, tgt_pool, device)
    t_train = time.time() - t0
    print(f"[train] {t_train:.0f}s", flush=True)

    src_cal_scores = score_model(model, src_cal, cfg, device)
    scores_ref = score_model(model, xe, cfg, device)
    q0 = float(np.quantile(src_cal_scores, 1 - ALPHA))
    rep["train_s"] = t_train
    rep["n_params"] = int(sum(p.numel() for p in model.parameters()))

    # ---- ONNX export (CPU, fp32) ----
    model_cpu = model.to("cpu").eval()
    wrap = ExportWrap(model_cpu).eval()
    x_cpu = xe[:64].float()
    with torch.no_grad():
        ref64 = wrap(x_cpu).numpy()
    fp32_path = os.path.join(OUT, "encoder_fp32.onnx")
    torch.onnx.export(wrap, torch.zeros(1, 2, 256), fp32_path, input_names=["x"], output_names=["score"],
                      opset_version=17, dynamic_axes={"x": {0: "b"}, "score": {0: "b"}})
    int8_path = os.path.join(OUT, "encoder_int8.onnx")
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8)
    rep["model_mb_fp32"] = os.path.getsize(fp32_path) / 1e6
    rep["model_mb_int8"] = os.path.getsize(int8_path) / 1e6

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 1e6
    sess32 = ort.InferenceSession(fp32_path, so, providers=["CPUExecutionProvider"])
    sess8 = ort.InferenceSession(int8_path, so, providers=["CPUExecutionProvider"])
    rss1 = proc.memory_info().rss / 1e6
    rep["ort_session_delta_mb"] = rss1 - rss0
    o32 = sess32.run(None, {"x": x_cpu.numpy()})[0]
    rep["export_max_abs_diff"] = float(np.abs(o32 - ref64).max())

    # ---- latency benchmarks (single thread) ----
    def bench_b1(sess, x, reps):
        for i in range(10):
            sess.run(None, {"x": x[i:i + 1]})
        ts = []
        for i in range(reps):
            xb = x[(10 + i) % (len(x) - 1):(10 + i) % (len(x) - 1) + 1]
            t = time.perf_counter()
            sess.run(None, {"x": xb})
            ts.append(time.perf_counter() - t)
        return float(np.median(ts)) * 1e6, float(np.percentile(ts, 95)) * 1e6

    def bench_bN(sess, x, bs, reps):
        xb0 = x[:bs]
        for _ in range(3):
            sess.run(None, {"x": xb0})
        ts = []
        for i in range(reps):
            st = (i * bs) % (len(x) - bs)
            xb = x[st:st + bs]
            t = time.perf_counter()
            sess.run(None, {"x": xb})
            ts.append(time.perf_counter() - t)
        med = float(np.median(ts)) * 1e6
        return med, med / bs

    xb1 = xe.numpy()
    rep["ort_fp32_b1_us"], rep["ort_fp32_b1_p95"] = bench_b1(sess32, xb1, 200)
    rep["ort_int8_b1_us"], rep["ort_int8_b1_p95"] = bench_b1(sess8, xb1, 200)
    rep["ort_fp32_b64_ms"], rep["ort_fp32_b64_us_per_win"] = bench_bN(sess32, xb1, 64, 60)
    rep["ort_int8_b64_ms"], rep["ort_int8_b64_us_per_win"] = bench_bN(sess8, xb1, 64, 60)
    torch.set_num_threads(1)
    model_t = model.cpu().eval()
    with torch.no_grad():
        for _ in range(5):
            wrap(x_cpu[:1])
        ts = []
        for i in range(100):
            j = i % len(x_cpu)
            xb = x_cpu[j:j + 1]
            t = time.perf_counter()
            wrap(xb)
            ts.append(time.perf_counter() - t)
    rep["torch_cpu_b1_us"] = float(np.median(ts)) * 1e6

    # ---- INT8 effect on scores and calibration ----
    def run_scores(sess, x, bs=256):
        out = []
        for i in range(0, len(x), bs):
            out.append(sess.run(None, {"x": x[i:i + bs]})[0])
        return np.concatenate(out)

    s32 = run_scores(sess32, xb1)
    s8 = run_scores(sess8, xb1)
    from scipy.stats import spearmanr
    rep["spearman_int8_vs_fp32"] = float(spearmanr(s32, s8).statistic)
    rep["score_max_abs_diff_int8"] = float(np.abs(s32 - s8).max())
    rep["score_mean_abs_diff_int8"] = float(np.abs(s32 - s8).mean())

    def aci_metrics(scores):
        q = q0
        preds = np.empty(len(scores), bool)
        for i, s in enumerate(scores):
            preds[i] = s > q
            q = q + GAMMA * max(abs(q), 1e-12) * ((1.0 if preds[i] else 0.0) - ALPHA)
        nrm = le < 0.5
        return {"gap": float(abs(preds[nrm].mean() - ALPHA)), "alarm": float(preds.mean()),
                "det": float(preds[~nrm].mean())}

    rep["metrics_torch_fp32"] = aci_metrics(scores_ref)
    rep["metrics_onnx_fp32"] = aci_metrics(s32)
    rep["metrics_onnx_int8"] = aci_metrics(s8)

    # ---- streaming demo with checkpoint/resume (int8, batch 1) ----
    N = min(4000, len(s32))
    q = q0
    alarms = np.empty(N, bool)
    t0 = time.time()
    ckpt = None
    for i in range(N):
        s = float(sess8.run(None, {"x": xb1[i:i + 1]})[0][0])
        alarms[i] = s > q
        q = q + GAMMA * max(abs(q), 1e-12) * ((1.0 if alarms[i] else 0.0) - ALPHA)
        if i == 1999:
            ckpt = {"t": i + 1, "q": q}
    demo_s = time.time() - t0
    rep["stream_demo_us_per_window"] = demo_s / N * 1e6
    rep["stream_demo_n"] = N
    with open(os.path.join(OUT, "demo_checkpoint.json"), "w") as f:
        json.dump(ckpt, f)
    # resume from checkpoint and compare next 200 alarms
    q2 = ckpt["q"]
    ok = True
    for j in range(2000, 2200):
        s = float(sess8.run(None, {"x": xb1[j:j + 1]})[0][0])
        a2 = s > q2
        q2 = q2 + GAMMA * max(abs(q2), 1e-12) * ((1.0 if a2 else 0.0) - ALPHA)
        if bool(a2) != bool(alarms[j]):
            ok = False
    rep["resume_identical"] = bool(ok)
    rep["stream_demo_alarm_rate"] = float(alarms.mean())

    json.dump(rep, open(os.path.join(OUT, "edge_deploy.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print("saved:", os.path.join(OUT, "edge_deploy.json"))


if __name__ == "__main__":
    main()
