# Cross-Domain Industrial Time-Series Anomaly Detection — Conformal Calibration Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23321391.svg)](https://doi.org/10.5281/zenodo.23321391)

Companion code and manuscripts for **"When Static Thresholds Fail: Online Conformal Calibration for Cross-Domain Industrial Anomaly Detection"** (manuscript under review, IEEE Transactions on Instrumentation and Measurement).

**This snapshot (v1.1.0)** — revision-round update: adds the title change, the NA-ACI proposition, target-side calibration baselines, fault-type decomposition, calibration-contamination and label-noise robustness, sensitivity/tracking/warm-up analyses, the edge-deployment study (ONNX/INT8/streaming), and a 135-paper companion survey overview.

## Headline findings

- **Static thresholds fail on 6/6 cross-domain folds in both directions** (normal-window coverage 0.18–0.999 at nominal 0.90).
- **ACI tracks the stream-wide alarm rate** (largest deviation 0.033 across 24 fold–level cells) and restores coverage on 5/6 folds; its failure (F3, synthetic) is the **alarm-budget defect**.
- **NA-ACI** (normal-anchored, exclusion parameter κ) removes the defect: F4 detection 0.12 → 0.90 at essentially unchanged FPR (0.090→0.094); mean range-F1 0.265 → 0.389; longest missed-anomaly streaks 549 → 70 windows (F4) and 343 → 42 (F6).
- **No target-side rule replaces feedback**: best (DSPOT) gap 0.094 on F4 vs ACI's 0.010; weighted conformal fixes F2 (0.002) but not F6; a P² quantile tracker fails everywhere (0.10–0.88).
- **Robustness**: contaminated initialization self-heals under ACI (gap 0.018 on F4); 20% symmetric label noise leaves the ACI gap at 0.018; NA-ACI recovers 7/8 CWRU fault classes to ≥0.996 detection.
- **Edge deployment** (F4 encoder, single-thread CPU): 78K parameters; ONNX export numerically exact (3e-9); INT8 dynamic quantization 0.46 → 0.30 MB with no measurable calibration change (Spearman 0.999); 0.55 ms/window (batch-1) / 91 µs/window (batch-64); checkpointed streaming demo 0.61 ms/window with bit-identical resume.

## Repository layout

```
paper/    LaTeX source + compiled main.pdf (13 pp) + submission docs
survey/   Companion survey (LaTeX + compiled main.pdf) + 135-paper corpus JSON
code/     Experiment drivers (main, calibration, traces, revision analyses,
          run_revision_batch{2,3,4}.py, run_edge_deploy.py)
results/  Summary JSON/MD outputs per experiment suite (scc-*, revision-batch*, edge-deploy)
reports/  Revision-round execution reports
```

## Datasets (all public)

NASA C-MAPSS (turbofan), NASA IMS and XJTU-SY and Paderborn (bearings), MFPT, CWRU, CARE wind SCADA, PVDAQ (photovoltaic). See `paper/LITERATURE_DOWNLOAD_LIST.md` and dataset provider pages for licences; no raw data is redistributed here.

## Reproduce

```bash
# recommend: Python 3.10+, torch (GPU), numpy, pandas, scikit-learn, onnx, onnxruntime
python code/run_main.py            # 6-fold x 3-seed main calibration benchmark
python code/run_traces.py          # per-window score/threshold traces
python code/run_revision_batch2.py # target-side baselines, contamination, label noise, fault types
python code/run_revision_batch3.py # kappa/gamma sweeps, P2 tracker, oracle decomposition, bootstrap
python code/run_edge_deploy.py     # ONNX export, INT8 quantization, CPU benchmark, streaming demo
```

## Citation

```bibtex
@misc{hu2026static,
  title  = {When Static Thresholds Fail: Online Conformal Calibration for Cross-Domain Industrial Anomaly Detection},
  author = {Hu, Lingyan and Zhu, Jibiao},
  year   = {2026},
  note   = {Manuscript under review; code and protocols archived at Zenodo},
  doi    = {10.5281/zenodo.23321391}
}
```

## Versions

- `v1.0.0` — submission snapshot: DOI `10.5281/zenodo.23321392`
- `v1.1.0` — revision-round update (this release); the concept DOI `10.5281/zenodo.23321391` always resolves to the latest version.

## License

Code: MIT (see `LICENSE`). Manuscripts © the authors.
