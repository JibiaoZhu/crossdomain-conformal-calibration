# 边缘部署（免硬件部分）执行报告 — 2026-10-12

脚本：`experiments/code/ssi-ad-ablation-spectral-relation-band/run_edge_deploy.py`（可复跑）
数据：`F:\AutoSci_CrossDomain_Handoff\results\edge-deploy\`（edge_deploy.json、encoder_fp32.onnx、encoder_int8.onnx、demo_checkpoint.json）
论文：相关句子已并入 §Sensitivity（Operational filtering and cost），13 页 0/0/0。

## 关键结果（F4 折，seed 0；本机单线程 x86，Intel Model 167）

| 指标 | 数值 |
|---|---|
| 编码器参数量 | **78,432**（论文原写 1.3M → 已全文更正） |
| ONNX 导出等价性 | max abs diff **3.3×10⁻⁹**（FFT 频带选择器替换为等值 DFT 基矩阵） |
| 模型大小 | FP32 0.46 MB → INT8 0.30 MB |
| ORT 延迟 b1 | FP32 **554 µs** / INT8 568 µs（中位；p95 ~715 µs） |
| ORT 延迟 b64 | FP32 **91.4 µs/窗** / INT8 85.3 µs/窗 |
| PyTorch CPU b1 | 1,108 µs/窗 |
| 校准层更新 | 0.36 µs/更新（纯 Python） |
| INT8 校准影响 | Spearman 0.999；gap 0.00974→0.00947（Δ3e-4）；检出 0.1183→0.1179；告警率完全一致 |
| 流式 demo（批1、INT8、含 ACI+决策） | **612.6 µs/窗**，4000 窗；告警率 0.09 |
| 断点恢复 | 保存 t=2000 状态 → 恢复后 200 窗输出 **bit-identical** ✓ |

## 结论与边界
- 校准层本身可忽略；端到端成本由编码器推理主导。单线程桌面 CPU 上 <1ms/窗已可运行于产线级采样（如 1–2 Hz 折；kHz 折需 batch 或核并行）。
- INT8 动态量化在此小模型上收益有限（b64 约 7%，b1 无收益）——如实报告。
- 树莓派 4（Cortex-A72）预计约慢 5–10×（≈3–6 ms/窗，推断值）；**真实边缘能耗（mJ/决策）** 需方案 A/B 硬件后补测（硬件清单见桌面《操作说明》）。

## 论文修正
- `Value … Params $\sim$1.3\,M` → `$\sim$78\,K`（Table VIII）
- "our 1.3M-parameter reconstruction encoder" → "our 78K-parameter reconstruction encoder"
- §Sensitivity：原 "edge-MCU/quantized deployment remains future work" 句替换为上表实测 + 流式/恢复结论；RPi/MCU 标为未来工作
