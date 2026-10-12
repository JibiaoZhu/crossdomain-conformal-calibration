# 大修建议执行报告（第一批）— 2026-10-12

对象：`论文_LaTeX源码/`（IEEE TIM 投稿版）。论文修改后 10 页、0 未定义引用、宽表已修。
数据依据：`F:\AutoSci_CrossDomain_Handoff\reports\revision-2026-10-12\revision_analysis.json`

## 已执行（28 处论文补丁 + 复核性计算）

| 条目 | 处理 | 关键证据/结果 |
|---|---|---|
| B1 相对增益 ACI 无保证 | **已改**：重定位为 scale-free 变体；补 log-阈值两点 SA 论证（长期告警率 = α + O(γ)）；注明不继承加性形式有限样本常数 | 方法节 + Limitations |
| B2 告警率未报告 | **已补**：新增 Table「stream-wide alarm rate」+验证段 | ACI 全流告警率 20/24 个格 ≤0.015，最大偏差 0.033；static 最高偏差 0.70 |
| B4 F4 三数矛盾 | **已诊断**：极限环实锤 | F4 阈值 p5–p95 = 0.005–2.0（400×），F6 = 0.014–2.8；F4 最长静默 579 窗≈全为连续漏报；NA-ACI κ=3 压到 51–106 窗。缺陷拆为 F3 型（预算消耗）+ F4/F6 型（windup 致盲） |
| B6 数据事实 | **部分执行**：MFPT 97.7/48.8 kHz 已改；Paderborn 两条**驳回**（见下） | 加澄清句避免审稿人误判 |
| B7 数字/术语统一 | **已完成**：0.72→0.90(κ=3)；matched→essentially unchanged (0.090→0.094)；interpretering→interpreting；跨节引用修正；interval→threshold inflation（全文 8 处） | — |
| B8 物理时间换算 | **已完成**：延迟双单位（F1 2–4 min；F4 2–21 ms；F6 5–10 ms）；基准节加窗长跨度说明 | — |
| C6 ε/q0 | **部分完成**：q0=源分位、ε=1e-6 已写入；warm-up 实验队列 | — |
| D5 AUPRC | **已完成**：Table II 加 AUPRC 列；低于先验检查：F2 0.099<0.106、F5 0.725<0.729（与低于随机结论独立互证） | — |
| H1 代码可用性 | **部分完成**：论文内补 GitHub URL（intro/benchmark/conclusion）；Zenodo DOI 队列 | https://github.com/JibiaoZhu/crossdomain-conformal-calibration |

## 评审驳回/修正（有据）

- **B6-Paderborn（128 kHz / run-to-failure）**：评审按 2016 经典版（64 kHz、定长、无退化）判断；本数据是新版 *time-varying run-to-failure* release（HF 镜像 B01–B13 取 B01–B04）：`data_B01_*` 204,800 点 × 1.6 s = **128 kHz**、测量随时间积累、README 明确 run-to-failure 协议。论文原表述正确。已加一句“区别于经典定长版”防误判。
- **B4 “不可能共存”**：固定阈值下不可能——但 ACI 阈值在极限环下正是这三个数共存的原因（已给出时序证据，见上）。
- **E1 步长表**：F2 步长不是 10 min（每窗滑步=128 样本；按逐涡轮流计 ≈21 h/步），F4 “5–11 ms” 未计 Paderborn 128 kHz（实际 1–10.7 ms）。建议本身（按告警频次/h 反推 α）合理，已队列。
- **B8 “CWRU 与 Paderborn 差 5 倍”**：实际 ~10.7 倍（12 vs 128 kHz）；按真实值换算。
- **F8 数据集计数**：“五个数据集”不实——本基准 7 个数据集、4 族、6 折；现文本已精确。

## 合理但队列（按优先级）

1. **硬缺口（下一步）**：B5 目标域标定基线族（目标分位/扩展窗/周期重标定/DSPOT）、D9 校准段污染、D8 开集故障、D10 标签噪声。
2. **统计与协议**：D1 TOST/多重比较、D4 block bootstrap、D6 macro/micro、D13 种子散布。
3. **方法扩展**：C1 命题与有限样本界、C2 排除率监控+回退、C3 缓变故障失效测试、C4 κ 全景（已有 κ=1.5/2/3/5 数据）、C5 γ 扫描、C7/C9/G3 标签偏移与加权保形、C8 DtACI/PID 正面对比、C10 P² 追踪基线。
4. **操作化**：E1 告警频次口径重报、E2 迟滞/去抖、E4 决策代价、E7 决策图+监控伪代码、E8 神谕分位分解（可立即从 traces 算）。
5. **叙事与文献**：F3 测量学口径（GUM/VIM）、F5 符号表、F6 2×2 跨域网格、F7 分数 ECDF 图、G1 保形异常检测线、G2 变点触发、G4 与 weighted conformal 的显式定位、H2 数据许可、H3 官方代码对比、H4 训练细节。
6. **I 组（礼貌拒绝）**：同意评审的拒绝理由，写入 response letter。
