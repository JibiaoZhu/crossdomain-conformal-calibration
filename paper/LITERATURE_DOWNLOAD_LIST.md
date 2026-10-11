# 文献下载清单（供学校订阅库/公开渠道获取）

> 目的：满足"大量查找高水平文献（IEEE Xplore / Springer / ScienceDirect / arXiv）"要求。以下条目均已在项目 wiki 中建页并验证过出处（arXiv ID / DOI），可直接检索下载；DOI 以论文 `references.bib` 为准，投稿前再核一次。

## A. 校准 / Conformal 方法（核心引用）

| # | 文献 | 渠道 | ID |
|---|------|------|----|
| 1 | Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift* (NeurIPS 2021) | arXiv | 2106.00170 |
| 2 | Lei et al., *Distribution-Free Predictive Inference for Regression* (JASA 2018) | arXiv | 1604.04173 |
| 3 | Zaffran et al., *Adaptive Conformal Predictions for Time Series* (ICML 2022) | arXiv | 2202.07282 |
| 4 | Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction*（背景教程） | arXiv | 2107.07511 |
| 5 | Siffer et al., *Anomaly Detection in Streams with Extreme Value Theory* (KDD 2017) | ACM DL / 学校库 | DOI 10.1145/3097983.3098144 |
| 6 | Stankeviciute et al., *Conformal Time-Series Forecasting* (NeurIPS 2021)（可选扩展） | arXiv | 2110.12149 |

## B. 跨域异常检测 / 故障诊断 / TSFM（对比与背景）

| # | 文献 | 渠道 | ID |
|---|------|------|----|
| 7 | CICADA: Cross-domain Interpretable Coding (2025) | arXiv | 2505.00415 |
| 8 | DACAD: Domain Adaptation Contrastive Learning (TKDE 2025) | arXiv / IEEE | 2404.11269 |
| 9 | A Survey of Deep Transfer Learning for AD in Industrial Time Series | arXiv | 2307.05638 |
| 10 | Frequency Matters: TSFM Failures Under Spectral Shift | arXiv | 2511.05619 |
| 11 | How Foundational are Foundation Models for TS Forecasting | arXiv | 2510.00742 |
| 12 | Chronos: Learning the Language of Time Series (TMLR 2024) | arXiv | 2403.07815 |
| 13 | PatchTST: A Time Series is Worth 64 Words (ICLR 2023) | arXiv | 2211.14730 |
| 14 | PETSA: Parameter-Efficient Test-Time Adaptation (2025) | arXiv | 2506.23424 |

## C. 数据集引文

| # | 数据集 | 渠道 |
|---|--------|------|
| 15 | CARE to Compare（风电 SCADA，arXiv:2404.10320） | arXiv |
| 16 | XJTU-SY（IEEE Trans. Reliability 2020，DOI 10.1109/TR.2018.2882682） | IEEE Xplore |
| 17 | MFPT 轴承故障数据集 | 公开数据集页（见 datasets/MFPT/README） |
| 18 | Paderborn KAt DataCenter（time-varying run-to-failure） | https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter |
| 19 | PHM Society 2016 Data Challenge (CMP) | 官方页（见 references.bib） |
| 20 | PVDAQ / OEDI (NREL) | https://oedi-data-lake.s3.amazonaws.com/pvdaq/ |

## D. 建议在订阅库做的补充检索式（近 3 年优先）

1. `("conformal prediction" OR "adaptive conformal") AND ("domain shift" OR "distribution shift") AND ("time series" OR "anomaly detection")` — IEEE Xplore / Springer
2. `("cross-domain" OR "domain generalization") AND ("anomaly detection" OR "fault diagnosis") AND ("industrial" OR "SCADA")` — 近 3 年
3. `("calibration" OR "threshold selection") AND ("predictive maintenance" OR "condition monitoring")` — 目标期刊过滤（IEEE TII / TIM / KBS / RESS）
4. 把上面命中的高相关论文（约 10–15 篇）加入 wiki（`ingest`）后，可进一步强化 Related Work 与竞争定位。

## 备注
- 论文当前引用 18 条已全部闭环；本清单是"扩展检索"用途，不阻塞投稿。
- 若学校库中检索到与 NA-ACI 直接竞争的 2025–2026 校准工作，请尽快反馈（novelty 复查用）。
