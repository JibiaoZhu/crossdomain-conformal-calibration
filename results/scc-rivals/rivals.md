# Rival methods, mechanism-level reimplementation (6 folds)

DACAD-style: anomaly-injection discriminative head + source-target moment alignment.
CICADA-style: 4 reconstruction experts over source groups + divergence-based expert weighting.

- F1/dacad_style: AUROC=0.602 gaps static=0.065 aci=0.003 na5=0.003 | R-F1(mean) aci=0.137 na5=0.137
- F1/cicada_style: AUROC=0.622 gaps static=0.075 aci=0.006 na5=0.008 | R-F1(mean) aci=0.093 na5=0.093
- F2/dacad_style: AUROC=0.491 gaps static=0.037 aci=0.003 na5=0.003 | R-F1(mean) aci=0.135 na5=0.135
- F2/cicada_style: AUROC=0.353 gaps static=0.499 aci=0.010 na5=0.014 | R-F1(mean) aci=0.100 na5=0.102
- F3/dacad_style: AUROC=0.846 gaps static=0.155 aci=0.093 na5=0.092 | R-F1(mean) aci=0.567 na5=0.568
- F3/cicada_style: AUROC=0.840 gaps static=0.155 aci=0.101 na5=0.059 | R-F1(mean) aci=0.585 na5=0.589
- F4/dacad_style: AUROC=0.262 gaps static=0.064 aci=0.002 na5=0.002 | R-F1(mean) aci=0.228 na5=0.228
- F4/cicada_style: AUROC=0.991 gaps static=0.158 aci=0.006 na5=0.005 | R-F1(mean) aci=0.237 na5=0.718
- F5/dacad_style: AUROC=0.588 gaps static=0.076 aci=0.002 na5=0.002 | R-F1(mean) aci=0.257 na5=0.257
- F5/cicada_style: AUROC=0.319 gaps static=0.238 aci=0.004 na5=0.004 | R-F1(mean) aci=0.258 na5=0.272
- F6/dacad_style: AUROC=0.529 gaps static=0.143 aci=0.001 na5=0.001 | R-F1(mean) aci=0.198 na5=0.198
- F6/cicada_style: AUROC=0.781 gaps static=0.133 aci=0.011 na5=0.011 | R-F1(mean) aci=0.262 na5=0.340
