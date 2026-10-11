# Traces analysis (R2 calibration baselines + R5 ablations + CIs + operational metrics)

## F1 (phm16)
- seed 0 | static_a0.1=0.990 aci_a0.1=0.897 aci_k5_a0.1=0.908 aci_k20_a0.1=0.973 rollingq_a0.1=0.857 evtpot_a0.1=0.990
  - ACI a0.1: FA/window=0.103 (per 1000 windows 103.0); detection delay windows: median=1.0 (runs=41, detected=7)
- bootstrap CI gap static_a0.1: [0.084, 0.094] (seed 0)
- bootstrap CI gap aci_a0.1: [0.000, 0.017] (seed 0)
## F2 (care)
- seed 0 | static_a0.1=0.185 aci_a0.1=0.890 aci_k5_a0.1=0.858 aci_k20_a0.1=0.699 rollingq_a0.1=0.859 evtpot_a0.1=0.185
  - ACI a0.1: FA/window=0.110 (per 1000 windows 109.7); detection delay windows: median=0.5 (runs=121, detected=24)
- bootstrap CI gap static_a0.1: [0.703, 0.726] (seed 0)
- bootstrap CI gap aci_a0.1: [0.002, 0.019] (seed 0)
## F3 (pv)
- seed 0 | static_a0.1=0.999 aci_a0.1=0.998 aci_k5_a0.1=0.999 aci_k20_a0.1=0.999 rollingq_a0.1=0.930 evtpot_a0.1=0.999
  - ACI a0.1: FA/window=0.002 (per 1000 windows 2.3); detection delay windows: median=0.0 (runs=168, detected=79)
- bootstrap CI gap static_a0.1: [0.097, 0.100] (seed 0)
- bootstrap CI gap aci_a0.1: [0.094, 0.100] (seed 0)
## F4 (cwru+paderborn)
- seed 0 | static_a0.1=0.999 aci_a0.1=0.910 aci_k5_a0.1=0.930 aci_k20_a0.1=0.997 rollingq_a0.1=0.898 evtpot_a0.1=0.999
  - ACI a0.1: FA/window=0.090 (per 1000 windows 90.3); detection delay windows: median=2.0 (runs=5, detected=5)
- bootstrap CI gap static_a0.1: [0.098, 0.099] (seed 0)
- bootstrap CI gap aci_a0.1: [0.004, 0.015] (seed 0)
## F5 (mfpt)
- seed 0 | static_a0.1=0.995 aci_a0.1=0.899 aci_k5_a0.1=0.910 aci_k20_a0.1=0.928 rollingq_a0.1=0.851 evtpot_a0.1=0.995
  - ACI a0.1: FA/window=0.101 (per 1000 windows 100.7); detection delay windows: median=49.0 (runs=1, detected=1)
- bootstrap CI gap static_a0.1: [0.094, 0.097] (seed 0)
- bootstrap CI gap aci_a0.1: [0.000, 0.009] (seed 0)
## F6 (xjtu)
- seed 0 | static_a0.1=0.999 aci_a0.1=0.915 aci_k5_a0.1=0.934 aci_k20_a0.1=0.942 rollingq_a0.1=0.909 evtpot_a0.1=0.999
  - ACI a0.1: FA/window=0.085 (per 1000 windows 85.0); detection delay windows: median=1.0 (runs=3, detected=3)
- bootstrap CI gap static_a0.1: [0.098, 0.099] (seed 0)
- bootstrap CI gap aci_a0.1: [0.010, 0.020] (seed 0)

## Operational metrics (seed 0, ACI alpha=0.1)
| fold | target | rate(Hz) | stride_s | FA/1000win | med delay (win) |
|---|---|---|---|---|---|
| F1 | phm16 | 1.0 | 128 | 103.0 | 1 |
| F2 | care | 0.0016666666666666668 | 7.68e+04 | 109.7 | 0 |
| F3 | pv | None | nan | 2.3 | 0 |
| F4 | cwru+paderborn | None | nan | 90.3 | 2 |
| F5 | mfpt | 97656.0 | 0.001311 | 100.7 | 49 |
| F6 | xjtu | 25600.0 | 0.005 | 85.0 | 1 |

## Runtime/memory (R6, from traces meta; per run)
(see meta.json: t_train_s, t_score_eval_s, t_scal_s, update_us_per_step, peak_mem_mb)
