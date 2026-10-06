# Fig. 6 — SE vs Nt @ 10 dB

- Code commit: c56f497 (`python mannet_hbf.py --exp fig6`), Kaggle kernel v1, T4 GPU, ~160 s total.
- Settings: Nr=NRF=Ns=2, K=128, 30 epochs, 320 train / 100 test channels, LR 1e-2, seed 0, I_net=10 (full settings in fig6.json).
- ManNet/subManNet retrained per Nt; trainable params scale with Nt (384 / 768 / 1536 / 3072 for Nt=16/32/64/128).

SE as % of fully digital (DBF) @ 10 dB:

| Nt | ManNet-FC | MO-AltMin-FC | AO-FC | OMP-FC | ManNet-Dyn-SC | subManNet-Dyn-SC | ManNet-Fix-SC | AO-Dyn-SC | AO-Fix-SC |
|---|---|---|---|---|---|---|---|---|---|
| 16  | 92.2 | 93.0 | 93.4 | 90.5 | 86.0 | 85.8 | 79.5 | 86.0 | 79.0 |
| 32  | 92.2 | 93.0 | 93.0 | 90.5 | 85.4 | 85.5 | 79.6 | 85.6 | 74.7 |
| 64  | 92.7 | 93.0 | 93.2 | 90.8 | 85.8 | 85.9 | 80.9 | 86.1 | 73.3 |
| 128 | 92.2 | 92.5 | 92.7 | 90.3 | 85.3 | 85.5 | 81.4 | 85.7 | 70.8 |

Observations: ManNet-FC beats OMP and trails MO-AltMin/AO by <1%, consistent with the known deviation (our baselines are strong).
SC columns are % of DBF (not of MO-AltMin-FC as in the CLAUDE.md status line). ManNet-Fix-SC stays well above AO-Fix-SC at large Nt.
Deviations unchanged: LR 1e-2; Algorithm 3 reading as implemented.
