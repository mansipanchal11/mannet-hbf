# Fig. 4 — convergence (SE vs iterations), Nt=128

- Code commit: 943ae6d (`python mannet_hbf.py --exp fig4`), Kaggle kernel v2, T4 GPU.
- Settings: Nr=NRF=Ns=2, K=128, L=3, 30 epochs, I_train=3, 320 train / 100 test channels, LR 1e-2, seed 0 (see fig4.json).
- x-axis: ManNet/subManNet = I_net * L layers (3, 6, ..., 60; I_net = 1..20, trained with I_train=3);
  AO = alternating sweeps (1..100, no early stopping); MO-AltMin = total Riemannian-CG iterations
  (digital precoder re-solved every 5 inner iterations, up to 100).

SE (bits/s/Hz):

| SNR | DBF | OMP | ManNet-FC @3 / @30 / @60 | subManNet-SC @3 / @30 | AO-FC @3 / @10 / @100 | MO-AltMin @10 / @50 / @100 |
|---|---|---|---|---|---|---|
| 10 dB | 17.13 | 15.54 | 14.44 / 15.86 / 15.86 | 13.82 / 14.69 | 15.58 / 15.90 / 15.94 | 15.32 / 15.89 / 15.94 |
| 20 dB | 23.76 | 22.14 | 20.98 / 22.46 / 22.46 | 20.32 / 21.25 | 22.17 / 22.50 / 22.54 | 21.89 / 22.49 / 22.54 |

Observations / deviations:
- ManNet saturates after ~30 layers (I_net=10, ~0.5% below AO); beyond I_net=10 there is no gain.
- Our AO converges in ~20 sweeps and MO-AltMin in ~100 inner iterations; the paper reports neither converged after 100.
  This is the known "stronger baselines" deviation, so ManNet does not show a convergence-speed advantage per iteration here.
  Per-iteration cost differs, so the paper's complexity figures (7-9) are the fairer comparison.
- MO-AltMin x-axis definition (digital update every 5 inner steps) is our reading; paper does not specify.
- Paper's Fig. 4 also has SDR-AltMin-Fix-SC and DSP-Dyn-SC (not implemented, to-do #4); here subManNet uses the strongest-subcarrier mapping.
