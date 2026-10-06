# Figs. 2-3 — training loss and SE vs L

- Code commit: 0894192 (`python mannet_hbf.py --exp fig23`), Kaggle kernel v3, T4 GPU.
- Settings: Nr=NRF=Ns=2, K=128, 30 epochs, 320 train / 100 test channels, LR 1e-2, seed 0 (see fig23.json).
- Fig. 2: Nt=64, L=6, I_train in {1,3}, ManNet and subManNet (strongest-subcarrier mask). Raw (un-normalised) loss in the json/npz.
  The plot normalises each network's curves by the larger epoch-1 loss; the paper's normalisation is unspecified.
- Fig. 3: Nt=128, I_train=3, SNR 20 dB (DBF = 23.51), one ManNet trained per L in 1..6 (the paper's L range was not checked), I_net in {1,2,5,10}.

Fig. 2 training loss (epoch 1 / 10 / 30):

| Network | I_train=1 | I_train=3 |
|---|---|---|
| ManNet | 13.10 / 10.67 / 9.96 | 11.61 / 6.89 / 6.52 |
| subManNet | 13.09 / 11.15 / 10.65 | 11.60 / 7.77 / 7.71 |

Fig. 3 SE (bits/s/Hz) @ 20 dB:

| I_net | L=1 | L=2 | L=3 | L=4 | L=5 | L=6 |
|---|---|---|---|---|---|---|
| 1  | 10.02 | 20.71 | 20.76 | 20.81 | 20.80 | 20.76 |
| 2  | 11.05 | 21.57 | 21.60 | 21.58 | 21.54 | 21.57 |
| 5  | 11.60 | 22.09 | 22.08 | 22.05 | 22.03 | 22.04 |
| 10 | 11.72 | 22.24 | 22.23 | 22.19 | 22.18 | 22.17 |

Observations vs paper:
- Fig. 2 matches qualitatively: I_train=3 converges faster and lower, ~flat after ~10 epochs; I_train=1 is still decreasing at epoch 30;
  ManNet's converged loss is below subManNet's. (Caveat: the logged loss is averaged over the I_train inner steps, so the two I_train curves are not strictly like-for-like.)
- Fig. 3: SE depends strongly on I_net (I_net 1 -> 10 gains ~1.5 bits/s/Hz) and hardly on L for L>=2, as in the paper.
  Differences: our L=1 is far worse (~10-12 vs ~21-22): a single layer has x_0=0, so w1 is unused and the net takes one gradient step per iteration.
  We see no (slight) SE increase with L beyond 2; it is flat or marginally decreasing. L=3 is adequate, consistent with the paper's choice.
