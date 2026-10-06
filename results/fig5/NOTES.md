# Fig. 5 — SE vs SNR (Nt=128)

- Code commit: 5ecb49b (`python mannet_hbf.py --exp all`, "main" part), Kaggle kernel v4, T4. Same settings as other figures (seed 0, 30 epochs, 320/100 channels, LR 1e-2).
- results.npz holds all SNRs (-10..10 dB); se_vs_snr.png is the plot.

@10 dB, % of fully digital (DBF = 17.134 bits/s/Hz): ManNet-FC 92.5, MO-AltMin 92.9, AO 93.0, OMP 90.7, ManNet-FC (I=1) 84.3,
ManNet-Dyn-SC 85.7, subManNet-Dyn-SC 85.7, ManNet-Fix-SC 82.1, AO-Dyn-SC 85.9, AO-Fix-SC 71.5.
SC designs relative to MO-AltMin-FC: ManNet-Dyn-SC 92.3, subManNet-Dyn-SC 92.3, ManNet-Fix-SC 88.4.

These reproduce the numbers previously recorded in CLAUDE.md exactly (deterministic seed), so the earlier un-archived run is now backed by data.
Same Kaggle run also regenerated Figs. 2-4, 6 (copies in out/kaggle_all/, not committed; earlier results/ dirs unchanged).
