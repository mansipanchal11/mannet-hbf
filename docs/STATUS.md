# Status — 2026-10-07

Full write-up: Claude Doc "ManNet HBF Reproduction – Progress Update (7 Oct 2026)". This file is the repo-side summary.

## Scope
Reproduce the numerical results of Nguyen et al., IEEE TSP vol. 71, 2023 (no official code): Figs. 2-9 and Table II for
ManNet/subManNet and the paper's baselines. Out of scope: new methods, other channel models, hardware, weakening our baselines.
Success = same setup/axes/methods per figure, honest comparison with the paper, mismatches documented not hidden.

## Status
| Item | Status | Where |
|---|---|---|
| Fig. 5 SE vs SNR | done, archived | results/fig5 |
| Fig. 6 SE vs Nt | done | results/fig6 |
| Fig. 4 convergence | done (FC, subManNet-SC); SDR-AltMin-Fix-SC, DSP-Dyn-SC missing | results/fig4 |
| Figs. 2-3 | done | results/fig23 |
| Figs. 7-9, Table II | not started | |
| Baselines: unfolded PGA, SDR-AltMin-Fix-SC, DSP-Dyn-SC, narrow-ManNet-SC | not started | |

## Infrastructure
Kaggle kernel v4 source == repo `mannet_hbf.py` (verified by pulling it). Kaggle default experiment is set by `KAGGLE_ARGV`
in the script (currently `--exp all`, runs Fig. 5 then 2-4, 6).

## Next
1. Overlay Figs. 2-6 on the paper's curves.
2. Missing baselines (SDR-AltMin-Fix-SC and DSP-Dyn-SC first).
3. Figs. 7-9, Table II.
4. Multiple seeds with CIs; revisit Algorithm 3 reading; investigate LR 1e-4.
