# ManNet HBF Reproduction

## Project
Reproducing "Deep Unfolding Hybrid Beamforming Designs for THz Massive MIMO Systems"
(Nguyen et al., IEEE TSP vol. 71, 2023, DOI 10.1109/TSP.2023.3322852). There is no official code.
The implementation is a single PyTorch script, `mannet_hbf.py`, originally written on Kaggle by a teammate.
The paper PDF is in `paper/` — check equations and algorithms against it, not against memory.

## Infrastructure
- GitHub repo = source of truth. Commit code, results (numbers, figures), and notes here.
- Kaggle = GPU runner only. No auto-sync; every `kaggle kernels push` creates a new version AND starts a run.
- Original notebook: tanishadixit0206/hybrid-beamforming-mannet (read-only for us)
- Our kernel: slytherin1175/hybrid-beamforming-mannet
- Kaggle auth: `KAGGLE_API_TOKEN` user env var (set via setx). NEVER copy credentials into the repo.

## Workflow (every change)
1. Edit code → `python mannet_hbf.py --quick` locally (must pass before any Kaggle push).
2. Commit + `git push`.
3. `kaggle kernels push -p .` → poll `kaggle kernels status slytherin1175/hybrid-beamforming-mannet`
   until complete → `kaggle kernels output slytherin1175/hybrid-beamforming-mannet -p ./out`.
4. Copy the key numbers/figures into `results/` with a short note (git commit hash, settings), commit.
Do not push to Kaggle for trivial edits — GPU quota is limited (~30 h/week).

`--quick`: Nt=16 (4x4 UPA), K=16, 2 epochs, 64 train / 5 test channels. Smoke test only (~15 s on CPU);
its SE numbers are meaningless. Local outputs go to `./out`, Kaggle outputs to `/kaggle/working`.

## Paper setup (full scale)
Nt=128 (UPA), Nr=NRF=Ns=2, K=128, P=4 paths, fc=300 GHz, BW=30 GHz, cyclic prefix K/4.
|D|=320 training channels, batch 32, 30 epochs, Adam, I_net^train=3, L=3, I_net=10 at inference.
Evaluate on 100 test channels. SE normalised to fully digital (FC) or MO-AltMin-FC (SC).

## Current status
Done: wideband channel with beam squint; ManNet + subManNet (3,072 params each), unsupervised training;
FC: ManNet 92.5% of DBF @10 dB, OMP 90.7%, MO-AltMin 92.9%, AO 93.0%;
SC: Algorithm 3 mapping, subManNet, heuristic ManNet-Dyn-SC (both 92.3% of MO-AltMin-FC @10 dB);
ManNet-Fix-SC 88.4% (slightly below paper's 89–93%).

Known deviations from the paper (keep documented, don't silently "fix"):
- LR 1e-2 instead of 1e-4 (1e-4 did not converge for us).
- Our MO-AltMin/AO baselines are stronger and faster than the paper's → ManNet ties them rather than beating them.
- Algorithm 3 pseudocode is ambiguous; we implement one reading. Document which.

## To do (rough priority)
1. Fig. 6: SE vs Nt ∈ {16, 32, 64, 128} — retrain ManNet/subManNet per Nt.
2. Fig. 4: SE vs iteration count (deep-unfolding methods at L, 2L, 3L…; AO/MO-AltMin vs inner iterations).
3. Figs. 2–3: training-loss curves (I_net^train ∈ {1,3}) and SE vs L for I_net ∈ {1,2,5,10}.
4. Baselines: unfolded PGA [Lavi & Shlezinger], SDR-AltMin-Fix-SC, DSP-Dyn-SC [Park et al.], narrow-ManNet-SC.
5. Figs. 7–9: operation counts (adds+mults) and runtime; Table II CNN parameter comparison.

## Conventions
- Keep everything runnable as one script on Kaggle; split into modules only if it stays Kaggle-compatible.
- Each experiment selectable via CLI flag (e.g. `--exp fig6`), saving results to out/<exp>/ as .npy/.json + .png.
- Fix random seeds; log settings with every result.
- Be concise in explanations; flag any place our implementation departs from the paper.
