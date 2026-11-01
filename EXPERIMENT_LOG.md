# EXPERIMENT_LOG

**No real experiment has been run. No model has been trained on SMC-LUD. Nothing below may be quoted as a result.**
The real dataset and the pretrained ImageNet weights were both unreachable from the build sandbox (HTTP 403).

Entry template (each run also writes a machine-readable `manifest.json`):
`ID · date · milestone · model · weights · split_id · seed · config · metrics · status · notes`

## Pipeline checks (M2, 2026-11-01): synthetic data, NOT results
Data: 160 patients, learnable toy signal (bright blob on noise, 64 px) written by `tests/synthetic.py`; Clean variant; patient-level
train/val/test = 70/20/10; split_id `93514ef72460`; seed 42; 8 epochs; from-scratch weights (no pretrained weights available); CPU;
torch 2.15.0, Python 3.13. All runs carry a "NOT A RESULT" watermark on every plot. Test set: 30 images from 16 patients.

| run | selected epoch | test AUROC [95% patient-bootstrap CI] | sens / spec @0.5 | val-fitted threshold → sens / spec | test size | time |
|---|---|---|---|---|---|---|
| efficientnet_b0 | 8/8 | 1.000 [1.000, 1.000] | 0.733 / 1.000 | 0.104 → 1.000 / 1.000 | 30 imgs / 16 pts | 27.9s |
| resnet50 | 5/8 | 1.000 [1.000, 1.000] | 1.000 / 1.000 | 0.485 → 1.000 / 1.000 | 30 imgs / 16 pts | 50.0s |
| null_resnet50 | 3/8 | 0.560 [0.379, 0.759] | 0.200 / 0.900 | 0.419 → 0.300 / 0.700 | 30 imgs / 16 pts | 50.3s |

What these runs do and do not show:
- They show the code path (split guard → train → validation-based selection → single test evaluation → plots/manifest) runs end to end for both
  architectures, and that a **patient-label-shuffle null control collapses to chance** (AUROC 0.56, CI spans 0.5) while the unshuffled toy task is solved. That is the intended behaviour of a leak-free pipeline.
- They say **nothing** about liver ultrasound, SMC-LUD, or transfer learning. The toy signal is trivially separable; the perfect scores and degenerate CIs (1.000-1.000, n=16 patients) are artefacts of that.
- Observed pitfall (fixed): EfficientNet-B0 first scored AUROC 1.000 but sensitivity 0.000 at the fixed 0.5 threshold (probabilities 0.08 vs 0.17 - ranked correctly, all below 0.5), and the first-epoch-at-AUROC-1.0 checkpoint had validation accuracy 0.5. Fixed by breaking AUROC ties on validation loss (epoch 6 → 8) and by also reporting a validation-fitted threshold. Real-data baselines must report both operating points.

## Real-data experiments
None yet. Run list will start at **E-001 (EfficientNet-B0, seed 42)** once the real data and pretrained weights are available (see HANDOFF).
