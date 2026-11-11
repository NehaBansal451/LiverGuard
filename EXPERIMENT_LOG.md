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

## Robustness-audit pipeline checks (M3, 2026-11-02): synthetic checkpoints, NOT results
Audited the two M2 smoke checkpoints (trained on the toy "bright blob" signal, from-scratch weights) with `scripts/audit_robustness.py`,
validation split (66 images / 32 patients), seed from `configs/robustness.yaml`, 12 conditions x 5 levels (levels 1-3 "plausible", 4-5 "stress", a-priori tiers, not expert-validated).
Every plot and manifest carries the "NOT A RESULT" watermark / `valid_as_result: false`.

| model | clean AUROC / acc | most disruptive (mean dAUROC, mean flip rate, median MFS) | 2nd | 3rd |
|---|---|---|---|---|
| resnet50 | 1.000 / 1.000 | brightness_dark (-0.463, 0.503, 2) | gamma_high (-0.310, 0.391, 4.5) | brightness_bright (-0.300, 0.306, 4.5) |
| efficientnet_b0 | 1.000 / 0.879 | brightness_dark (-0.453, 0.427, 5) | brightness_bright (-0.288, 0.439, 3) | gamma_high (-0.244, 0.400, 4) |

Pipeline checks that DID pass (these are what the runs establish):
- **Reproducibility**: re-running the resnet50 audit gave byte-identical `per_image.csv` and `summary_by_level.csv`. The per-image RNG is keyed by SHA-256(seed, image_id, condition), so results do not depend on batch size, image order, level, or which other conditions are run (tests cover each).
- **Every distortion tested independently** (`tests/test_distortions.py`): shape/dtype/range contract, identity-like at weakest level, monotone strength, determinism, seed sensitivity, and an analytic check per family (e.g. brightness scales intensity exactly when unclipped; contrast preserves the mean; gamma matches 255*(x/255)^g; JPEG changes bytes but not size; rotation/crop geometry).
- Identity and blackout sanity checks in the engine tests (identity -> flip rate 0; all-black input -> collapse); the clean level equals the training-time predictions.
- Test split refused without `--final-report`; checkpoint/index `split_id` mismatch refused.

Problems found and fixed during M3:
1. float32 intermediate rounding broke exact-scaling brightness test (ties) -> implementation moved to float64 (not the test).
2. Fidelity proxies (PSNR/correlation) flagged harmless 95% centre-crops and small rotations as "destroyed": they compare unaligned pixels. Geometric families now skip those two checks (regression test added). Whether a lesion leaves the field of view is NOT assessable by pixel metrics; that needs expert review (`review/label_preservation_review.csv`).
3. Heatmap annotation colour was chosen from the signed value rather than the cell colour; tier labels and gallery title overlapped -> fixed and regenerated.

What these runs do NOT show: anything about liver ultrasound. The toy model keys on a bright blob, so brightness/gamma sensitivity is an artefact of the toy task; 66 images give coarse flip rates (steps near 0.5 mean "all one class" collapse). Real-data audits must use the validation split first, with patient-level bootstrap CIs (`--n-boot`).

## Reliability + explainability pipeline checks (M4/M5, 2026-11-03): synthetic checkpoints, NOT results
Same smoke checkpoints and validation split as the M3 audit (66 images / 32 patients, toy "bright blob" signal, from-scratch weights).
`scripts/reliability.py` (20 MC-Dropout samples, 5 TTA views, 10 bins, 300 patient-bootstrap reps) and `scripts/explain.py` (centre-proxy ROI, top-20% CAM, 3 matched random controls). All outputs watermarked, `valid_as_result: false`.

Calibration (uncalibrated / temperature scaling cross-fitted over patient folds on val):

| checkpoint | ECE width (95% CI) | ECE / Brier / NLL cross-fitted | Brier | errors at 0.5 |
|---|---|---|---|---|
| resnet50 | 0.001 [0.000, 0.002] | 0.000 / 0.000 / 0.000 | 0.000 | 0 |
| efficientnet_b0 | 0.130 [0.081, 0.213] | 0.107 / 0.104 / 0.312 | 0.101 | 8 FN |
| null_resnet50 (shuffled labels) | 0.213 [0.144, 0.351] | 0.023 / 0.251 / 0.696 | 0.300 | 31 |

Selective prediction (AURC; oracle / random): efficientnet_b0 TTA std 0.036, margin 0.037, MC-Dropout std 0.048, MC mutual info 0.049 (oracle 0.009, random 0.121); null model margin 0.637 vs random 0.470, i.e. **worse than no ranking** (uncertainty ranks errors badly when the model learned nothing).

Region masking / Grad-CAM faithfulness (AUROC on masked images; controls = same-area random placement):

| checkpoint | full | ROI only | background only | CAM-delete minus random-delete (evidence drop, 95% CI) |
|---|---|---|---|---|
| resnet50 | 1.00 | 1.00 (control 0.67) | 0.50 (control 0.86) | -0.030 [-0.132, 0.085] |
| efficientnet_b0 | 1.00 | 1.00 (control 0.98) | 0.77 (control 0.98) | -0.079 [-0.129, -0.033] |

Grad-CAM vs fully randomised model (Spearman, 24 images): resnet50 median 0.55 (42% of images > 0.5); efficientnet_b0 median 0.41 (29%).

What these show: the code paths run on real model outputs, deterministically, and the guards bite. What can be read from them as *behaviour of these toy models* (not liver ultrasound):
- CAM-delete did **not** hurt more than deleting a random same-area region (paired difference negative / CI spans 0), and keep-only variants collapse to a constant output for resnet50 (AUROC exactly 0.50, identical to its control). That is the out-of-distribution-fill caveat in action: a mean-grey image with a small visible patch is outside the training distribution, so these probes are confounded for these models.
- Grad-CAM correlates 0.4-0.55 with a randomised network on this toy data, so on these models the maps are partly image structure (the sanity check is doing its job).
- Perfect-model ECE ~0 (resnet50) and the null model's temperature-scaled ECE 0.023 with Brier still 0.251 show that **ECE can look fine while discrimination is nil**; always read ECE with AUROC/Brier. Single-bin collapse after scaling (all confidences ~0.5) explains the low ECE.

Problems found and fixed in M4/M5:
1. Grad-CAM forward hook stayed installed and crashed no-grad passes (occlusion) with "cannot register a hook on a tensor that doesn't require gradient" -> hook now ignores passes without gradients (regression test).
2. Annotation-mask lookup used the bare file stem, so `0.png` of one patient would match every patient's `0.png` -> now keyed by the image's relative path (test).
3. Plot issues: case-panel row titles overlapping, occlusion colour scale stretching pure noise (now floored at +-0.1, true max printed), failure-gallery title overstating "confident" errors, legend over data.
4. Test mistakes (mine): a threshold example whose confidence was exactly 0.9, and a guessed column name for predictions.csv.

Not shown by any run: real-data calibration, ROI from real annotations, Clean-vs-Caliper analysis.

## Robust-training pipeline check (M6, 2026-11-04): synthetic, NOT results
`scripts/robust_train.py all` then `final --final-report` with `configs/smoke_tiny.yaml` (tiny CNN, 4 epochs, 32 px, from scratch), 3 seeds (42-44), 200 bootstrap reps,
same synthetic index as M3-M5. Val: 66 images / 32 patients; **test: 30 images / 16 patients**. Watermarked, `valid_as_result: false`.
Protocol exercised: baselines trained without evaluating test -> val audit -> weaknesses -> plan -> 5 interventions x 3 seeds -> val ablation with paired patient-bootstrap -> selection frozen with checkpoint hashes -> test re-audit of all models (identical images/thresholds/distortions; refused without `--final-report` and without an intact selection).

What the run showed (about these toy models only):
- The 4-epoch baselines are weak (val clean AUROC 0.60), so the diagnosis flagged 11 of 12 conditions as "weaknesses" and no negative control could be formed. A real model is needed before weaknesses mean anything.
- Frozen selection (validation): `noise`. Test re-audit, mean over seeds, difference vs baseline (seed range): clean AUROC +0.064 [0.000, +0.133] inconclusive; plausible-tier AUROC +0.049 [+0.023, +0.062] **improved** (all seeds); plausible flip rate -0.037 inconclusive; clean ECE -0.056 inconclusive; plausible ECE -0.006 inconclusive; worst-case plausible AUROC inconclusive.
- **Attribution caveat (found while analysing, added to the report generator):** the plausible-tier gain tracks the clean gain. The degradation gap (plausible minus clean AUROC) changed by -0.015 (seed range [-0.071, +0.023], inconclusive; validation -0.006), i.e. `noise` made a better model overall, not a less-degrading one. The trade-off plot points lie along the diagonal. Without that metric the "improved" verdict would have been read as robustness. The metric is descriptive and was added after seeing validation results; it is not part of the pre-declared selection rule, which is unchanged.
- `combined` was not selected: no verdict other than inconclusive (clean ECE -0.169 on test, seeds disagree).
- Across seeds the clean-AUROC difference of a single intervention ranges up to +-0.19: seed noise is as large as any effect here. Test n=30 images / 16 patients cannot resolve effects of this size.
Also fixed: first reports had no way to separate "better model" from "more robust"; the selection rule still allows that confound (candidate for a rule change **before** the first real run, decision pending - see DECISIONS D42).

## Real-data experiments
None yet. Run list will start at **E-001 (EfficientNet-B0, seed 42)** once the real data and pretrained weights are available (see HANDOFF).

## M7 app verification (2026-11-05)
No new experiments. App tests ran the real CLIs (audit, reliability, explain) on the synthetic test workspace via the job runner, then rendered every page with `streamlit.testing.v1.AppTest`; the headless server was also driven with Playwright (9 pages, 0 exceptions). Bugs found and fixed: `metric_bars` expected a `model` key (Compare page KeyError); sidebar path/button text unreadable on navy; temperature dict printed raw; duplicate watermark banner on Grad-CAM tab; training comparison table lacked AUROC (nested dict skipped). All displayed numbers are from watermarked synthetic runs.

## Audit run (2026-11-06)
No new research experiments. `scripts/demo_pipeline.sh` on synthetic data (tiny_cnn, `configs/demo_tiny.yaml`): train/audit/reliability/explain/M6 diagnose-train-validate-final all exit 0; frozen selection = baseline (no intervention met the rule); all outputs NOT A RESULT. Earlier `smoke_tiny.yaml` runs gave val AUROC 0.0-0.1 (random-sign response of an under-trained net, not an orientation bug; see AUDIT_REPORT.md).
