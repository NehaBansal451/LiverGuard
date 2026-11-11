# DECISIONS

| ID | Decision | Reason |
|----|----------|--------|
| D1 | Positive class = HCC (label 1); Hemangioma = 0 | Malignant class drives sensitivity/AUROC; set in `dataset.yaml`, not in code |
| D2 | Dataset facts live in `configs/dataset.yaml`; no hardcoded paths/labels | Requirement + the published docs disagree with each other, so values must be editable |
| D3 | Indexer is layout-tolerant (path tokens + CSV overlay; CSV wins, conflicts reported) | Figshare and paper describe different folder layouts; real layout unverified |
| D4 | Published counts are compared, not trusted | 8 documentation inconsistencies already found (see dataset.yaml) |
| D5 | Official split is preserved; rebuilt split = separate file, `StratifiedGroupKFold` (10 folds) grouped by patient | No-leakage rule; never silently replace the official split |
| D6 | Follow non-commercial research-only terms despite the CC-BY label | Figshare note + paper Data availability; the stricter statement governs |
| D7 | Dataset never enters git, Docker images or public hosting; hosted demo must use user-supplied or synthetic data | Licence + patient privacy |
| D8 | Clean-vs-Caliper domain-shift tests may only use Caliper images of patients whose Clean images are in the same split (or a patient-complete split) | ~654 patients have both variants; pooling would leak |
| D9 | Perceptual hash = own 64-bit dHash (Pillow+NumPy) | Avoids an extra dependency |
| D10 | M1 pins no ML libraries; `requirements-ml.txt` is unpinned until M2 installs and tests it | Do not claim compatibility that was not tested |
| D11 | M2 reuses the earlier M2 draft (`data.py`, `trainer.py`, `evaluate.py`, `plots.py`, `scripts/train.py`); `config.py`, `models.py`, `metrics.py` were **reconstructed** from the interfaces those files require | The originals of these three files (and `__init__.py`) were overwritten by mistake during M2 before the draft was noticed; no VCS history existed. Behaviour is covered by tests, but any intent not encoded in the surviving files is lost |
| D12 | 2-class softmax head (index 1 = HCC), cross-entropy with train-only inverse-frequency class weights | Matches the earlier draft; p(HCC) = softmax[:, 1] |
| D13 | Pretrained weights must load or the run fails; `pretrained_weights_path` supports offline weights; no silent random fallback | A from-scratch model must never be reported as "transfer learning" |
| D14 | Baseline augmentation = horizontal flip + rotation <= 10 deg only. Photometric/noise/blur/JPEG augmentation reserved for M6 | Keeps the M6 ablation interpretable. Caveat: flips change apparent organ laterality; revisit if flip proves harmful |
| D15 | Checkpoint = best `val_auroc`; ties broken by lower validation loss | A first-to-hit-1.000 checkpoint was far worse calibrated than a later one |
| D16 | Two reported operating points: fixed config threshold (0.5) and Youden threshold fitted on **validation only** | A rank-perfect, miscalibrated model can show sensitivity 0 at 0.5. Neither is tuned on test |
| D17 | Test split decoded only after training finishes and the best checkpoint is restored (enforced by a test) | Prevents test-set peeking and accidental model selection on test |
| D18 | Runs that are smoke tests or null controls are watermarked and flagged `valid_as_result: false` in the manifest | Prevents synthetic or control numbers being mistaken for findings |
| D19 | Patient-level cluster bootstrap for all CIs | Several images per patient are correlated; image-level CIs are too narrow |
| D20 | M3 reuses the earlier M3 draft (`config`, `distortions`, `fidelity`, `robustness.yaml`, distortion tests); inspected before editing; nothing overwritten | Lesson from D11. Added: measures, engine, plots, review, CLI, engine tests |
| D21 | Distortions applied at model-input resolution after resize, in float64; noise greyscale-shared across channels | What the model sees; avoids float32 tie-rounding artefacts |
| D22 | Per-image RNG = SHA-256(seed, image_id, condition), independent of level/batch/order | Reproducible and order-free; levels share random draws (common random numbers), so severity curves are nested and comparable |
| D23 | Audit defaults to the VALIDATION split; TEST requires `--final-report`; checkpoint `split_id` must equal the index split_id | Diagnosis feeds M6 choices, so it must not touch test (no tuning on test) |
| D24 | Tiers: levels <= `plausible_max_level` (3) = plausible, above = stress test; a-priori, not radiologist-validated. Label preservation is an assumption; fidelity proxies only screen, expert sheet is the check | Brief requires distinguishing plausible from extreme |
| D25 | Primary ranking = most negative mean dAUROC over levels; secondary = mean flip rate; PFR, confidence drift (signed, max(p,1-p)), MFS (censored at L+1, median label ">L") reported alongside | AUROC is threshold-free; flip rate captures changes AUROC hides |
| D26 | Paired patient-level cluster bootstrap for deltas (clean and distorted on the same resample) | Images of one patient are correlated; pairing narrows CIs correctly |
| D27 | Fidelity proxies skip PSNR/correlation for geometric families (rotation, crop) | Unaligned pixel metrics gave false "destroyed" flags |
| D28 | Audit plots use a validated colour-blind-safe categorical palette with marker/line-style redundancy rather than the navy/teal/lavender brand palette | Brand colours failed the distinguishability check; brand palette stays for the M7 UI chrome |
| D29 | Speckle = simplified multiplicative Gaussian, not a physical ultrasound speckle model | Documented limitation; revisit if speckle proves decisive on real data |
| D30 | M4/M5 reuse the earlier drafts found in `liverguard/reliability`, `liverguard/explain`, `liverguard/session.py`, `liverguard/plotstyle.py`; reviewed, bugs fixed, tests added | D11 lesson: inspect before writing |
| D31 | Shared `session.open_session` for post-training analyses: same split guards (test needs `final_report`, split_id must match, patient subsample, watermark) | One place enforces the no-test-tuning rule |
| D32 | Calibration: ECE top-label, equal-width and equal-mass both reported with bin counts; Brier + NLL; patient-level bootstrap CIs | ECE is bin-dependent and small-n biased |
| D33 | Temperature scaling fitted on VALIDATION only; on the val split the headline calibrated numbers are patient-fold cross-fitted (in-sample row labelled optimistic); on test T is fitted on val and applied unchanged | Scoring T on the data it was fitted on flatters it |
| D34 | Entropy and confidence are rank-equivalent for a binary model; risk-coverage therefore uses margin to the operating threshold, MC-Dropout std/MI and TTA std | Avoids presenting duplicate curves as independent evidence |
| D35 | MC-Dropout samples head dropout only (fast path verified equal to full passes); TTA includes horizontal flip | Documented limitation: not full epistemic uncertainty; flip changes apparent organ side |
| D36 | ROI = user-supplied annotation masks (found by relative image path) else a centred geometric proxy; no automatic liver/lesion segmentation is produced. Every region experiment has a matched-area random-translation control | No pretending a proxy is clinical; separates "hole in image" from region content |
| D37 | Grad-CAM implemented with hooks (no extra dependency), with a model-randomisation sanity check and CAM deletion/insertion faithfulness vs matched random regions; cases shown = most-wrong errors + seeded random correct per class (no cherry-picking) | Heatmaps alone are not evidence of reasoning |


| D38 | M6 reuses the earlier drafts in `liverguard/improve/`, `scripts/robust_train.py`, `tests/test_improve.py` and the small `ml/` hooks (`robust` augmentation, `evaluate_test`); reviewed, extended, tested | D11 lesson |
| D39 | Robust augmentation draws severities only inside the plausible tier, using the audit's own distortion functions with a training-only RNG; stress levels are never trained on | Stress results then measure extrapolation; no overlap with the audit's noise realisations |
| D40 | Protocol enforced in code: test not evaluated at training time; selection written with checkpoint hashes before any test audit; `final` needs `--final-report` and an intact selection; all models re-audited with identical code/threshold (`val_youden`) | No tuning on test |
| D41 | Weakness = plausible-tier level, not fidelity-flagged, with AUROC drop >= 0.02 (CI excludes 0) or flip-rate lower CI >= 0.05, in a majority of baseline seeds; a negative-control intervention is added when possible | Stress failures and possibly label-destroying changes never drive training |
| D42 | Added descriptive metric `plaus_minus_clean_auroc` and an automatic attribution caveat in reports. Selection rule left unchanged for this run; **open decision**: before the first real run, consider requiring this gap to improve (not only plausible AUROC) | Plausible-tier AUROC gains can come from a better clean model |
| D43 | App reads ONLY manifests/CSVs written by the ML modules; the SQLAlchemy registry (`<workspace>/liverguard.db`) indexes them (rescan is idempotent) and copies headline numbers, never computes metrics | Single source of truth; a UI number can always be traced to a file on disk |
| D44 | Long work (train/audit/reliability/explain/M6) runs as background subprocess CLIs with log files and an exit-code file written by the shell wrapper; Cancel kills the process group | Survives Streamlit reruns; `waitpid` was unreliable across reruns |
| D45 | `app/ops.py` is pure (no Streamlit) and `app/state.py` holds caching/glue; charts are pure DataFrame->Figure functions | Testable without a browser |
| D46 | Chart colours use the validated categorical palette with marker/dash redundancy; brand navy/teal/lavender only for chrome. Non-valid runs get a banner and a "NOT A RESULT" chart overlay on every page | Colour-blind safe; synthetic output can never look like a finding |
| D47 | Test-split analyses (audit/reliability/explain/final) are disabled in the UI until the final-report box is ticked; robust-training "final" is locked until `selection.json` exists | UI mirrors the CLI protocol guards (checked by AppTest) |
| D48 | The app's Model Laboratory leaves test evaluation OFF by default; run names are validated and finished runs are never overwritten | Found in audit: default evaluation of test on every experiment invites tuning on test; names were used as paths |
