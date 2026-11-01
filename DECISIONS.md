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

