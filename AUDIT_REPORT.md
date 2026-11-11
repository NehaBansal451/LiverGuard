# AUDIT REPORT (independent QA / ML review)

Scope: whole repository after Milestone 7. Method: code reading of the high-risk paths (splits, data loading, training, metrics, calibration, robustness measures, session guards, registry/jobs), static analysis (ruff), the full test suite, a fresh end-to-end CLI run on synthetic data, UI workflow tests (`streamlit.testing`), and a headless-server render of the pages with screenshots. Environment: Linux, Python 3.13, CPU. **No real SMC-LUD data and no pretrained weights were available, so nothing here speaks to real-data behaviour.**

## Verification status
| Area | How it was checked | Status |
|---|---|---|
| Unit/integration suite | `python -m pytest tests` | **262 passed** (full run, exit 0) |
| Pipeline end to end | `scripts/demo_pipeline.sh` (verify → train → audit → reliability → explain → robust diagnose/train/validate) then `robust_train.py final --final-report` | ran to completion, exit 0, synthetic data only |
| All 9 pages | AppTest on a test workspace and on the demo workspace; headless Streamlit + Playwright: 0 exceptions on all pages | render OK |
| Pages visually inspected | Dashboard, Reliability, Image Audit (single + dataset-level), Model Comparison, Explainability (top), sidebar | looked at |
| Pages NOT visually inspected | Dataset Explorer, Model Laboratory, Robust Training Lab, Research Reports (render + interaction tested, appearance not reviewed); narrow/mobile layouts | unverified |
| UI workflows exercised | training launched from the UI and indexed; dataset verification from the UI; PDF generation; Grad-CAM / occlusion / ROI buttons; severity slider; test-split locks; invalid/duplicate run names; empty workspace | tested |
| Real EfficientNet-B0 / ResNet-50 training, pretrained weights, real data | – | **not done** |
| Docker / docker-compose | files written; no daemon available | **untested** |
| GPU, multi-user, large datasets, mobile layout | – | **untested** |

## Defects found and fixed in this review
| # | Severity | Finding | Fix | Regression test |
|---|---|---|---|---|
| 1 | High (workflow) | `scripts/train.py` crashed with `KeyError: 'test'` *after* a successful run whenever the test split was not evaluated (job reported failed). Found by the new UI end-to-end test. | Script handles both cases and prints the validation line plus a "test not evaluated" note | `test_train_from_the_ui_end_to_end` |
| 2 | High (leakage risk) | Model Laboratory trained with `evaluate_test=True` by default, so every UI experiment revealed test metrics (invites tuning on test). | Checkbox, default OFF, with explanation; manifest records the policy | same test asserts the default and the manifest |
| 3 | High (incomplete feature) | The dataset-level robustness audit had **no UI**: three chart builders (degradation curves, heat map, ranking) were unused and no page launched `audit_robustness.py`. | New "Dataset-level audit" tab (viewer + launcher with the test-split lock) | `test_dataset_level_audit_tab_shows_real_curves_and_guards_test` |
| 4 | Medium (DB) | Registry kept "ghost" runs after their manifests were deleted | `scan` prunes rows under the scanned root whose manifest no longer exists | `test_registry_prunes_runs_whose_manifest_was_deleted` |
| 5 | Medium (DB/jobs) | Cancelling an already-finished job recorded it as cancelled | cancel checks the exit-code file first | `test_cancelling_an_already_finished_job_keeps_its_real_outcome` |
| 6 | Medium (security/robustness) | Run names became paths: `../x` could escape the workspace; an existing run could be overwritten | name validation + refusal to overwrite a finished run | `test_run_names_cannot_escape_the_workspace`, UI tests |
| 7 | Medium (UI) | Audit caption said "sub-sampled patients" for *complete* audits (read a dict's truthiness) | `subsample_note()` reads the `complete` flag | `test_registry_reads_training_seed_and_audit_subsample_flag` |
| 8 | Low (UI) | Checkpoint labels showed "seed ?" (seed is under `config.seed`) | registry reads config seed | same test |
| 9 | Medium (misinterpretation) | Image pickers allowed auditing a TRAIN image, or an image under a different split definition than the model was trained on, without warning | banners for both cases | covered by page-render tests (banner logic not separately asserted) |
| 10 | Low (misinterpretation) | A calibration temperature pinned at the search bound (near-chance model) was shown as a normal "calibrated" result | warning banner | not separately asserted |
| 11 | Medium (deployment) | `requirements.txt` listed only Milestone-1 packages (no torch/streamlit/plotly/sqlalchemy/reportlab) | consolidated; `pip install --dry-run` resolves | dry-run only |
| 12 | Low | Unused imports/variables | ruff F401 auto-fixed, two dead variables removed; 4 cosmetic lint notes remain | suite |

## Investigated and found NOT to be a bug
- **Validation AUROC of 0.00–0.10 in the smoke runs.** Suspected inverted label/probability orientation. Evidence against: the AUROC flips between ≈0 and 1.0 across seeds with the training loss stuck at chance (an under-trained net with a random-sign monotone response to the planted signal), a properly trained run reaches 1.0, and the engine tests assert clean AUROC > 0.9 on the same synthetic task. Resolution: added `configs/demo_tiny.yaml` (longer, higher LR) for the demo; `smoke_tiny.yaml` is only a minimal pipeline check.
- Patient-level split guards, AUROC/sensitivity/specificity, Youden threshold fitting (val only), ECE/Brier/NLL, temperature scaling (cross-fitted on val), flip rate / MFS / confidence drift and the paired patient bootstrap were read line by line; no defect found. This is a code review plus the existing unit tests, **not** an independent numerical re-derivation of every metric.

## Open issues (not fixed)
1. **D42**: should the M6 selection rule also require the degradation gap to improve? Needs a decision before any real run.
2. `TrainConfig`/baseline YAMLs still default to `evaluate_test: true` (kept because the milestone asked for a clean-test baseline evaluation). Recommend `false` until the final report.
3. Exact-duplicate cross-split images are only detected by `verify_dataset.py --hash`, not enforced at training.
4. BatchNorm statistics update during frozen-backbone epochs (model kept in train mode).
5. Docker files untested; no CI configuration; type hints are not checked by mypy.
6. Database is a thin index (3 tables); the richer schema in the original plan (datasets, images, metrics tables) was not built.
7. Features in the master plan not implemented: see `LIMITATIONS.md`.

## Reproduce this audit
```bash
python -m pytest tests -q                       # 262 passed
bash scripts/demo_pipeline.sh /tmp/demo_ws      # exit 0
ruff check --select F,B,E9,PLE --ignore E501,F811,B008,B905 liverguard scripts   # 4 cosmetic notes
```
