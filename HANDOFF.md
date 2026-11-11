# HANDOFF (end of M7 app)

## Read first
`DEVELOPMENT_STATUS.md`, `EXPERIMENT_LOG.md`, `DECISIONS.md` (D11-D47), `RESEARCH_NOTES.md`, `ARCHITECTURE.md`.

## Status
M1-M4, M6, M7 code complete, M5 mostly; 262 tests pass. Run the app: `python scripts/run_app.py --workspace artifacts --index-csv artifacts/dataset/image_index.csv`. **No real data / pretrained weights were available: no real weakness, improvement or trade-off has been measured.** All runs are watermarked synthetic smoke tests.

## Next exact task (needs you, on a machine with internet; GPU strongly preferred)
1. Download SMC-LUD, `verify_dataset.py`, check `image_index.csv`.
2. **Decide D42 before running** (should the selection rule require the degradation gap `plaus_minus_clean_auroc` to improve?). Changing the rule after seeing real validation results is not allowed.
3. `python scripts/robust_train.py all --workdir artifacts/m6 --config configs/baseline_efficientnet_b0.yaml --index-csv artifacts/dataset/image_index.csv --seeds 42,43,44 [--pretrained-weights-path ...]`
   (cost: (1 + planned interventions) x 3 seeds trainings + audits; start with `--epochs 2 --max` sanity if unsure).
4. Read `artifacts/m6/diagnosis/weakness_report.md` and `validation/report.md`; clinician fills the M3 review sheet to confirm plausible-tier label preservation.
5. Only then: `robust_train.py final ... --final-report` (once).
Then: M5 remainder, Docker/MLflow, M8 final documentation.

## Resume
```bash
cd liverguard && python -m pytest -q      # expect 262 passed (~7 min on 2 CPUs)
python scripts/robust_train.py --help
```
Git: M1+M2 committed (9e8ba11); everything since is uncommitted - commit before more work.

## After the independent audit
Read AUDIT_REPORT.md first. Next exact task: obtain real SMC-LUD + weights, run `verify_dataset.py`, adjust aliases, train baselines with `evaluate_test: false`, decide D42. Demo: DEMO_GUIDE.md.
