# DEMONSTRATION GUIDE

A 15-minute walkthrough of LiverGuard on the **synthetic** demo workspace. State this up front when presenting: *the data is synthetic, the models are tiny, nothing shown is a finding; the demo shows that the audit machinery works, not what real ultrasound models do.*

## Prepare (3-4 min, once)
```bash
bash scripts/demo_pipeline.sh demo_workspace
python scripts/robust_train.py final --workdir demo_workspace/m6 --config configs/demo_tiny.yaml \
    --index-csv demo_workspace/dataset/image_index.csv --seeds 42,43,44 --n-boot 50 --final-report   # optional, ~1.5 min; enables the Dashboard's baseline-vs-robust view
python scripts/run_app.py --workspace demo_workspace --index-csv demo_workspace/dataset/image_index.csv
```
Open http://localhost:8501. The sidebar shows the run counts; every run carries a red NOT A RESULT banner.

## Tour (follow the BUILD → ATTACK → DIAGNOSE → MEASURE → FIX → RE-TEST cycle)
1. **Research Dashboard**: pipeline status chips read "pipeline check only" (they would read "done" only for valid results). Point out that the status is derived from manifests, not asserted.
2. **Dataset Explorer**: class/patient counts, split table (patient-level), browse images, Clean vs Caliper tab, Verify tab (starts the real verification as a background job; paste the synthetic root `demo_workspace/raw`).
3. **Model Laboratory** (BUILD): training curves and confusion matrix of the demo run; *Train* tab starts a real job (choose `tiny_cnn`, un-tick pretrained, image size 64, 1-2 epochs). Note "Evaluate the TEST split" is off by default; run names cannot overwrite a finished run.
4. **Image Audit Studio** (ATTACK): *Single image*: pick a validation image, distortion `gaussian_noise`, raise the severity; watch original / distorted / difference, the probability and flip warning, the fidelity expander, and the severity sweep. *Dataset-level audit*: degradation curves, "Most disruptive" ranking, heat maps, fidelity-flagged conditions. Explain plausible vs stress tiers.
5. **Explainability Studio** (DIAGNOSE): Grad-CAM for predicted/HCC/Hemangioma, occlusion map, ROI vs background with matched-area controls. Say plainly: the ROI is a centre proxy, not the liver; the maps are heuristics. *Dataset-level* tab: the randomisation check.
6. **Reliability Center** (MEASURE): reliability diagram, ECE/Brier/NLL, risk-coverage curve, most-uncertain images, failure cases. Show the banner when the fitted temperature hits its search bound (near-chance model).
7. **Robust Training Lab** (FIX): weaknesses and plan from the diagnosis, stage buttons, the final stage locked until a selection exists and a confirmation box is ticked. Validation results: paired differences vs baseline.
8. **Model Comparison** (RE-TEST): two checkpoints on the same split; ablation tab with clean vs plausible vs stress metrics. In the demo the frozen selection is *baseline* (no intervention beat it): a correct outcome of the pre-declared rule, worth showing.
9. **Research Reports**: filter history, pick runs, *Generate PDF*, download. Open the PDF: page 1 states the runs are not valid results.

## Things to demonstrate on purpose
- Select **test** in the Reliability or Audit launchers: the start button stays disabled until "Final report" is ticked.
- Training a run with an existing name, or `../x`: refused with a message.
- Point the app at an empty folder (`LIVERGUARD_WORKSPACE=/tmp/empty`): every page shows an empty state, no fake numbers.

## What this demo cannot show
Real-data behaviour, pretrained EfficientNet/ResNet performance, clinically meaningful heat maps, or any claim about robustness of real models.
