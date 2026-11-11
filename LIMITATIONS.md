# LIMITATIONS

LiverGuard is a research / model-auditing platform. It is **not** a medical device, has **no clinical validation**, and must not inform patient care.

## Evidence status
- **No real-data result exists.** SMC-LUD and ImageNet weights were unreachable during development. Every metric produced so far comes from synthetic images (blobs with a planted class signal that any CNN separates easily, e.g. validation AUROC 1.0) and is watermarked NOT A RESULT. Nothing here estimates how any model behaves on real ultrasound.
- Only the `tiny_cnn` smoke architecture has been run through the whole pipeline. EfficientNet-B0 / ResNet-50 construction and short smoke training were exercised, not a real training run.
- The data-layout assumptions (folder tokens, split/variant aliases) were validated against a synthetic tree built from the dataset paper's description, not the real download.

## Scientific limitations (see RESEARCH_NOTES.md for the full table)
- **Dataset**: single centre; curated "typical" images; hemangioma labelled radiologically, HCC pathologically (label-source bias); strong age/sex differences between classes; published figures (98.88%) are likely optimistic. Clean and Caliper images share patients, so the variants are not independent.
- **Distortions are synthetic** approximations of acquisition change. Label preservation is an *assumption*: the plausible/stress tiers are a-priori, fidelity proxies only screen, and the expert review sheet has not been completed by a clinician. Degradation under a flagged condition is not evidence of fragility.
- **Explainability**: Grad-CAM, occlusion and masking are heuristics; none shows what a clinician would use. The default ROI is a centred geometric proxy, **not the liver or lesion**. No segmentation masks are shipped.
- **Uncertainty**: MC-Dropout perturbs only the head (understates epistemic uncertainty); TTA measures sensitivity to chosen augmentations; ECE is bin-dependent and noisy; confidence is not clinical certainty. A temperature at the edge of its search range signals a near-chance model.
- **Robust training (M6)**: the improvement metric on plausible-tier conditions is *descriptive* because training and testing use the same distortion family; the selection rule's treatment of the degradation gap is an open decision (D42). Few seeds on one split are not independent samples of a population.
- **Statistics**: CIs are patient-level cluster bootstraps; the test split is small, so intervals will be wide.

## Engineering limitations
- **Test-set hygiene depends on discipline**: baseline YAMLs set `evaluate_test: true`, so a CLI baseline run reports test metrics immediately. The CLIs and the app lock the dedicated test analyses behind `--final-report` / a checkbox, and the Model Laboratory defaults test evaluation OFF, but nothing prevents someone from tuning on test numbers they have already seen. Use `evaluate_test: false` while experimenting.
- Exact-duplicate checks across splits are only enforced by `verify_dataset.py --hash`, not by training.
- Frozen-backbone epochs still update BatchNorm statistics (the model stays in train mode).
- The app is a single-user local tool: no authentication, no multi-user concurrency tests, SQLite only. Do not expose it publicly with the dataset mounted.
- Image browsing / statistics read files from disk on demand; very large datasets and narrow/mobile layouts were not tested.
- Not built: Clean-vs-Caliper domain-shift study, caliper-removal control, Integrated Gradients, ROI-aware training, OOD detection, cross-dataset generalisation, MLflow, DVC, FastAPI, PostgreSQL, a verified Docker image.
