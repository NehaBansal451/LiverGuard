# RESEARCH_NOTES

Sources read: Figshare record XML (31112716 v3) and Tak et al., *Sci Data* 13:649 (2026). The Figshare page and download were **not reachable** from the build sandbox, so licence, version and layout are verified from those two documents only. No image has been inspected.

## Documented dataset facts
5,385 B-mode 2D images, 1,021 patients, 2015-2024, Samsung Medical Center. HCC 2,716 images / 600 patients (histopathology); Hemangioma 2,669 / 421 (radiological diagnosis, no histology). Clean 2,656 images / 744 patients; Caliper 2,729 images. Official 70/20/10 patient-level split documented for **Clean only**: 1,858 / 530 / 268. Metadata CSV: patient_id, image_filename, class_label (+ HCC stage and tumour size).

## Documentation problems found (also in `dataset.yaml`)
5,466 vs 5,385 images · 422 vs 421 hemangioma patients · HCC height 163.57 vs 165.74 cm · CC-BY label vs non-commercial restriction · "single image per HCC case" vs ~4.5 images/patient · DOI 31112716 vs 29755535 · Caliper split undocumented · ~654 patients have both variants.

## Scientific risks to test, not assume
1. **Spectrum / selection bias.** Images are curated "typical" examples read by radiologists; hemangioma labels come from imaging, HCC from pathology. The published 98.88% (3 errors in 268 images) is likely an optimistic upper bound for a clean two-class problem, not field performance.
2. **Shortcut candidates.** Cohorts differ strongly in age (66.7 vs 55.5 y) and sex (81.8% vs 41.3% male). HCC usually arises in a cirrhotic liver, so background parenchyma, burned-in text/scales, probe settings and Caliper marks could predict class. M5 must include background-only, lesion-masked and caliper-removed controls.
3. **Clustered data.** Several images per patient: image-level CIs are too narrow. Use patient-level cluster bootstrap; the test split is small (about 74 patients if the 10% fraction applies to patients; unverified).
4. **Single run, single institution.** The paper reports no seeds or variance (ResNet-50 92.16% vs proposed 98.88%). Use multiple seeds; no device/acquisition metadata is documented, so device-level shift cannot be assessed.
5. **Caliper variant.** Caliper share differs slightly by class (HCC 48.9%, Hemangioma 52.5%); both variants share patients (D8).
6. **Label preservation under distortion** is an assumption to check in M3 (mild plausible changes vs extreme stress tests).

## M3 limitations (robustness audit)
- Distortions are synthetic and applied after resizing; they approximate, not reproduce, scanner/acquisition variation. Speckle is multiplicative Gaussian, not a physical speckle model.
- Plausible (L1-3) vs stress (L4-5) tiers are a-priori. Label preservation is unverified until a clinician completes the review sheet; PSNR/correlation/std/saturation only screen. Rotation and crop can remove the lesion and are invisible to pixel metrics.
- Flip rate depends on the threshold (use a validation-fitted one, report both); AUROC change is threshold-free. Confidence = max(p, 1-p) is model confidence, not clinical certainty.
- Audit on the validation split guides M6; final numbers belong to the test split, run once with `--final-report`.

## M4/M5 method limitations (also embedded in every manifest)
| method | what it measures | main limitations |
|---|---|---|
| Grad-CAM | gradient-weighted last-conv activation for one class logit | low resolution (7x7 at 224 px), per-image min-max normalised, can resemble image structure even for untrained nets (we report a randomisation check), class maps are near-complements, not evidence of medical reasoning |
| Occlusion | drop in class probability when a patch is replaced by the image mean | resolution = patch size, grey patch is out-of-distribution, single patches miss redundant evidence, costly (selected cases only) |
| ROI / background masking | prediction on ROI-only vs background-only images, with matched-area random controls | fill creates OOD inputs (keep-only variants can collapse to one class); AUROC > 0.5 on a region shows information present, not used clinically; the centre proxy is NOT the liver or lesion; no segmentation is generated, annotations must be user-supplied |
| CAM deletion/insertion | evidence drop when the top-CAM region is removed/kept vs a random region of equal area | same OOD caveat; supports, does not prove, that the map marks used regions |
| Predictive entropy / confidence | uncertainty of the single forward pass | rank-equivalent for binary output; overconfident when miscalibrated; not clinical certainty |
| MC-Dropout | variation under head dropout | head only, so understates epistemic uncertainty |
| TTA spread | stability under flip, +-5 deg, 5% zoom | probe of geometric sensitivity; flip alters apparent organ side |
| ECE | top-label calibration gap | bin-dependent, biased at small n (both width/mass reported); can be low for a model with no discrimination, so read with AUROC and Brier |
| Brier / NLL | proper scoring rules | mix calibration and discrimination; NLL dominated by confident errors |
| Temperature scaling | one global logit rescale fitted on val | cannot fix subgroup miscalibration, does not change AUROC, cross-fitted on val to avoid optimism |
| Risk-coverage | error of retained cases when abstaining on the most uncertain | noisy at low coverage on small splits; abstention thresholds must come from validation |
| Failure cases | most-wrong errors, confident errors, patient-level disagreement | screening for human review, not causal explanation; few errors on small splits |
Open question for real data: because fill-based probes are confounded by out-of-distribution inputs, treat masking results as one line of evidence next to occlusion, calibration and the Clean-vs-Caliper analysis (M5 remainder).
