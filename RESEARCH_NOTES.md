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
