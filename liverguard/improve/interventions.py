"""Named robust-training interventions = groups of M3 audit conditions used as TRAINING augmentation.

Design rules
- Augmentation draws a fresh random severity in the PLAUSIBLE tier only (continuous, between the level-1 and the
  level-`plausible_max_level` parameter of the audit config). Stress levels are never trained on, so stress-tier results
  measure extrapolation.
- Training randomness is keyed by 'train/<path>' and a per-process generator, never by the audit's (seed, image_id) keys, so
  training never sees the audit's exact noise realisations.
- CAVEAT (in every report): training on a corruption family and then auditing the same family measures adaptation to a KNOWN
  corruption. Conditions outside an intervention's family measure transfer; compare.py reports both separately.
- ROI-aware training needs annotation masks that do not exist for SMC-LUD; it is deliberately not offered.
"""
from __future__ import annotations

INTERVENTIONS: dict[str, list[str]] = {
    "baseline": [],
    "brightness": ["brightness_dark", "brightness_bright"],
    "contrast": ["contrast_low", "contrast_high"],
    "gamma": ["gamma_low", "gamma_high"],
    "photometric": ["brightness_dark", "brightness_bright", "contrast_low", "contrast_high", "gamma_low", "gamma_high"],
    "noise": ["gaussian_noise", "speckle_noise"],
    "blur": ["gaussian_blur"],
    "jpeg": ["jpeg"],
    "geometric": ["rotation", "center_crop"],
    "combined": ["brightness_dark", "brightness_bright", "contrast_low", "contrast_high", "gamma_low", "gamma_high",
                 "gaussian_noise", "speckle_noise", "gaussian_blur", "jpeg", "rotation", "center_crop"],
}
# smallest single-family intervention that contains the condition (used to turn a diagnosed weakness into an experiment)
_TARGET = {"brightness_dark": "brightness", "brightness_bright": "brightness", "contrast_low": "contrast",
           "contrast_high": "contrast", "gamma_low": "gamma", "gamma_high": "gamma", "gaussian_noise": "noise",
           "speckle_noise": "noise", "gaussian_blur": "blur", "jpeg": "jpeg", "rotation": "geometric",
           "center_crop": "geometric"}


def conditions_of(name: str) -> list[str]:
    if name not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention {name!r}; known: {sorted(INTERVENTIONS)}")
    return INTERVENTIONS[name]


def intervention_for(condition: str) -> str | None:
    return _TARGET.get(condition)
