"""M5 engine: Grad-CAM + occlusion on selected cases, region-masking and CAM-faithfulness experiments on the whole split.

Case selection is NOT cherry-picked: the most confident errors (all of them up to n_cases) plus a seeded random draw of correct cases
from each class. Quantitative claims come from the whole-split tables, not from the panels.
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from liverguard import __version__
from liverguard.data.report import json_default
from liverguard.explain import masking, plots
from liverguard.explain.gradcam import GradCAM, batched_gradcam, randomization_check
from liverguard.explain.occlusion import occlusion_map
from liverguard.ml.data import build_transform
from liverguard.reliability.failure import label_failures
from liverguard.session import open_session, predict_proba
from liverguard.utils import file_sha256, get_logger

log = get_logger(__name__)

LIMITATIONS = {
    "grad_cam": "Low-resolution, per-image normalised map of one class logit's gradient; not evidence of medical reasoning; can look "
                "plausible for untrained models (see randomisation check); must be read with masking/occlusion results.",
    "occlusion": "Resolution = patch size; grey patch is out-of-distribution; single-patch removal misses redundant evidence; "
                 "describes model behaviour, not anatomy.",
    "masking": "Filled regions are out-of-distribution; region-only AUROC > 0.5 shows information is present, not that it is used "
               "clinically; the centre proxy is NOT the liver or lesion; matched-area random controls are reported alongside.",
    "roi": "No liver/lesion segmentation is generated. ROI = user-supplied annotation masks if given, else a centred geometric proxy.",
    "cam_faithfulness": "Deletion/insertion of the top-CAM region compared with matched random regions; a larger drop than the "
                        "control supports (does not prove) that the map marks regions the model uses.",
    "general": "Research tool only; no clinical validation. Model-behaviour findings do not transfer to other models or datasets.",
}


@dataclass
class ExplainOutputs:
    masking_summary: pd.DataFrame
    cam_faithfulness: pd.DataFrame
    sanity: np.ndarray
    cases: pd.DataFrame
    manifest: dict
    paths: dict[str, Path]


def _tensor(imgs, tf, device) -> torch.Tensor:
    return torch.stack([tf(i) for i in imgs]).to(device)


def _boot_mean(x: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> list[float]:
    if n_boot <= 0 or len(x) < 2:
        return [float("nan")] * 2
    uniq = np.unique(groups); mem = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    v = [x[np.concatenate([mem[g] for g in rng.choice(uniq, len(uniq))])].mean() for _ in range(n_boot)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def _boot_auroc(y, p, groups, n_boot, seed):
    if n_boot <= 0 or len(np.unique(y)) < 2:
        return [float("nan")] * 2
    uniq = np.unique(groups); mem = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed); v = []
    for _ in range(n_boot):
        idx = np.concatenate([mem[g] for g in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[idx])) == 2:
            v.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if v else [float("nan")] * 2


def run_explainability(checkpoint, index_csv, out_dir, *, split: str = "val", variant: str | None = "clean",
                       threshold: float | str = 0.5, device: str = "auto", final_report: bool = False, split_file=None,
                       data_root=None, max_patients: int | None = None, seed: int = 0, mask_dir=None,
                       proxy_area: float = 0.4, cam_fraction: float = 0.2, n_control: int = 3, fill: str = "mean",
                       n_cases: int = 3, occ_patch: int | None = None, occ_stride: int | None = None, n_boot: int = 300,
                       sanity_n: int = 24, batch_size: int = 32, make_plots: bool = True) -> ExplainOutputs:
    out = Path(out_dir)
    S = open_session(checkpoint, index_csv, split=split, variant=variant, threshold=threshold, device=device,
                     final_report=final_report, split_file=split_file, data_root=data_root, max_patients=max_patients, seed=seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tf = build_transform(False)
    imgs, ids, y, groups, size = S.load_images(), S.ids, S.y, S.groups, S.size
    p = predict_proba(S.model, imgs, S.device, batch_size)
    X = _tensor(imgs, tf, S.device)

    # ----- ROI source (annotation if supplied, else geometric proxy)
    masks, src = [], []
    for i in ids:
        m = masking.load_annotation_mask(mask_dir, i, size)
        masks.append(m if m is not None else None)
    if mask_dir is not None:
        have = np.array([m is not None for m in masks])
        if not have.any():
            raise ValueError(f"No annotation masks matching image ids were found in {mask_dir}")
        roi_source = f"annotation masks ({int(have.sum())}/{len(ids)} images have one; others excluded from ROI experiments)"
    else:
        have = np.ones(len(ids), bool)
        masks = [masking.proxy_roi(size, proxy_area)] * len(ids)
        roi_source = f"centre_proxy (centred circle, {proxy_area:.0%} of the image) - NOT the liver or the lesion"

    # ----- Grad-CAM for predicted class on the whole split (needed for faithfulness) + class-specific maps
    cam_pred, probs = batched_gradcam(S.model, X, "predicted", batch_size)
    pred_cls = probs.argmax(1)

    # ----- region experiments: collect p(HCC) per (image, variant)
    rows = []

    def score(variants: dict[str, list]) -> dict[str, float]:
        return {k: float(np.mean(predict_proba(S.model, v, S.device, batch_size))) for k, v in variants.items()}
    for j, im in enumerate(imgs):
        rows.append({"image_id": ids[j], "variant": "full", "p": float(p[j])})
        if have[j]:
            for k, v in score(masking.region_variants(im, masks[j], rng, n_control, fill)).items():
                rows.append({"image_id": ids[j], "variant": k, "p": v})
        for k, v in score(masking.cam_variants(im, cam_pred[j], cam_fraction, rng, n_control, fill)).items():
            rows.append({"image_id": ids[j], "variant": k, "p": v})
    long = pd.DataFrame(rows)
    meta = pd.DataFrame({"image_id": ids, "label": y, "patient_key": groups, "pred_cls": pred_cls, "p_full": p})
    long = long.merge(meta, on="image_id")

    summ = []
    for v, g in long.groupby("variant", sort=False):
        g = g.sort_values("image_id")
        yy, pp, gg = g["label"].to_numpy(), g["p"].to_numpy(), g["patient_key"].to_numpy()
        lo, hi = _boot_auroc(yy, pp, gg, n_boot, seed)
        flips = ((pp >= S.threshold) != (g["p_full"].to_numpy() >= S.threshold)).mean()
        summ.append({"variant": v, "n": len(g), "auroc": roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan,
                     "auroc_lo": lo, "auroc_hi": hi, "accuracy": float(((pp >= S.threshold) == yy).mean()),
                     "mean_p_hcc": float(pp.mean()), "flip_rate_vs_full": float(flips)})
    msum = pd.DataFrame(summ)
    order = ["full", "roi_only", "control_roi_only", "background_only", "control_background_only"]
    msum_region = msum[msum["variant"].isin(order)].set_index("variant").loc[[o for o in order if o in set(msum["variant"])]].reset_index()

    # CAM faithfulness: drop in probability of the originally predicted class
    ev = long.copy()
    ev["p_pred_full"] = np.where(ev["pred_cls"] == 1, ev["p_full"], 1 - ev["p_full"])
    ev["p_pred_var"] = np.where(ev["pred_cls"] == 1, ev["p"], 1 - ev["p"])
    ev["drop"] = ev["p_pred_full"] - ev["p_pred_var"]
    faith = []
    for v in ("cam_delete", "control_cam_delete", "cam_keep", "control_cam_keep"):
        g = ev[ev["variant"] == v].sort_values("image_id")
        d, gg = g["drop"].to_numpy(), g["patient_key"].to_numpy()
        lo, hi = _boot_mean(d, gg, n_boot, seed)
        faith.append({"variant": v, "n": len(g), "mean_evidence_drop": float(d.mean()), "lo": lo, "hi": hi})
    # paired difference: delete (CAM) minus delete (random): positive = the CAM region matters more than a random one
    a = ev[ev.variant == "cam_delete"].sort_values("image_id"); b = ev[ev.variant == "control_cam_delete"].sort_values("image_id")
    diff = a["drop"].to_numpy() - b["drop"].to_numpy()
    lo, hi = _boot_mean(diff, a["patient_key"].to_numpy(), n_boot, seed)
    faith.append({"variant": "paired_difference (cam_delete - control)", "n": len(a), "mean_evidence_drop": float(diff.mean()),
                  "lo": lo, "hi": hi})
    faith_df = pd.DataFrame(faith)

    # ----- ROI localisation of CAM mass: mean CAM inside ROI / mean CAM overall (1 = uniform)
    ratios = [float(cam_pred[j][masks[j]].mean() / cam_pred[j].mean()) if have[j] and cam_pred[j].mean() > 0 else np.nan
              for j in range(len(ids))]

    # ----- model-randomisation sanity check on a seeded subsample
    k = min(sanity_n, len(ids))
    sel = np.sort(rng.choice(len(ids), k, replace=False))
    sanity = randomization_check(S.model, X[sel], pred_cls[sel], seed)

    # ----- cases: confident errors + seeded random correct per class
    base = label_failures(pd.DataFrame({"image_id": ids, "label": y, "p": p, "entropy": 0.0}), S.threshold)
    errs = base[~base["correct"]].sort_values("error_rank").head(n_cases)
    pick = [(i, "confident error") for i in errs.index]
    for cls, name in ((1, "random correct HCC"), (0, "random correct hemangioma")):
        pool = base[base["correct"] & (base["label"] == cls)].index.to_numpy()
        pick += [(i, name) for i in rng.choice(pool, min(n_cases, len(pool)), replace=False)] if len(pool) else []
    cases, case_rows = [], []
    gc = GradCAM(S.model)
    try:
        for j, group in pick:
            x = X[j:j + 1]
            c1, _ = gc(x, 1); c0, _ = gc(x, 0)
            tgt = int(pred_cls[j])
            occ, _ = occlusion_map(S.model, x, tgt, occ_patch or max(size // 7, 4), occ_stride or max(size // 14, 2))
            roi = masks[j] if masks[j] is not None else masking.proxy_roi(size, proxy_area)
            ro, bo = masking.fill_outside(imgs[j], roi, fill), masking.fill_outside(imgs[j], ~roi, fill)
            pr = predict_proba(S.model, [ro, bo], S.device)
            cases.append({"image": imgs[j], "image_id": ids[j], "group": group, "true": "HCC" if y[j] else "Hemangioma",
                          "p": float(p[j]), "pred": "HCC" if p[j] >= S.threshold else "Hemangioma", "cam_hcc": c1[0], "cam_hem": c0[0],
                          "occ": occ, "roi_only": ro, "background_only": bo, "p_roi_only": float(pr[0]), "p_background_only": float(pr[1])})
            case_rows.append({"image_id": ids[j], "group": group, "label": int(y[j]), "p": float(p[j]), "p_roi_only": float(pr[0]),
                              "p_background_only": float(pr[1]), "occlusion_max_drop": float(occ.max())})
    finally:
        gc.close()
    cases_df = pd.DataFrame(case_rows)

    # ----- write
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in (("masking_per_image", long.drop(columns=["label", "patient_key", "pred_cls", "p_full"])),
                        ("masking_summary", msum), ("cam_faithfulness", faith_df), ("cases", cases_df),
                        ("cam_roi_ratio", pd.DataFrame({"image_id": ids, "cam_roi_ratio": ratios})),
                        ("gradcam_randomization_check", pd.DataFrame({"image_id": [ids[i] for i in sel], "spearman_vs_random_model": sanity}))):
        paths[name] = out / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    wm = S.watermark
    if make_plots:
        sub = f"{split} split · n={len(ids)} · ROI: {roi_source.split(' (')[0]}"
        paths["fig_masking"] = plots.plot_masking_summary(msum_region, out / "figures" / "masking_auroc.png", f"Region masking · {sub}", wm)
        paths["fig_cam_faithfulness"] = plots.plot_cam_faithfulness(
            faith_df[~faith_df["variant"].str.startswith("paired")], out / "figures" / "gradcam_deletion_insertion.png",
            f"Grad-CAM deletion / insertion (top {cam_fraction:.0%}) · {sub}", wm)
        paths["fig_sanity"] = plots.plot_sanity(sanity, out / "figures" / "gradcam_randomization_check.png",
                                                f"Grad-CAM sanity check · {sub}", wm)
        if cases:
            paths["fig_cases"] = plots.plot_case_panels(cases, out / "figures" / "case_panels.png",
                                                        f"Explanations on selected cases · {split} split · ROI: {roi_source.split(' (')[0]}", wm)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "valid_as_result": wm is None, "watermark": wm,
        "split": split, "variant": variant, "seed": seed, "threshold": {"value": S.threshold, "source": S.threshold_source},
        "scope": "FINAL REPORT (test split)" if final_report else "DIAGNOSTIC on the validation split",
        "roi_source": roi_source, "settings": {"proxy_area": proxy_area, "cam_fraction": cam_fraction, "n_control": n_control,
                                               "fill": fill, "n_cases": n_cases, "n_boot": n_boot, "sanity_n": int(k)},
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(Path(checkpoint)), "arch": S.meta.get("arch"),
                       "split_id": S.meta.get("split_id")},
        "dataset": {"split_id": S.split_id, "n_images": len(ids), "n_patients": int(len(set(groups))), "subsampled": bool(max_patients)},
        "masking": json.loads(msum.to_json(orient="records")), "cam_faithfulness": json.loads(faith_df.to_json(orient="records")),
        "gradcam_randomization": {"n": int(np.isfinite(sanity).sum()), "median_spearman": float(np.nanmedian(sanity)) if np.isfinite(sanity).any() else None,
                                  "share_above_0.5": float(np.nanmean(sanity > 0.5)) if np.isfinite(sanity).any() else None},
        "limitations": LIMITATIONS,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(S.device), "liverguard": __version__},
    }
    paths["manifest"] = out / "explain_manifest.json"
    manifest["artifacts"] = {k: str(v) for k, v in paths.items()}
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
    return ExplainOutputs(msum, faith_df, sanity, cases_df, manifest, paths)
