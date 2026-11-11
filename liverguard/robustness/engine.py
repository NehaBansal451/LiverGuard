"""Robustness audit engine: distort -> predict -> compare with clean -> degradation tables, plots, manifest.

Guarantees
- Original images are never modified; every distorted copy is a new object.
- Fully reproducible: noise / rotation sign depend only on (seed, image_id, condition), not on batching, order, or which other
  conditions are run, so a subset audit reproduces the same numbers as a full audit.
- Clean predictions use exactly the M2 evaluation preprocessing (resize -> ToTensor -> Normalize), so the "clean" level equals the
  training run's own predictions (checked by a test).
- Split discipline: by default the audit runs on the VALIDATION split (diagnosis, used to design M6 interventions). The TEST split
  is only audited with an explicit final_report=True, because choosing interventions from a test-set audit would tune on test.
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

from liverguard import __version__
from liverguard.data.report import json_default
from liverguard.data.splits import split_manifest_id
from liverguard.ml.data import build_transform, load_resized, load_split_frame
from liverguard.ml.models import load_checkpoint
from liverguard.ml.trainer import resolve_device
from liverguard.robustness import measures, plots, review
from liverguard.robustness.config import RobustnessConfig
from liverguard.robustness.distortions import apply_distortion
from liverguard.robustness.fidelity import GEOMETRIC_FAMILIES, at_risk, fidelity_metrics
from liverguard.utils import file_sha256, get_logger

log = get_logger(__name__)

ASSUMPTIONS = [
    "Severity tiers ('plausible' vs 'stress') are a-priori judgements, not radiologist-validated.",
    "Label preservation under distortion is ASSUMED, not shown. Fidelity flags are heuristic proxies; confirm with expert review "
    "(review/label_preservation_review.csv + gallery).",
    "Distortions are applied at model-input resolution after resizing; JPEG/blur/noise at native resolution would differ.",
    "Speckle is modelled as simple multiplicative Gaussian noise, not true ultrasound speckle statistics.",
    "Flip rate and accuracy/F1 depend on the decision threshold; AUROC change and mean |dp| are threshold-free.",
    "Confidence = probability of the predicted class; it is not clinical certainty.",
]


@dataclass
class AuditOutputs:
    tables: measures.AuditTables
    per_image: pd.DataFrame
    manifest: dict
    paths: dict[str, Path]


def resolve_threshold(threshold: float | str, checkpoint: Path) -> tuple[float, str]:
    """Number -> fixed. 'val_youden' -> the validation-fitted threshold stored in the training run's manifest.json."""
    if isinstance(threshold, str) and threshold != "val_youden":
        threshold = float(threshold)
    if isinstance(threshold, (int, float)):
        return float(threshold), "fixed"
    mf = checkpoint.parent / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"threshold='val_youden' needs {mf} (the training run's manifest)")
    value = json.loads(mf.read_text())["results"]["val_youden_threshold"]
    return float(value), f"val_youden from {mf.name}"


def subsample_patients(part: pd.DataFrame, max_patients: int, seed: int) -> pd.DataFrame:
    """Seeded, class-proportional patient subsample (whole patients only). For quick iterations, NOT final reporting."""
    rng = np.random.default_rng(seed)
    pat = part.drop_duplicates("patient_key")[["patient_key", "label"]]
    keep: list[str] = []
    for _, g in pat.groupby("label"):
        keys = sorted(g["patient_key"])
        rng.shuffle(keys)
        keep += keys[:max(1, round(max_patients * len(g) / len(pat)))]
    return part[part["patient_key"].isin(keep)].reset_index(drop=True)


@torch.inference_mode()
def _predict(model, images, tf, device, batch_size: int) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.stack([tf(im) for im in images[i:i + batch_size]]).to(device)
        out.append(torch.softmax(model(x).float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def run_robustness_audit(checkpoint: str | Path, index_csv: str | Path, out_dir: str | Path, *,
                         cfg: RobustnessConfig, split: str = "val", variant: str | None = "clean",
                         threshold: float | str = 0.5, batch_size: int = 64, device: str = "auto",
                         final_report: bool = False, split_file: str | Path | None = None,
                         data_root: str | Path | None = None, max_patients: int | None = None,
                         review_n: int = 3, make_plots: bool = True) -> AuditOutputs:
    checkpoint, out = Path(checkpoint), Path(out_dir)
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    if split == "test" and not final_report:
        raise ValueError("Auditing the TEST split requires final_report=True (--final-report). Use the validation split to "
                         "diagnose weaknesses and choose interventions; the test split is for the final report only.")
    dev = resolve_device(device)
    torch.manual_seed(cfg.seed)
    model, meta = load_checkpoint(checkpoint, dev)
    thr, thr_source = resolve_threshold(threshold, checkpoint)

    frame = load_split_frame(index_csv, split_file=split_file, variant=variant, data_root=data_root)
    sid = split_manifest_id(frame)
    if meta.get("split_id") and meta["split_id"] != sid:
        raise ValueError(f"Checkpoint was trained on split_id {meta['split_id']} but this index/variant gives {sid}. Auditing "
                         "on a different split risks evaluating on training images; refusing.")
    part = frame[frame["split"] == split].reset_index(drop=True)
    if max_patients:
        part = subsample_patients(part, max_patients, cfg.seed)
    if part["label"].nunique() < 2:
        log.warning("Audit split has a single class: AUROC-based measures will be NaN")
    ids, y = part["image_id"].tolist(), part["label"].to_numpy(dtype=int)
    size = int(meta["image_size"])
    originals = [load_resized(p, size) for p in part["path"]]          # never modified below
    tf = build_transform(False)
    log.info("Auditing %d images / %d patients, split=%s, threshold=%.3f (%s)", len(part), part["patient_key"].nunique(),
             split, thr, thr_source)

    p_clean = _predict(model, originals, tf, dev, batch_size)
    P: dict[str, np.ndarray] = {}
    FID: dict[str, dict] = {}
    for c in cfg.conditions:
        L = len(c.params)
        P[c.name] = np.zeros((L, len(ids)))
        FID[c.name] = {k: np.zeros((L, len(ids))) for k in ("psnr_db", "correlation", "std_ratio", "new_saturated_fraction")}
        FID[c.name]["at_risk"] = np.zeros((L, len(ids)), dtype=bool)
        for k in range(1, L + 1):
            dist = [apply_distortion(im, c.func, c.params[k - 1], seed=cfg.seed, image_id=i, condition=c.name)
                    for im, i in zip(originals, ids)]
            P[c.name][k - 1] = _predict(model, dist, tf, dev, batch_size)
            for j, (o, d) in enumerate(zip(originals, dist)):
                fm = fidelity_metrics(o, d)
                for key in ("psnr_db", "correlation", "std_ratio", "new_saturated_fraction"):
                    FID[c.name][key][k - 1, j] = fm[key]
                FID[c.name]["at_risk"][k - 1, j] = at_risk(fm, cfg.fidelity, geometric=c.family in GEOMETRIC_FAMILIES)
        log.info("condition %-18s done (%d levels)", c.name, L)

    tables = measures.summarize(y, part["patient_key"].to_numpy(), p_clean, cfg.conditions, P, FID, cfg, thr)
    per_image = _per_image_table(part, p_clean, P, FID, cfg, thr)

    valid = bool(meta.get("valid_as_result", True))
    reasons = []
    if meta.get("experiment_kind") == "smoke":
        reasons.append("model from a SMOKE TEST on synthetic data")
    if meta.get("label_control", "none") != "none":
        reasons.append("model from a NULL CONTROL (shuffled labels)")
    watermark = ("NOT A RESULT: " + " + ".join(reasons)) if not valid else None
    manifest = _manifest(checkpoint, meta, index_csv, sid, part, split, variant, final_report, thr, thr_source, cfg, tables,
                         max_patients, valid, watermark, dev)

    paths = _write(out, tables, per_image, manifest)
    if make_plots:
        paths.update(plots.make_all(out, tables, cfg, thr, split, watermark, originals[:review_n], ids[:review_n], cfg.seed))
    rv = review.review_template(
        ids[:review_n], cfg, {(i, c.name, k): bool(FID[c.name]["at_risk"][k - 1, j])
                              for j, i in enumerate(ids[:review_n]) for c in cfg.conditions for k in range(1, len(c.params) + 1)})
    (out / "review").mkdir(exist_ok=True)
    paths["review_template"] = out / "review" / "label_preservation_review.csv"
    rv.to_csv(paths["review_template"], index=False)
    manifest["artifacts"] = {k: str(v) for k, v in paths.items()}
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
    return AuditOutputs(tables, per_image, manifest, paths)


def _per_image_table(part, p_clean, P, FID, cfg, thr) -> pd.DataFrame:
    frames = []
    base = part[["image_id", "patient_key", "label"]].reset_index(drop=True)
    for c in cfg.conditions:
        for k in range(1, len(c.params) + 1):
            p = P[c.name][k - 1]
            d = base.copy()
            d["condition"], d["family"], d["level"], d["tier"], d["param"] = c.name, c.family, k, cfg.tier(k), c.params[k - 1]
            d["p_clean"], d["p_dist"] = p_clean, p
            d["pred_clean"], d["pred_dist"] = (p_clean >= thr).astype(int), (p >= thr).astype(int)
            d["flip"] = measures.prediction_flips(p_clean, p, thr)
            right_c, right_d = d["pred_clean"] == d["label"], d["pred_dist"] == d["label"]
            d["flip_to_wrong"], d["flip_to_right"] = right_c & ~right_d, ~right_c & right_d
            d["conf_clean"], d["conf_dist"] = measures.predicted_confidence(p_clean), measures.predicted_confidence(p)
            d["conf_drift"] = d["conf_dist"] - d["conf_clean"]
            for key in ("psnr_db", "correlation", "std_ratio", "new_saturated_fraction", "at_risk"):
                d[key] = FID[c.name][key][k - 1]
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _manifest(checkpoint, meta, index_csv, sid, part, split, variant, final_report, thr, thr_source, cfg, tables,
              max_patients, valid, watermark, dev) -> dict:
    top = tables.conditions[["rank", "condition", "mean_d_auroc_all", "mean_pfr_all", "mfs_median_label"]].head(5)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "valid_as_result": valid, "watermark": watermark,
        "audit_scope": ("FINAL REPORT on the test split" if final_report else
                        "DIAGNOSTIC audit on the validation split (use to choose M6 interventions)"),
        "split": split, "variant": variant, "seed": cfg.seed,
        "threshold": {"value": thr, "source": thr_source},
        "subsampled": {"max_patients": max_patients, "complete": not max_patients},
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint), "arch": meta.get("arch"),
                       "image_size": meta.get("image_size"), "split_id": meta.get("split_id"),
                       "weights_source": meta.get("weights_source"), "experiment_kind": meta.get("experiment_kind"),
                       "label_control": meta.get("label_control"), "training_seed": meta.get("seed")},
        "dataset": {"index_csv": str(index_csv), "split_id": sid, "n_images": int(len(part)),
                    "n_patients": int(part["patient_key"].nunique()), "n_hcc_images": int((part["label"] == 1).sum())},
        "config": json.loads(cfg.model_dump_json()),
        "clean": tables.clean, "worst_case": tables.worst_case, "any_flip": tables.any_flip,
        "top_disruptive_conditions": top.to_dict(orient="records"),
        "assumptions": ASSUMPTIONS,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(dev),
                        "liverguard": __version__},
    }


def _write(out: Path, tables, per_image, manifest) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {"per_image": out / "per_image.csv", "summary": out / "summary_by_level.csv",
             "conditions": out / "condition_ranking.csv", "families": out / "family_worst_case.csv",
             "manifest": out / "audit_manifest.json"}
    per_image.to_csv(paths["per_image"], index=False)
    tables.summary.to_csv(paths["summary"], index=False)
    tables.conditions.to_csv(paths["conditions"], index=False)
    tables.families.to_csv(paths["families"], index=False)
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
    return paths
