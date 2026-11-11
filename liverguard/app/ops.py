"""Backend operations used by the UI. No Streamlit here: every function calls the ML modules and is unit-tested directly."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from liverguard.app.config import Workspace
from liverguard.explain import masking
from liverguard.explain.gradcam import GradCAM
from liverguard.explain.occlusion import occlusion_map
from liverguard.ml.config import TrainConfig
from liverguard.ml.data import preprocess_pil
from liverguard.ml.models import load_checkpoint
from liverguard.ml.trainer import resolve_device
from liverguard.robustness.config import Condition, RobustnessConfig
from liverguard.robustness.distortions import apply_distortion
from liverguard.robustness.fidelity import GEOMETRIC_FAMILIES, at_risk, fidelity_metrics
from liverguard.session import predict_proba, watermark_for

CLASSES = ("Hemangioma", "HCC")      # index 0 / 1; p(HCC) = softmax[:, 1]


# ---------------------------------------------------------------- model + single image
def load_model(checkpoint: str | Path, device: str = "auto") -> tuple[torch.nn.Module, dict, torch.device]:
    dev = resolve_device(device)
    model, meta = load_checkpoint(checkpoint, dev)
    return model.eval(), meta, dev


def checkpoint_watermark(meta: dict) -> str | None:
    return watermark_for(meta)


def p_hcc(model, imgs: list[Image.Image], device) -> np.ndarray:
    return predict_proba(model, imgs, device)


def open_rgb(path_or_file, size: int | None = None) -> Image.Image:
    img = Image.open(path_or_file).convert("RGB")
    return img.resize((size, size), Image.Resampling.BICUBIC) if size else img


def distort(img: Image.Image, cond: Condition, level: int, *, seed: int, image_id: str) -> Image.Image:
    if not 1 <= level <= len(cond.params):
        raise ValueError(f"{cond.name} has levels 1..{len(cond.params)}, got {level}")
    return apply_distortion(img, cond.func, cond.params[level - 1], seed=seed, image_id=image_id, condition=cond.name)


def compare_prediction(model, device, original: Image.Image, distorted: Image.Image, threshold: float) -> dict[str, Any]:
    p0, p1 = (float(x) for x in predict_proba(model, [original, distorted], device))
    c0, c1 = max(p0, 1 - p0), max(p1, 1 - p1)
    return {"p_original": p0, "p_distorted": p1, "pred_original": CLASSES[int(p0 >= threshold)], "pred_distorted": CLASSES[int(p1 >= threshold)],
            "flip": (p0 >= threshold) != (p1 >= threshold), "confidence_original": c0, "confidence_distorted": c1, "confidence_drift": c1 - c0,
            "abs_dp": abs(p1 - p0)}


def fidelity_report(original: Image.Image, distorted: Image.Image, cond: Condition, rob: RobustnessConfig) -> dict[str, Any]:
    m = fidelity_metrics(original, distorted)
    m["at_risk"] = at_risk(m, rob.fidelity, geometric=cond.family in GEOMETRIC_FAMILIES)
    m["geometric_skips_pixel_metrics"] = cond.family in GEOMETRIC_FAMILIES
    return m


def severity_sweep(model, device, img: Image.Image, image_id: str, conds: list[Condition], seed: int, threshold: float) -> pd.DataFrame:
    """p(HCC) of ONE image at level 0 (original) and every level of every selected condition (real inference on every row)."""
    imgs, rows = [img], [("original", 0, "original")]
    for c in conds:
        for k in range(1, len(c.params) + 1):
            imgs.append(distort(img, c, k, seed=seed, image_id=image_id))
            rows.append((c.name, k, "plausible"))
    p = predict_proba(model, imgs, device)
    base = p[0]
    out = []
    for (name, k, _), pi in zip(rows[1:], p[1:]):
        out.append({"condition": name, "level": k, "p": float(pi), "flip": bool((pi >= threshold) != (base >= threshold))})
    df = pd.DataFrame(out)
    start = pd.DataFrame([{"condition": c.name, "level": 0, "p": float(base), "flip": False} for c in conds])
    return pd.concat([start, df], ignore_index=True).sort_values(["condition", "level"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- explainability
def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a [0,1] heat map (inferno) over the image."""
    from matplotlib import colormaps
    heat = (colormaps["inferno"](np.clip(cam, 0, 1))[..., :3] * 255).astype(np.uint8)
    base = np.asarray(img.convert("RGB").resize(cam.shape[::-1], Image.Resampling.BICUBIC), dtype=np.float32)
    return Image.fromarray(((1 - alpha) * base + alpha * heat).astype(np.uint8))


def gradcam(model, img: Image.Image, size: int, device, target: str | int = "predicted") -> tuple[np.ndarray, np.ndarray]:
    """(cam HxW in [0,1], probs (2,)). target: 'predicted' | 0 | 1 (class-specific)."""
    x = preprocess_pil(img, size).unsqueeze(0).to(device)
    gc = GradCAM(model)
    try:
        cam, probs = gc(x, target)
    finally:
        gc.close()
    return cam[0], probs[0]


def occlusion(model, img: Image.Image, size: int, device, target: int, patch: int, stride: int) -> tuple[np.ndarray, float]:
    x = preprocess_pil(img, size).unsqueeze(0).to(device)
    return occlusion_map(model, x, target, patch=patch, stride=stride)


def roi_experiment(model, device, img: Image.Image, roi: np.ndarray, seed: int, n_control: int = 5) -> pd.DataFrame:
    """p(HCC) of: full image, ROI-only, background-only, and matched-area random-placement controls (mean over draws)."""
    v = masking.region_variants(img, roi, np.random.default_rng(seed), n_control=n_control)
    rows = [("full image", [img], "")]
    rows += [("ROI only", v["roi_only"], ""), ("background only", v["background_only"], ""),
             ("control: same-area region only", v["control_roi_only"], "mean over random placements"),
             ("control: same-area region removed", v["control_background_only"], "mean over random placements")]
    out = []
    for name, imgs, note in rows:
        ps = predict_proba(model, imgs, device)
        out.append({"variant": name, "p_hcc": float(ps.mean()), "p_min": float(ps.min()), "p_max": float(ps.max()), "n": len(ps), "note": note})
    return pd.DataFrame(out)


def load_uploaded_mask(file, size: int) -> np.ndarray:
    with Image.open(file) as im:
        m = np.asarray(im.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0
    if m.sum() == 0 or m.all():
        raise ValueError("The mask is empty or covers the whole image; it must mark a proper region.")
    return m


def mask_figure_image(img: Image.Image, roi: np.ndarray) -> Image.Image:
    a = np.asarray(img.convert("RGB").resize(roi.shape[::-1])).astype(np.float32)
    edge = roi ^ np.roll(roi, 1, 0) | roi ^ np.roll(roi, 1, 1)
    a[edge] = [255, 220, 0]
    a[~roi] *= 0.5
    return Image.fromarray(a.astype(np.uint8))


# ---------------------------------------------------------------- command builders (jobs)
def check_run_name(name: str) -> str:
    """Run names become directory and file names: letters, digits, '_', '-', '.' only; no path separators, no leading dot."""
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", name or ""):
        raise ValueError("Run name must be 1-81 characters of letters, digits, '_', '-' or '.', starting with a letter or digit.")
    return name


def train_config_file(ws: Workspace, params: dict[str, Any], name: str) -> Path:
    """Validate with the real TrainConfig (so bad settings fail here, not mid-run) and write the YAML train.py reads."""
    check_run_name(name)
    cfg = TrainConfig(**{**params, "run_name": name})
    path = ws.jobs_dir / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json()), sort_keys=False))
    return path


def train_argv(ws: Workspace, cfg_path: Path, out_dir: Path, *, seed: int | None = None, label_control: str | None = None,
               weights: str | None = None) -> list[str]:
    a = [sys.executable, ws.script("train.py"), "--config", str(cfg_path), "--index-csv", str(ws.index_csv), "--out-dir", str(out_dir)]
    if seed is not None:
        a += ["--seed", str(seed)]
    if label_control and label_control != "none":
        a += ["--label-control", label_control]
    if weights:
        a += ["--pretrained-weights-path", weights]
    return a


def audit_argv(ws: Workspace, ckpt: str, out_dir: Path, *, split: str = "val", threshold: str = "0.5", conditions: list[str] | None = None,
               n_boot: int = 200, seed: int = 0, final_report: bool = False, max_patients: int | None = None) -> list[str]:
    a = [sys.executable, ws.script("audit_robustness.py"), "--checkpoint", ckpt, "--index-csv", str(ws.index_csv), "--out-dir", str(out_dir),
         "--config", str(ws.robustness_config), "--split", split, "--threshold", threshold, "--n-boot", str(n_boot), "--seed", str(seed)]
    if conditions:
        a += ["--conditions", ",".join(conditions)]
    if max_patients:
        a += ["--max-patients", str(max_patients)]
    return a + (["--final-report"] if final_report else [])


def reliability_argv(ws: Workspace, ckpt: str, out_dir: Path, *, split: str = "val", threshold: str = "0.5", n_boot: int = 200,
                     mc_samples: int = 20, final_report: bool = False) -> list[str]:
    a = [sys.executable, ws.script("reliability.py"), "--checkpoint", ckpt, "--index-csv", str(ws.index_csv), "--out-dir", str(out_dir),
         "--split", split, "--threshold", threshold, "--n-boot", str(n_boot), "--mc-samples", str(mc_samples)]
    return a + (["--final-report"] if final_report else [])


def explain_argv(ws: Workspace, ckpt: str, out_dir: Path, *, split: str = "val", threshold: str = "0.5", n_boot: int = 200,
                 mask_dir: str | None = None, final_report: bool = False) -> list[str]:
    a = [sys.executable, ws.script("explain.py"), "--checkpoint", ckpt, "--index-csv", str(ws.index_csv), "--out-dir", str(out_dir),
         "--split", split, "--threshold", threshold, "--n-boot", str(n_boot)]
    if mask_dir:
        a += ["--mask-dir", mask_dir]
    return a + (["--final-report"] if final_report else [])


def robust_argv(ws: Workspace, stage: str, workdir: Path, cfg_path: Path, seeds: str, *, n_boot: int = 200, interventions: str | None = None,
                epochs: int | None = None, final_report: bool = False, weights: str | None = None) -> list[str]:
    a = [sys.executable, ws.script("robust_train.py"), stage, "--workdir", str(workdir), "--config", str(cfg_path), "--index-csv", str(ws.index_csv),
         "--seeds", seeds, "--rob-config", str(ws.robustness_config), "--n-boot", str(n_boot)]
    if interventions:
        a += ["--interventions", interventions]
    if epochs:
        a += ["--epochs", str(epochs)]
    if weights:
        a += ["--pretrained-weights-path", weights]
    return a + (["--final-report"] if final_report else [])


def tail(path: str | Path, n: int = 40) -> str:
    p = Path(path)
    return "\n".join(p.read_text(errors="replace").splitlines()[-n:]) if p.exists() else ""


# ---------------------------------------------------------------- dataset
def dataset_overview(idx: pd.DataFrame) -> dict[str, Any]:
    pc = idx.groupby("patient_key").size()
    return {"n_images": len(idx), "n_patients": idx["patient_key"].nunique(), "images_per_patient": pc.describe().to_dict(),
            "by_class": idx["class_label"].value_counts().to_dict(), "by_split": idx["split"].value_counts().to_dict(),
            "by_variant": idx["variant"].value_counts().to_dict()}


def image_stats(paths: list[str], max_n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Measured on the files themselves (a seeded sample if there are many): size, mean/std intensity, 1st/99th percentile, readable?"""
    paths = list(paths)
    if len(paths) > max_n:
        paths = list(np.random.default_rng(seed).choice(paths, max_n, replace=False))
    rows = []
    for p in paths:
        try:
            with Image.open(p) as im:
                g = np.asarray(im.convert("L"), dtype=np.float32)
                w, h = im.size
            rows.append({"path": p, "width": w, "height": h, "mean": float(g.mean()), "std": float(g.std()), "p1": float(np.percentile(g, 1)),
                         "p99": float(np.percentile(g, 99)), "ok": True})
        except Exception:                                                   # noqa: BLE001 - corrupt files are a finding, not a crash
            rows.append({"path": p, "width": np.nan, "height": np.nan, "mean": np.nan, "std": np.nan, "p1": np.nan, "p99": np.nan, "ok": False})
    return pd.DataFrame(rows)


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- pipeline status (dashboard)
STAGES = [("BUILD", ("train",), "baseline models trained"), ("ATTACK", ("audit",), "robustness audits"),
          ("DIAGNOSE", ("diagnosis",), "weaknesses identified (M6 diagnosis)"), ("MEASURE", ("reliability", "explain"), "calibration, uncertainty, explainability"),
          ("FIX", ("m6_validation",), "targeted robust training, validation ablation"), ("RE-TEST", ("m6_final",), "frozen selection re-audited on the test split")]


def pipeline_status(kinds_valid: list[tuple[str, bool]], has_diagnosis: bool) -> list[dict[str, str]]:
    """kinds_valid: [(run kind, valid_as_result)] from the registry. Status is derived from what exists on disk, nothing else:
    'not run' | 'pipeline check only' (all runs are smoke/null) | 'done' (at least one valid run)."""
    out = []
    for stage, kinds, desc in STAGES:
        rows = [v for k, v in kinds_valid if k in kinds]
        if "diagnosis" in kinds:
            rows = rows + ([False] if has_diagnosis else [])
        status = "not run" if not rows else ("done" if any(rows) else "pipeline check only")
        out.append({"stage": stage, "status": status, "detail": desc, "n_runs": str(len(rows))})
    return out
