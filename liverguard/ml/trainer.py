"""Experiment runner: train on train, select on validation, evaluate test exactly once.

Guarantees: patient-level separation is asserted before any training; the test split is not touched
until training has finished and the best-validation checkpoint has been restored; the decision
threshold is fixed in the config (never tuned on test).
"""
from __future__ import annotations

import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from liverguard import __version__
from liverguard.data.report import json_default
from liverguard.data.splits import split_manifest_id
from liverguard.ml import plots
from liverguard.ml.config import TrainConfig
from liverguard.ml.data import (UltrasoundDataset, apply_patient_label_shuffle, assert_patient_separation,
                                compute_class_weights, load_split_frame, make_loader)
from liverguard.ml.evaluate import evaluate_frame, predict_loader, prediction_table
from liverguard.ml.metrics import compute_metrics, youden_threshold
from liverguard.ml.models import build_model, param_groups, save_checkpoint, set_backbone_requires_grad
from liverguard.utils import get_logger

log = get_logger(__name__)
CLASS_NAMES = ["Hemangioma", "HCC"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, grad_clip=None, amp=False):
    train = optimizer is not None
    model.train(train)
    total, n, probs, labels = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits.float(), y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            total += loss.item() * len(y)
            n += len(y)
            probs.append(torch.softmax(logits.detach().float(), 1)[:, 1].cpu().numpy())
            labels.append(y.cpu().numpy())
    p, y = np.concatenate(probs), np.concatenate(labels)
    m = compute_metrics(y, p)
    return {"loss": total / n, "acc": m["accuracy"], "auroc": m["auroc"]}


def _score(val: dict, monitor: str) -> float:
    v = val["auroc"] if monitor == "val_auroc" else -val["loss"]
    return -np.inf if np.isnan(v) else v   # a NaN AUROC can never be 'best'


def is_improvement(score: float, best: float, loss: float, best_loss: float, min_delta: float) -> bool:
    """Better monitor score by > min_delta, or an equal score with a lower validation loss (tie-break).

    Without the tie-break, an AUROC plateau (e.g. several epochs at 1.000) keeps the FIRST epoch even when
    later epochs are far better calibrated.
    """
    return score > best + min_delta or (score == best and loss < best_loss)


def run_experiment(cfg: TrainConfig, index_csv: str | Path, out_dir: str | Path, *,
                   split_file: str | Path | None = None, data_root: str | Path | None = None) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    wm = cfg.watermark

    # ---- data (patient separation verified inside load_split_frame)
    frame = load_split_frame(index_csv, split_file=split_file, variant=cfg.variant, data_root=data_root)
    if cfg.label_control == "patient_shuffle":
        frame = apply_patient_label_shuffle(frame, cfg.seed)
    assert_patient_separation(frame)
    parts = {s: frame[frame["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    sid = split_manifest_id(frame)
    log.info("split_id=%s  images: %s", sid, {k: len(v) for k, v in parts.items()})

    train_ds = UltrasoundDataset(parts["train"], cfg.image_size, augment=cfg.augment == "basic", cache=cfg.cache_images)
    val_ds = UltrasoundDataset(parts["val"], cfg.image_size, augment=False, cache=cfg.cache_images)
    train_loader = make_loader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, seed=cfg.seed)
    val_loader = make_loader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, seed=cfg.seed)

    # ---- model / optimisation
    model, weights_src = build_model(cfg.arch, pretrained=cfg.pretrained, weights_path=cfg.pretrained_weights_path,
                                     dropout=cfg.dropout)
    model.to(device)
    weights = compute_class_weights(parts["train"]["label"].to_numpy(), 2) if cfg.class_weighting == "auto" else np.ones(2)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(param_groups(model, cfg.lr, cfg.backbone_lr_mult), weight_decay=cfg.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")

    history: list[dict] = []
    best, best_loss, best_epoch, bad = -np.inf, np.inf, 0, 0
    ckpt_path = out / "best.pt"
    ckpt_meta = {"arch": cfg.arch, "num_classes": 2, "dropout": cfg.dropout, "image_size": cfg.image_size,
                 "class_names": CLASS_NAMES, "positive_class": "HCC", "weights_source": weights_src,
                 "split_id": sid, "seed": cfg.seed, "experiment_kind": cfg.experiment_kind,
                 "label_control": cfg.label_control, "valid_as_result": cfg.valid_as_result}

    for epoch in range(1, cfg.epochs + 1):
        frozen = epoch <= cfg.freeze_epochs
        set_backbone_requires_grad(model, not frozen)
        scale = 0.5 * (1 + np.cos(np.pi * (epoch - 1) / cfg.epochs))   # cosine decay per epoch
        for g, lr0 in zip(optimizer.param_groups, base_lrs):
            g["lr"] = lr0 * scale
        tr = _run_epoch(model, train_loader, device, criterion, optimizer, scaler, cfg.grad_clip, cfg.amp)
        va = _run_epoch(model, val_loader, device, criterion)
        history.append({"epoch": epoch, "phase": "frozen" if frozen else "finetune",
                        **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}})
        score = _score(va, cfg.monitor)
        log.info("epoch %d/%d [%s] train_loss=%.4f val_loss=%.4f val_auroc=%.4f",
                 epoch, cfg.epochs, history[-1]["phase"], tr["loss"], va["loss"], va["auroc"])
        if is_improvement(score, best, va["loss"], best_loss, cfg.min_delta):
            best, best_loss, best_epoch, bad = score, va["loss"], epoch, 0
            save_checkpoint(ckpt_path, model, {**ckpt_meta, "epoch": epoch, "val_" + cfg.monitor.split("_")[1]: va})
        elif not frozen:
            bad += 1
            if bad >= cfg.patience:
                log.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
                break
    if not ckpt_path.exists():  # validation score never became finite: keep final weights, and say so
        save_checkpoint(ckpt_path, model, {**ckpt_meta, "epoch": len(history)})
        best_epoch = len(history)
        log.warning("No finite validation score; saved last-epoch weights as the checkpoint")
    hist = pd.DataFrame(history)
    hist.to_csv(out / "history.csv", index=False)

    # ---- restore best-validation weights, then evaluate (test is touched only here, once)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)["model_state"]
    model.load_state_dict(state)
    results: dict = {}
    preds = []
    probs: dict[str, np.ndarray] = {}
    for name in ("val", "test"):
        ds = val_ds if name == "val" else UltrasoundDataset(parts[name], cfg.image_size, False, cfg.cache_images)
        loader = val_loader if name == "val" else make_loader(ds, batch_size=cfg.batch_size, shuffle=False,
                                                              num_workers=cfg.num_workers, seed=cfg.seed)
        p, _, _ = predict_loader(model, loader, device)
        probs[name] = p
        results[name] = evaluate_frame(parts[name], p, threshold=cfg.threshold,
                                       n_boot=cfg.bootstrap_samples, seed=cfg.seed)
        preds.append(prediction_table(parts[name], p, cfg.threshold))
    # Second operating point: threshold fitted on VALIDATION predictions only, applied unchanged to test.
    thr_val = youden_threshold(parts["val"]["label"].to_numpy(), probs["val"])
    results["val_youden_threshold"] = thr_val
    results["test_at_val_youden"] = evaluate_frame(parts["test"], probs["test"], threshold=thr_val,
                                                   n_boot=cfg.bootstrap_samples, seed=cfg.seed)
    pd.concat(preds).to_csv(out / "predictions.csv", index=False)
    te = results["test"]

    # ---- plots
    title = f"{cfg.arch} (seed {cfg.seed})"
    best_row = hist[hist["epoch"] == best_epoch].iloc[0]
    plots.plot_training_curves(hist, out / "training_curves.png", title, wm, best_epoch)
    plots.plot_confusion_matrix(te["confusion_matrix"], CLASS_NAMES, out / "confusion_matrix_test.png",
                                f"{title}: test", wm)
    ptab = pd.concat(preds)
    pt = ptab[ptab["split"] == "test"]
    if pt["label"].nunique() == 2:
        plots.plot_roc(pt["label"], pt["p_hcc"], out / "roc_test.png", f"{title}: test ROC", te["auroc"],
                       te["ci95_patient_bootstrap"]["auroc"], wm)

    manifest = {
        "run_name": cfg.run_name or out.name, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "valid_as_result": cfg.valid_as_result, "watermark": wm, "experiment_kind": cfg.experiment_kind,
        "label_control": cfg.label_control, "config": json.loads(cfg.model_dump_json()),
        "dataset": {"index_csv": str(index_csv), "split_file": str(split_file) if split_file else None,
                    "variant": cfg.variant, "split_id": sid, "patient_leakage_check": "passed",
                    "images": {k: len(v) for k, v in parts.items()},
                    "patients": {k: int(v["patient_key"].nunique()) for k, v in parts.items()},
                    "class_weights": [float(w) for w in weights]},
        "model": {"arch": cfg.arch, "weights_source": weights_src},
        "training": {"epochs_run": len(history), "best_epoch": int(best_epoch),
                     "selected_on": cfg.monitor, "selected_val_row": best_row.to_dict(),
                     "seconds": round(time.time() - t0, 1), "device": str(device)},
        "results": results,
        "artifacts": {p.name: str(p) for p in sorted(out.iterdir()) if p.is_file()},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "liverguard": __version__},
        "test_set_policy": "Test split evaluated once, after training, with the best-validation checkpoint. "
                           "Primary operating point: threshold fixed in config. Secondary: Youden threshold "
                           "fitted on validation only. Neither is tuned on test.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
    return manifest
