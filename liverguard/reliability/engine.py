"""M4 engine: run uncertainty + calibration + selective-prediction + failure analysis on one checkpoint / split.

Calibration discipline
- split='val' (default, diagnostic): temperature T is fitted on validation; because scoring T on the same images would flatter it,
  the headline calibrated numbers are CROSS-FITTED (patient folds), and the in-sample number is shown only as a labelled extra.
- split='test' (needs final_report=True): T is fitted on the VALIDATION split of the same index and applied unchanged to test.
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
from liverguard.ml.data import build_transform, load_resized
from liverguard.reliability import calibration as cal
from liverguard.reliability import failure, plots, selective, uncertainty as unc
from liverguard.session import Session, open_session, predict_proba
from liverguard.utils import file_sha256, get_logger

log = get_logger(__name__)

LIMITATIONS = [
    "Confidence and predictive entropy are rank-equivalent for a binary model; entropy adds no ranking information over confidence.",
    "MC-Dropout samples only the head dropout (feature-level); it understates full epistemic uncertainty.",
    "TTA spread probes sensitivity to small geometric changes (incl. horizontal flip); it is a stability probe, not a calibrated uncertainty.",
    "ECE is bin-dependent and biased for small n; equal-width and equal-mass ECE and bin counts are reported; CIs resample patients.",
    "Temperature scaling is one global rescale fitted on validation; it does not change AUROC and cannot fix subgroup miscalibration.",
    "Risk-coverage is estimated on one finite split; abstention thresholds must be chosen on validation, not on the reported split.",
    "Model confidence is not clinical certainty. LiverGuard is a research / model-auditing tool, not a diagnostic system.",
]


@dataclass
class ReliabilityOutputs:
    per_image: pd.DataFrame
    calibration: pd.DataFrame
    risk_coverage: pd.DataFrame
    manifest: dict
    paths: dict[str, Path]


def _mc_samples(model, images, device, bs, n_samples: int, seed: int) -> np.ndarray:
    """(n_samples, N) MC-Dropout probabilities; each batch uses a seed derived from (seed, batch start) -> reproducible."""
    tf, parts = build_transform(False), []
    for i in range(0, len(images), bs):
        x = torch.stack([tf(im) for im in images[i:i + bs]]).to(device)
        parts.append(unc.mc_dropout_probs(model, x, n_samples=n_samples, seed=seed + i))
    return np.concatenate(parts, axis=1)


def run_reliability_analysis(checkpoint, index_csv, out_dir, *, split: str = "val", variant: str | None = "clean",
                             threshold: float | str = 0.5, device: str = "auto", final_report: bool = False,
                             split_file=None, data_root=None, max_patients: int | None = None, seed: int = 0,
                             mc_samples: int = 20, n_bins: int = 10, n_boot: int = 500, confident_at: float = 0.9,
                             batch_size: int = 64, n_failures: int = 8, make_plots: bool = True) -> ReliabilityOutputs:
    out = Path(out_dir)
    S: Session = open_session(checkpoint, index_csv, split=split, variant=variant, threshold=threshold, device=device,
                              final_report=final_report, split_file=split_file, data_root=data_root,
                              max_patients=max_patients, seed=seed)
    torch.manual_seed(seed)
    images = S.load_images()
    y, groups, ids = S.y, S.groups, S.ids
    p = predict_proba(S.model, images, S.device, batch_size)
    mcs = unc.mc_summary(_mc_samples(S.model, images, S.device, batch_size, mc_samples, seed))
    tta = unc.tta_summary(unc.tta_probs(S.model, images, S.device, batch_size), S.threshold)

    df = S.part[["image_id", "patient_key", "label", "path"]].copy()
    df["image_id"] = ids
    df["p"] = p
    df["confidence"] = unc.confidence(p)
    df["entropy"] = unc.binary_entropy(p)
    for k, v in {**mcs, **tta}.items():
        df[k] = v

    # ---------------------------------------------------------------- calibration sets
    sets: dict[str, np.ndarray] = {"uncalibrated": p}
    notes: list[str] = []
    if split == "val":
        p_cf, Ts = cal.crossfit_temperature(y, p, groups, k=5, seed=seed)
        T_in = cal.fit_temperature(y, p)
        sets["temperature (cross-fitted on val)"] = p_cf
        sets["temperature (in-sample on val; optimistic)"] = cal.apply_temperature(p, T_in)
        T_info = {"mode": "val_crossfit", "fold_temperatures": Ts, "in_sample_T": T_in}
        notes.append("Calibrated numbers on the validation split are cross-fitted over patient folds; the in-sample row is optimistic.")
        p_cal = p_cf
    else:
        val = S.frame[S.frame["split"] == "val"].reset_index(drop=True)
        vp = predict_proba(S.model, [load_resized(q, S.size) for q in val["path"]], S.device, batch_size)
        T = cal.fit_temperature(val["label"].to_numpy(dtype=int), vp)
        sets["temperature (fitted on val, applied to test)"] = cal.apply_temperature(p, T)
        T_info = {"mode": "val_fit_test_apply", "T": T, "n_val_images": int(len(val))}
        p_cal = sets[next(k for k in sets if k != "uncalibrated")]
    df["p_calibrated"] = p_cal
    df = failure.label_failures(df, S.threshold, confident_at)

    rows, bins = [], {}
    for name, q in sets.items():
        s = cal.calibration_summary(y, q, n_bins)
        s["ece_ci95"] = cal.cluster_bootstrap(y, q, groups, lambda a, b: cal.expected_calibration_error(a, b, n_bins),
                                              n_boot=n_boot, seed=seed)
        s["brier_ci95"] = cal.cluster_bootstrap(y, q, groups, cal.brier_score, n_boot=n_boot, seed=seed)
        s["probabilities"] = name
        rows.append(s)
        bins[name] = cal.reliability_table(y, q, n_bins)
    calib = pd.DataFrame(rows)

    # ---------------------------------------------------------------- selective prediction
    err = (~df["correct"]).to_numpy().astype(float)
    # errors use the operating threshold, so the single-pass score is the distance to THAT threshold in logit space
    # (identical in ranking to 1 - max(p, 1-p) when threshold = 0.5)
    margin = -np.abs(cal.logit(df["p"].to_numpy()) - cal.logit(np.array([S.threshold]))[0])
    scores = {"margin to threshold": margin, "MC-Dropout std": df["mc_std"].to_numpy(),
              "MC-Dropout mutual info": df["mc_mi"].to_numpy(), "TTA std": df["tta_std"].to_numpy()}
    csum = selective.coverage_summary(err, scores)
    curves = {k: selective.risk_coverage(err, v) for k, v in scores.items()}
    rc_long = pd.concat([c.assign(score=k) for k, c in curves.items()], ignore_index=True)

    ebu = pd.concat([failure.error_by_uncertainty(df, s) for s in ("entropy", "mc_std", "tta_std") if df[s].nunique() > 3],
                    ignore_index=True) if len(df) >= 8 else pd.DataFrame()
    pat = failure.patient_level(df, S.threshold)
    top = df[~df["correct"]].sort_values("error_rank").head(n_failures)

    # ---------------------------------------------------------------- write
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def w(name, frame):
        paths[name] = out / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    w("per_image_reliability", df.drop(columns=["path"]))
    w("calibration_summary", calib.assign(ece_ci95=calib["ece_ci95"].astype(str), brier_ci95=calib["brier_ci95"].astype(str)))
    w("reliability_bins", pd.concat([b.assign(probabilities=k) for k, b in bins.items()], ignore_index=True))
    w("risk_coverage", rc_long)
    w("selective_summary", csum)
    w("failure_cases", top.drop(columns=["path"]))
    w("outcome_summary", failure.outcome_summary(df))
    w("patient_level", pat)
    if len(ebu):
        w("error_by_uncertainty", ebu)

    wm = S.watermark
    if make_plots:
        sub = f"{split} split · n={len(df)} images / {len(set(groups))} patients"
        keys = list(sets)
        paths["fig_reliability"] = plots.plot_reliability({k: bins[k] for k in keys[:2]}, {k: calib.set_index("probabilities").loc[k].to_dict() for k in keys[:2]},
                                                           out / "figures" / "reliability_diagram.png", f"Reliability diagram · {sub}", wm)
        paths["fig_confidence"] = plots.plot_confidence_distribution(df, out / "figures" / "confidence_distribution.png",
                                                                      f"Confidence and entropy: correct vs incorrect · {sub}", wm)
        orc = selective.risk_coverage(err, err)
        paths["fig_risk_coverage"] = plots.plot_risk_coverage(curves, orc, float(err.mean()), csum, out / "figures" / "risk_coverage.png",
                                                              f"Risk-coverage · {sub}", wm)
        if len(ebu):
            paths["fig_error_by_uncertainty"] = plots.plot_error_by_uncertainty(ebu, out / "figures" / "error_by_uncertainty.png",
                                                                                f"Error rate by uncertainty group · {sub}", wm)
        items = []
        by_id = dict(zip(ids, images))
        for _, r in top.iterrows():
            items.append({"image": by_id[r["image_id"]], "caption": f"{r['image_id']}\ntrue {'HCC' if r['label'] else 'Hemangioma'} · p(HCC)={r['p']:.2f}\n"
                          f"{r['outcome']} · entropy {r['entropy']:.2f} · TTA std {r['tta_std']:.2f}"})
        paths["fig_failures"] = plots.plot_failure_gallery(items, out / "figures" / "failure_gallery.png",
                                                           f"Largest-margin errors, worst first (not all are high-confidence; see confident_error column) · {sub}", wm)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "valid_as_result": wm is None, "watermark": wm, "split": split, "variant": variant, "seed": seed,
        "scope": "FINAL REPORT (test split)" if final_report else "DIAGNOSTIC on the validation split",
        "threshold": {"value": S.threshold, "source": S.threshold_source}, "confident_error_cut": confident_at,
        "mc_samples": mc_samples, "n_bins": n_bins, "n_boot": n_boot, "temperature": T_info, "notes": notes,
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(Path(checkpoint)), "arch": S.meta.get("arch"),
                       "split_id": S.meta.get("split_id"), "experiment_kind": S.meta.get("experiment_kind")},
        "dataset": {"split_id": S.split_id, "n_images": int(len(df)), "n_patients": int(len(set(groups))),
                    "subsampled": bool(max_patients)},
        "calibration": json.loads(calib.to_json(orient="records")),
        "selective": json.loads(csum.to_json(orient="records")),
        "outcomes": df["outcome"].value_counts().to_dict(), "n_confident_errors": int(df["confident_error"].sum()),
        "limitations": LIMITATIONS,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(S.device),
                        "liverguard": __version__},
    }
    paths["manifest"] = out / "reliability_manifest.json"
    manifest["artifacts"] = {k: str(v) for k, v in paths.items()}
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
    return ReliabilityOutputs(df, calib, rc_long, manifest, paths)
