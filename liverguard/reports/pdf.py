"""Build a PDF report (reportlab) from registry runs. Every number comes from a manifest or CSV written by the ML modules.

Non-result runs (smoke tests, null controls) are labelled on the title page and again at their section; a report that contains any
is stamped as such. Method limitations are copied from the manifests when present, else from a fixed text per kind.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY, TEAL, LINE = colors.HexColor("#14213D"), colors.HexColor("#3E8E8B"), colors.HexColor("#E4E0DA")
DISCLAIMER = "LiverGuard is a research and model-auditing platform. It is not a clinical diagnostic system; nothing here is clinically validated."
KIND_TITLES = {"train": "Training run", "audit": "Robustness audit", "reliability": "Reliability analysis", "explain": "Explainability analysis",
               "m6_validation": "Robust training: validation ablation and frozen selection", "m6_final": "Robust training: final test re-audit"}
FALLBACK_LIMITS = {
    "train": ["Single split and (usually) a single seed: differences between models smaller than seed-to-seed variation are not findings.",
              "Metrics on the test split are reported once, after model selection on validation."],
    "audit": ["Distortions are synthetic and applied after resizing; they approximate, not reproduce, acquisition variation.",
              "Plausible vs stress tiers are a-priori; label preservation is unverified until an expert reviews the sheet.",
              "Rotation and crop can remove the lesion; pixel-fidelity proxies cannot detect this."],
    "reliability": ["ECE is bin-dependent and biased at small n; read it with AUROC and Brier score.",
                    "Model confidence is not clinical certainty. MC-Dropout here samples the head only."],
    "explain": ["Grad-CAM is low-resolution and not evidence of medical reasoning.", "Fill-based masking creates out-of-distribution inputs.",
                "The centre-proxy ROI is not the liver or the lesion unless annotation masks were supplied."],
    "m6_validation": ["Plausible-tier AUROC gains can reflect a better clean model, not less degradation: see the degradation-gap metric."],
    "m6_final": ["The test split is used once, after the selection is frozen."],
}


def _styles():
    ss = getSampleStyleSheet()
    return {"title": ParagraphStyle("t", parent=ss["Title"], textColor=NAVY, fontSize=22, alignment=0),
            "h1": ParagraphStyle("h1", parent=ss["Heading1"], textColor=NAVY, fontSize=15),
            "h2": ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY, fontSize=11.5),
            "body": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12),
            "small": ParagraphStyle("s", parent=ss["BodyText"], fontSize=7.5, leading=9.5, textColor=colors.HexColor("#5B6475")),
            "warn": ParagraphStyle("w", parent=ss["BodyText"], fontSize=9, leading=12, textColor=colors.HexColor("#8A2F1C"),
                                   backColor=colors.HexColor("#F9EAE6"), borderPadding=6)}


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return "n/a" if x != x else (str(x) if isinstance(x, int) else f"{x:.{nd}f}")
    return str(x)


def _table(rows: list[list[Any]], widths: list[float] | None = None) -> Table:
    t = Table([[_fmt(c) for c in r] for r in rows], colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                           ("GRID", (0, 0), (-1, -1), 0.3, LINE), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F5F2")]),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def _df_table(df: pd.DataFrame, cols: list[str], nd: int = 3, max_rows: int = 25) -> Table:
    cols = [c for c in cols if c in df.columns]
    return _table([cols] + [[_fmt(v, nd) for v in r] for r in df[cols].head(max_rows).itertuples(index=False)])


def _read_csv(d: Path, name: str) -> pd.DataFrame | None:
    p = d / name
    return pd.read_csv(p) if p.exists() else None


def _figures(out_dir: Path, limit: int = 6) -> list[Path]:
    pics = sorted((out_dir / "figures").glob("*.png")) + sorted(out_dir.glob("*.png"))
    return pics[:limit]


def _image(path: Path, width_mm: float = 150):
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    return RLImage(str(path), width=width_mm * mm, height=width_mm * mm * h / w)


def _run_section(run: dict[str, Any], S: dict) -> list:
    out_dir, man = Path(run["out_dir"]), run.get("manifest") or {}
    el: list = [Paragraph(f"{KIND_TITLES.get(run['kind'], run['kind'])}: {run['name']}", S["h1"])]
    facts = [["created", run.get("created_utc") or "n/a"], ["valid as a result", "yes" if run["valid_as_result"] else "NO"],
             ["architecture", run.get("arch") or "n/a"], ["split analysed", run.get("split") or "see manifest"], ["split_id", run.get("split_id") or "n/a"],
             ["seed", run.get("seed") if run.get("seed") is not None else "n/a"], ["checkpoint sha256", (run.get("checkpoint_sha256") or "n/a")[:24]],
             ["manifest", run["manifest_path"]]]
    el += [_table([["field", "value"]] + facts, [40 * mm, 130 * mm]), Spacer(1, 4)]
    if run.get("watermark"):
        el += [Paragraph(f"<b>{run['watermark']}</b>: this section documents a pipeline check, not a finding.", S["warn"]), Spacer(1, 4)]
    m = run.get("metrics") or {}
    k = run["kind"]
    if k == "train":
        rows = [["split", "n", "AUROC", "accuracy", "sensitivity", "specificity", "F1", "Brier"]]
        for sp in ("val", "test"):
            r = (man.get("results") or {}).get(sp)
            if r:
                ci = (r.get("ci95_patient_bootstrap") or {}).get("auroc")
                rows.append([sp, r["n"], f"{_fmt(r['auroc'])} [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else _fmt(r["auroc"]), r["accuracy"], r["sensitivity"],
                             r["specificity"], r["f1"], r["brier"]])
        el += [Paragraph("Performance (threshold " + _fmt((man.get("results") or {}).get("val", {}).get("threshold")) + ", patient-level bootstrap CI)", S["h2"]), _table(rows)]
        h = _read_csv(out_dir, "history.csv")
        if h is not None:
            el += [Paragraph("Training history", S["h2"]), _df_table(h, ["epoch", "phase", "train_loss", "val_loss", "train_auroc", "val_auroc"], 4, 40)]
    elif k == "audit":
        c = m.get("clean") or {}
        el += [Paragraph(f"Clean performance: AUROC {_fmt(c.get('auroc'))}, accuracy {_fmt(c.get('accuracy'))}, F1 {_fmt(c.get('f1'))} "
                         f"(n={m.get('n_images')} images).", S["body"])]
        cond = _read_csv(out_dir, "condition_ranking.csv")
        if cond is not None:
            el += [Paragraph("Distortions ranked by mean AUROC change", S["h2"]),
                   _df_table(cond.sort_values("rank"), ["rank", "condition", "mean_d_auroc_all", "mean_pfr_all", "pfr_at_plausible_max", "mfs_median_label", "worst_auroc"])]
        el += [Paragraph("Assumptions", S["h2"])] + [Paragraph("- " + a, S["small"]) for a in (man.get("assumptions") or [])]
    elif k == "reliability":
        cal = _read_csv(out_dir, "calibration_summary.csv")
        if cal is not None:
            el += [Paragraph("Calibration", S["h2"]), _df_table(cal, ["probabilities", "ece_width", "ece_mass", "mce", "brier", "nll", "mean_confidence"])]
        sel = _read_csv(out_dir, "selective_summary.csv")
        if sel is not None:
            el += [Paragraph("Selective prediction (AURC, lower is better)", S["h2"]), _df_table(sel, ["score", "aurc", "aurc_oracle", "aurc_random"])]
        oc = _read_csv(out_dir, "outcome_summary.csv")
        if oc is not None:
            el += [Paragraph("Outcomes", S["h2"]), _df_table(oc, ["outcome", "n", "mean_p", "mean_entropy"])]
    elif k == "explain":
        el += [Paragraph(f"Region source: {man.get('roi_source', 'n/a')}", S["body"])]
        ms = _read_csv(out_dir, "masking_summary.csv")
        if ms is not None:
            el += [Paragraph("Region masking", S["h2"]), _df_table(ms, ["variant", "n", "auroc", "auroc_lo", "auroc_hi", "flip_rate_vs_full"])]
        cf = _read_csv(out_dir, "cam_faithfulness.csv")
        if cf is not None:
            el += [Paragraph("Grad-CAM deletion / insertion vs matched random regions", S["h2"]), _df_table(cf, ["variant", "n", "mean_evidence_drop", "lo", "hi"])]
    else:                                                                  # m6
        pm = _read_csv(out_dir, "paired_diffs_summary.csv")
        if pm is not None:
            el += [Paragraph("Paired differences vs baseline (same seed)", S["h2"]),
                   _df_table(pm, ["intervention", "metric", "n_seeds", "mean_diff", "min_diff", "max_diff", "verdict"], 3, 40)]
        rep = out_dir / "report.md"
        if rep.exists():
            el += [Paragraph("Report text", S["h2"])] + [Paragraph(l.replace("&", "&amp;").replace("<", "&lt;"), S["small"]) for l in rep.read_text().splitlines()[:40] if l.strip()]
    lim = man.get("limitations")
    items = ([f"{a}: {b}" for a, b in lim.items()] if isinstance(lim, dict) else [str(x) for x in lim]) if lim else FALLBACK_LIMITS.get(k, [])
    el += [Paragraph("Limitations", S["h2"])] + [Paragraph("- " + i, S["small"]) for i in items]
    figs = _figures(out_dir)
    if figs:
        el += [Paragraph("Figures", S["h2"])] + [KeepTogether([_image(f), Paragraph(f.name, S["small"])]) for f in figs]
    return el + [PageBreak()]


def build_report(path: str | Path, title: str, runs: list[dict[str, Any]], author_note: str = "") -> Path:
    """runs: dicts with keys kind, name, out_dir, manifest_path, created_utc, valid_as_result, watermark, arch, split, split_id, seed,
    checkpoint_sha256, metrics (headline dict), manifest (full manifest dict, optional). Returns the written path."""
    if not runs:
        raise ValueError("Select at least one run to report on")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    S = _styles()
    nonres = [r for r in runs if not r["valid_as_result"]]
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm, title=title,
                            author="LiverGuard")
    el: list = [Paragraph("LiverGuard", S["small"]), Paragraph(title, S["title"]),
                Paragraph(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from {len(runs)} registered run(s).", S["body"]),
                Spacer(1, 6), Paragraph(DISCLAIMER, S["body"]), Spacer(1, 6)]
    if nonres:
        el += [Paragraph(f"<b>{len(nonres)} of {len(runs)} run(s) in this report are NOT valid results</b> (smoke tests on synthetic data and/or null controls). "
                         "Their numbers validate the software pipeline only and must not be quoted as findings.", S["warn"]), Spacer(1, 6)]
    if author_note.strip():
        el += [Paragraph("Notes", S["h2"]), Paragraph(author_note.replace("&", "&amp;").replace("<", "&lt;"), S["body"])]
    el += [Paragraph("Runs included", S["h2"]),
           _table([["kind", "name", "valid result", "created"]] + [[KIND_TITLES.get(r["kind"], r["kind"]), r["name"], "yes" if r["valid_as_result"] else "NO",
                                                                   r.get("created_utc") or "n/a"] for r in runs]), PageBreak()]
    for r in runs:
        el += _run_section(r, S)
    doc.build(el)
    return path


def run_to_dict(run, manifest: bool = True) -> dict[str, Any]:
    """Registry Run row -> dict for build_report (loads the full manifest from disk)."""
    d = {c: getattr(run, c) for c in ("kind", "name", "out_dir", "manifest_path", "created_utc", "valid_as_result", "watermark", "arch", "split",
                                       "split_id", "seed", "checkpoint_sha256")}
    d["metrics"] = json.loads(run.metrics_json or "{}")
    d["manifest"] = json.loads(Path(run.manifest_path).read_text()) if manifest and Path(run.manifest_path).exists() else {}
    return d
