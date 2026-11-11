"""Page 1: Research Dashboard - everything here is read from the registry / run files."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from liverguard.app import charts, ops, state, theme


def render() -> None:
    w = state.setup("Research Dashboard", "Build → Attack → Diagnose → Measure → Fix → Re-test: reliability auditing for liver ultrasound AI (HCC vs hemangioma)")
    reg = state.registry()
    runs = reg.runs()
    idx = state.index_df()

    c = st.columns(4)
    if idx is not None:
        o = ops.dataset_overview(idx)
        theme.kpi(c[0], "Images", f"{o['n_images']:,}", f"{o['n_patients']:,} patients (patient-level splits)")
        theme.kpi(c[1], "Classes", " · ".join(f"{k} {v:,}" for k, v in o["by_class"].items()), "images per class")
    else:
        theme.kpi(c[0], "Images", "n/a", "no image index in this workspace")
        theme.kpi(c[1], "Classes", "n/a", "run scripts/verify_dataset.py")
    valid = sum(r.valid_as_result for r in runs)
    theme.kpi(c[2], "Registered runs", str(len(runs)), f"{valid} valid results · {len(runs) - valid} pipeline checks")
    theme.kpi(c[3], "Models trained", str(len(reg.runs("train"))), "checkpoints in the registry")
    if idx is None:
        theme.banner("No image index found at " + str(w.index_csv) + ". Dataset pages stay empty until scripts/verify_dataset.py has been run on the real data.")
    if runs and not any(r.valid_as_result for r in runs):
        theme.banner("Every run in this workspace is a smoke test or null control on synthetic data. Nothing on this dashboard is a research finding.", "bad")

    st.markdown("### Research pipeline")
    has_diag = next(w.root.rglob("weakness_report.md"), None) is not None
    stat = ops.pipeline_status([(r.kind, r.valid_as_result) for r in runs], has_diag)
    cols = st.columns(len(stat))
    tone = {"not run": "info", "pipeline check only": "warn", "done": "ok"}
    for col, s in zip(cols, stat):
        col.markdown(f'<div class="lg-kpi"><div class="l">{s["stage"]}</div><div style="margin:6px 0">{theme.badge(s["status"], tone[s["status"]])}</div>'
                     f'<div class="n">{s["detail"]} · {s["n_runs"]} run(s)</div></div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### Latest experiments")
        if runs:
            df = pd.DataFrame([{"kind": r.kind, "name": r.name, "split": r.split, "model": r.arch, "created (UTC)": r.created_utc[:16],
                                "valid result": "yes" if r.valid_as_result else "NO"} for r in runs[:10]])
            st.dataframe(df, hide_index=True, width="stretch")
        else:
            theme.empty_state("No experiments yet", "Start with the Model Laboratory.")
    with right:
        st.markdown("### Key reliability metrics")
        rel = reg.runs("reliability")
        if rel:
            rows = []
            for r in rel[:6]:
                cal = json.loads(r.metrics_json).get("calibration") or []
                base = next((x for x in cal if x.get("probabilities") == "uncalibrated"), cal[0] if cal else {})
                rows.append({"run": r.name, "split": r.split, "ECE": theme.fmt(base.get("ece_width")), "Brier": theme.fmt(base.get("brier")),
                             "valid": "yes" if r.valid_as_result else "NO"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            theme.empty_state("No reliability analysis yet", "Run one on the Reliability Center page.")

    st.markdown("### Baseline versus robust model")
    m6 = reg.runs("m6_final") or reg.runs("m6_validation")
    if m6:
        r = m6[0]
        theme.watermark_banner(None if r.valid_as_result else r.watermark)
        pm = state.csv(Path(r.out_dir) / "paired_diffs_summary.csv")
        sel = json.loads(r.metrics_json).get("chosen")
        st.caption(f"{'Test re-audit' if r.kind == 'm6_final' else 'Validation ablation'} · frozen selection: **{sel}** · differences vs baseline, same seed")
        if pm is not None:
            a, b = st.columns(2)
            a.plotly_chart(charts.tradeoff_scatter(pm, None if r.valid_as_result else "x"), width="stretch")
            show = pm[pm["intervention"] == sel][["metric", "mean_diff", "min_diff", "max_diff", "verdict"]] if sel else pm.head(0)
            b.markdown(f"**{sel or 'selected intervention'}: paired differences**")
            b.dataframe(show.round(3), hide_index=True, width="stretch")
    else:
        theme.empty_state("No robust-training comparison yet", "Use the Robust Training Lab, then Model Comparison.")
    state.finish()
