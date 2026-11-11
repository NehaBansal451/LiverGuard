"""Page 8: Model Comparison - any two checkpoints (matched audits) and the M6 ablation / baseline-vs-robust results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from liverguard.app import charts, state, theme

POINT = ["clean_auroc", "clean_accuracy", "clean_sensitivity", "clean_specificity", "clean_brier", "clean_ece", "plaus_auroc", "plaus_flip_rate", "stress_auroc", "stress_flip_rate", "plaus_ece"]


def render() -> None:
    state.setup("Model Comparison", "Baseline versus improved models on exactly the same untouched patient-level split.")
    t_pair, t_abl = st.tabs(["Two checkpoints", "Ablation (robust training)"])
    with t_pair:
        _pair()
    with t_abl:
        _ablation()
    state.finish()


def _pair() -> None:
    runs = state.checkpoint_runs()
    if len(runs) < 2:
        theme.empty_state("Need at least two trained checkpoints to compare")
        return
    c1, c2 = st.columns(2)
    ia = c1.selectbox("Model A", range(len(runs)), format_func=lambda k: state.run_label(runs[k]), key="cmp_a")
    ib = c2.selectbox("Model B", range(len(runs)), index=1, format_func=lambda k: state.run_label(runs[k]), key="cmp_b")
    a, b = runs[ia], runs[ib]
    wm = None
    if not (a.valid_as_result and b.valid_as_result):
        wm = "NOT A RESULT"
        theme.banner("At least one model is a smoke/synthetic/null run: every number below is a pipeline check, not a finding.")
    st.markdown("#### Audits on the same split and threshold (matched by checkpoint)")
    audits = state.registry().runs("audit")
    ha = [r for r in audits if r.checkpoint_sha256 and r.checkpoint_sha256 == a.checkpoint_sha256]
    hb = [r for r in audits if r.checkpoint_sha256 and r.checkpoint_sha256 == b.checkpoint_sha256]
    common = sorted({r.split for r in ha} & {r.split for r in hb})
    if not common:
        theme.empty_state("No matched audits", "Run a robustness audit of both models on the same split (Image Audit / scripts/audit_robustness.py) to compare robustness.")
    else:
        split = st.selectbox("Split", common, key="cmp_split")
        ra = next(r for r in ha if r.split == split)
        rb = next(r for r in hb if r.split == split)
        sa, sb = state.csv(Path(ra.out_dir) / "summary_by_level.csv"), state.csv(Path(rb.out_dir) / "summary_by_level.csv")
        if sa is not None and sb is not None:
            metric = st.selectbox("Metric", [m for m in ("auroc", "accuracy", "f1", "pfr", "conf_drift", "mean_abs_dp") if m in sa], key="cmp_metric")
            conds = sorted(set(sa["condition"]) & set(sb["condition"]))
            sel = st.multiselect("Conditions", conds, default=conds[:4], key="cmp_conds")
            if sel:
                import plotly.graph_objects as go
                fig = go.Figure()
                for i, (tag, s, r) in enumerate((("A", sa, ra), ("B", sb, rb))):
                    for j, c in enumerate(sel):
                        g = s[s["condition"] == c].sort_values("level")
                        fig.add_scatter(x=g["level"], y=g[metric], mode="lines+markers", name=f"{tag}: {c}", line=dict(color=charts.SERIES[j % len(charts.SERIES)], dash="solid" if tag == "A" else "dash"),
                                        marker=dict(symbol=charts.MARKERS[i]))
                fig.update_layout(height=420, title=f"{metric} vs severity (solid = A, dashed = B)", xaxis_title="severity level", yaxis_title=metric)
                st.plotly_chart(charts._wm(fig, wm), width="stretch")
            st.caption("Both audits must share the split, threshold and distortion config for a fair comparison; check the audit manifests in Research Reports.")
    st.markdown("#### Clean performance recorded at training time")
    tab = []
    for tag, r in (("A", a), ("B", b)):
        d = {"model": tag, "run": r.name, "arch": r.arch, "valid result": r.valid_as_result}
        hl = json.loads(r.metrics_json or "{}")
        d["best_epoch"] = hl.get("best_epoch")
        for split in ("val", "test"):
            for k in ("auroc", "accuracy", "sensitivity", "specificity", "f1", "brier"):
                if isinstance(hl.get(split), dict) and hl[split].get(k) is not None:
                    d[f"{split}_{k}"] = round(float(hl[split][k]), 4)
        tab.append(d)
    st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch")


def _ablation() -> None:
    runs = state.registry().runs("m6_validation") + state.registry().runs("m6_final")
    if not runs:
        theme.empty_state("No robust-training results yet", "Use the Robust Training Lab to run diagnose → train → validate (→ final).")
        return
    i = st.selectbox("Result set", range(len(runs)), format_func=lambda k: f"{'TEST (frozen selection)' if runs[k].kind == 'm6_final' else 'validation ablation'} · {runs[k].name} · {runs[k].created_utc[:16]}", key="abl_run")
    r = runs[i]
    out = Path(r.out_dir)
    wm = None if r.valid_as_result else r.watermark
    theme.watermark_banner(wm)
    pm = state.csv(out / "point_metrics.csv")
    ps = state.csv(out / "paired_diffs_summary.csv")
    pc = state.csv(out / "per_condition_auroc.csv")
    if pm is not None:
        st.markdown("**Clean, plausible-tier and stress-tier metrics per model (mean over seeds)**")
        agg = pm.groupby("intervention")[[c for c in POINT if c in pm]].mean().reset_index()
        st.dataframe(agg.round(4), hide_index=True, width="stretch")
        met = st.multiselect("Compare", [c for c in POINT if c in agg], default=[c for c in ("clean_auroc", "plaus_auroc", "plaus_flip_rate", "clean_ece") if c in agg], key="abl_met")
        if met:
            st.plotly_chart(charts.metric_bars(agg.rename(columns={"intervention": "model"}).to_dict("records"), met, "Baseline vs interventions (mean over seeds)", wm), width="stretch")
    if ps is not None:
        a, b = st.columns(2)
        mets = sorted(ps["metric"].unique())
        m1 = a.selectbox("Paired difference vs baseline", mets, index=mets.index("clean_auroc") if "clean_auroc" in mets else 0, key="abl_m")
        st.plotly_chart(charts.paired_forest(ps, m1, wm), width="stretch")
        st.dataframe(ps[ps["metric"] == m1].round(4), hide_index=True, width="stretch")
        if (ps["metric"] == "plaus_auroc").any() and (ps["metric"] == "clean_auroc").any():
            st.plotly_chart(charts.tradeoff_scatter(ps, wm), width="stretch")
    if pc is not None:
        mean = pc.groupby(["intervention", "condition", "level"])["auroc"].mean().reset_index()
        cond = st.selectbox("Per-condition AUROC", sorted(mean["condition"].unique()), key="abl_cond")
        import plotly.graph_objects as go
        fig = go.Figure()
        for j, (n, g) in enumerate(mean[mean["condition"] == cond].groupby("intervention")):
            fig.add_scatter(x=g["level"], y=g["auroc"], mode="lines+markers", name=n, line=charts._series_style(j), marker=dict(symbol=charts.MARKERS[j % len(charts.MARKERS)]))
        fig.update_layout(height=380, title=f"AUROC under {cond} by severity (mean over seeds)", xaxis_title="severity level", yaxis_title="AUROC")
        st.plotly_chart(charts._wm(fig, wm), width="stretch")
    rep = out / "report.md"
    if rep.exists():
        with st.expander("Written report"):
            st.markdown(rep.read_text())
    theme.limitations({"Interpretation": "Clean-image cost is reported next to robustness gain; an intervention that raises robustness while lowering clean AUROC is a trade-off, not an improvement.",
                       "Test split": "TEST numbers come from one frozen selection evaluated once; other variants are descriptive only."})
