"""Page 7: Robust Training Lab - diagnose -> targeted training -> validation ablation -> frozen selection -> one test re-audit (scripts/robust_train.py)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from liverguard.app import charts, ops, state, theme
from liverguard.improve.interventions import INTERVENTIONS

STAGE_HELP = {
    "diagnose": "Train the baseline(s), audit on VALIDATION, derive weaknesses and an intervention plan.",
    "train": "Train every planned intervention (same seeds, split, architecture, epochs; only the augmentation differs).",
    "validate": "Audit all models on VALIDATION, paired patient-bootstrap vs baseline, apply the pre-declared selection rule and freeze it.",
    "final": "One test-split re-audit of the frozen selection. Not repeatable without invalidating the protocol.",
}


def _status(wd: Path) -> list[tuple[str, bool, str]]:
    n_models = len(list((wd / "models").glob("*/manifest.json"))) if (wd / "models").exists() else 0
    return [("diagnose", (wd / "diagnosis" / "plan.json").exists(), "diagnosis/plan.json"),
            ("train", n_models > 0, f"{n_models} trained model(s)"),
            ("validate", (wd / "validation" / "selection.json").exists(), "validation/selection.json"),
            ("final", (wd / "final_test" / "final_manifest.json").exists(), "final_test/final_manifest.json")]


def render() -> None:
    w = state.setup("Robust Training Lab", "Fix what the audit found: targeted augmentation, ablated on validation, then one frozen test re-audit.")
    wd = Path(st.text_input("Experiment working directory", str(w.root / "m6"), key="rt_wd"))
    st.markdown("".join(theme.badge(f"{s}: {'done' if ok else 'pending'}", "ok" if ok else "info") + " " for s, ok, _ in _status(wd)), unsafe_allow_html=True)
    t_plan, t_run, t_res = st.tabs(["Diagnosis & plan", "Run stages", "Validation results"])
    with t_plan:
        _diagnosis(wd)
    with t_run:
        _run(w, wd)
    with t_res:
        _results(wd)
    state.finish()


def _diagnosis(wd: Path) -> None:
    rep, plan = wd / "diagnosis" / "weakness_report.md", wd / "diagnosis" / "plan.json"
    if not rep.exists():
        theme.empty_state("No diagnosis yet", "Run the 'diagnose' stage: it audits the baseline on validation and proposes which interventions to try.")
        return
    if plan.exists():
        p = json.loads(plan.read_text())
        st.markdown("**Weaknesses found:** " + (", ".join(f"`{x}`" for x in p.get("weaknesses", [])) or "none above threshold"))
        st.markdown("**Planned interventions:** " + ", ".join(f"`{x}`" for x in p.get("plan", [])))
    st.markdown(rep.read_text())


def _run(w, wd: Path) -> None:
    cfgs = sorted((w.repo / "configs").glob("*.yaml"))
    cfgs = [c for c in cfgs if c.name not in ("dataset.yaml", "robustness.yaml")]
    if not cfgs:
        theme.empty_state("No training configs in configs/")
        return
    cfg = st.selectbox("Baseline training config (every variant uses it)", cfgs, format_func=lambda p: p.name, key="rt_cfg")
    c1, c2, c3 = st.columns(3)
    seeds = c1.text_input("Seeds (>= 3 for a defensible claim)", "42,43,44", key="rt_seeds")
    nb = c2.number_input("Bootstrap replicates", 50, 2000, 200, 50, key="rt_nb")
    ep = c3.number_input("Epochs (0 = config value)", 0, 200, 0, key="rt_ep")
    plan = st.multiselect("Interventions (empty = use the diagnosed plan)", [k for k in INTERVENTIONS if k != "baseline"], key="rt_plan")
    weights = st.text_input("Pretrained weights file (optional, local)", key="rt_w")
    if not all(s.strip().isdigit() for s in seeds.split(",")):
        st.error("Seeds must be comma-separated integers.")
        return
    interventions = ",".join(["baseline"] + plan) if plan else None
    st.caption(f"Index: `{w.index_csv}` ({'found' if w.index_csv.exists() else 'MISSING'})")
    stat = {s: ok for s, ok, _ in _status(wd)}
    for stage in ("diagnose", "train", "validate"):
        a, b = st.columns([1, 4])
        if a.button(f"Run {stage}", key=f"rt_{stage}", disabled=not w.index_csv.exists()):
            state.start_job("m6", f"{stage} {wd.name}", ops.robust_argv(w, stage, wd, cfg, seeds, n_boot=int(nb), interventions=interventions,
                                                                      epochs=int(ep) or None, weights=weights or None), str(wd))
        b.caption(STAGE_HELP[stage] + ("  ✓ outputs exist (stage is resumable)" if stat[stage] else ""))
    st.markdown("---")
    ready = stat["validate"]
    ok = st.checkbox("I understand the frozen selection is evaluated once on the TEST split and this cannot be repeated without invalidating the result", key="rt_final_ok", disabled=not ready)
    a, b = st.columns([1, 4])
    if a.button("Run final test re-audit", key="rt_final", type="primary", disabled=not (ready and ok)):
        state.start_job("m6", f"final {wd.name}", ops.robust_argv(w, "final", wd, cfg, seeds, n_boot=int(nb), interventions=interventions, final_report=True), str(wd))
    b.caption(STAGE_HELP["final"] + ("" if ready else "  Locked until 'validate' has produced selection.json."))
    state.jobs_panel("m6", key="jobs_m6")


def _results(wd: Path) -> None:
    v = wd / "validation"
    if not (v / "selection.json").exists():
        theme.empty_state("No validation ablation yet")
        return
    sel = json.loads((v / "selection.json").read_text())
    st.markdown(f"**Frozen selection:** `{sel.get('chosen')}`  ·  rule: {sel.get('rule', {}).get('needs', '')}")
    cand = pd.DataFrame(sel.get("candidates", []))
    if len(cand):
        st.dataframe(cand.round(4), hide_index=True, width="stretch")
    ps = state.csv(v / "paired_diffs_summary.csv")
    wm = sel.get("watermark")
    if ps is not None:
        a, b = st.columns(2)
        a.plotly_chart(charts.paired_forest(ps, "clean_auroc", wm), width="stretch")
        if (ps["metric"] == "plaus_auroc").any():
            b.plotly_chart(charts.paired_forest(ps, "plaus_auroc", wm), width="stretch")
        st.plotly_chart(charts.tradeoff_scatter(ps, wm), width="stretch")
    pm = state.csv(v / "point_metrics.csv")
    if pm is not None:
        st.markdown("**Per-model validation metrics**")
        st.dataframe(pm.round(4), hide_index=True, width="stretch")
    rep = v / "report.md"
    if rep.exists():
        with st.expander("Full validation report"):
            st.markdown(rep.read_text())
    theme.limitations({"Attribution": "Plausible-tier AUROC differences are descriptive: an intervention trains on the same distortions it is tested on, and several seeds on one split are not independent samples of the population.",
                       "Selection": "The selection is made on validation only; the test re-audit evaluates that frozen choice and cannot change it."})
