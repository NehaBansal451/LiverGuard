"""Dataset-level robustness audit: launch scripts/audit_robustness.py and read its outputs (degradation curves, ranking, worst case, flips)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from liverguard.app import charts, ops, state, theme

METRICS = {"d_auroc": "Change in AUROC", "pfr": "Prediction flip rate", "conf_drift": "Confidence drift", "d_accuracy": "Change in accuracy", "d_f1": "Change in F1",
           "mean_abs_dp": "Mean |Δp(HCC)|"}


def subsample_note(man: dict) -> str:
    """The audit manifest stores {'max_patients', 'complete'}; only complete=False means a patient sub-sample (the dict itself is always truthy)."""
    return "sub-sampled patients, not for final reporting" if (man.get("subsampled") or {}).get("complete") is False else "all patients of the split"


def render(w, run, thr_choice: str) -> None:
    mine = [r for r in state.registry().runs("audit") if run.checkpoint_sha256 and r.checkpoint_sha256 == run.checkpoint_sha256]
    if not mine:
        mine_names = [r for r in state.registry().runs("audit") if r.checkpoint_path and Path(r.checkpoint_path).resolve() == Path(run.checkpoint_path).resolve()]
        mine = mine_names
    st.markdown(f"**Audits of this checkpoint:** {len(mine)}")
    if mine:
        i = st.selectbox("Audit run", range(len(mine)), format_func=lambda k: f"{mine[k].name} · {mine[k].split} · {mine[k].created_utc[:16]}"
                         + ("" if mine[k].valid_as_result else "  [NOT A RESULT]"), key="au_run")
        _view(mine[i])
    else:
        theme.empty_state("No audit of this checkpoint yet", "Start one below: it distorts every image of the chosen split at every severity level and re-runs the model.")
    _launch(w, run, thr_choice)


def _view(r) -> None:
    out = Path(r.out_dir)
    wm = None if r.valid_as_result else r.watermark
    theme.watermark_banner(wm)
    man = state.manifest(r)
    s = state.csv(out / "summary_by_level.csv")
    cond = state.csv(out / "condition_ranking.csv")
    fam = state.csv(out / "family_worst_case.csv")
    if s is None:
        st.error("summary_by_level.csv is missing for this audit run.")
        return
    thr = man.get("threshold") or {}
    st.caption(f"Split **{r.split}** · threshold {thr.get('value')} ({thr.get('source')}) · {man.get('dataset', {}).get('n_images', '?')} images · "
               f"{subsample_note(man)}")
    clean = man.get("clean") or {}
    c = st.columns(4)
    for col, (lab, k) in zip(c, [("Clean AUROC", "auroc"), ("Clean accuracy", "accuracy"), ("Clean sensitivity", "sensitivity"), ("Clean specificity", "specificity")]):
        theme.kpi(col, lab, theme.fmt(clean.get(k)), "undistorted images, same split")
    wc = (man.get("worst_case") or {}).get("plausible_only", {}).get("auroc") or {}
    af = man.get("any_flip") or {}
    c = st.columns(3)
    theme.kpi(c[0], "Worst plausible-tier AUROC", theme.fmt(wc.get("value")), f"{wc.get('condition')} level {wc.get('level')}")
    theme.kpi(c[1], "Max flip rate (plausible tiers)", theme.fmt((man.get("worst_case") or {}).get("max_pfr_plausible")), "share of predictions that change")
    theme.kpi(c[2], "Images flipping under ≥1 plausible condition", theme.fmt(af.get("any_condition_plausible_levels")), "label-free instability")

    t1, t2, t3, t4 = st.tabs(["Degradation curves", "Most disruptive", "Heat maps", "Tables & figures"])
    with t1:
        metric = st.selectbox("Metric", list(METRICS), format_func=METRICS.get, key="au_metric")
        names = sorted(s["condition"].unique())
        sel = st.multiselect("Conditions", names, default=names[:5], key="au_conds")
        pmax = int(s[s["tier"] == "plausible"]["level"].max()) if (s["tier"] == "plausible").any() else 0
        if sel:
            st.plotly_chart(charts.degradation_curves(s, metric, sel, pmax, watermark=wm), width="stretch")
        st.caption("Error bars are paired patient-level bootstrap 95% intervals when the audit was run with bootstrap replicates; the shaded band marks stress levels "
                   "(a-priori tier: label preservation is not assured there).")
    with t2:
        if cond is not None:
            st.plotly_chart(charts.ranking_bar(cond, wm), width="stretch")
            show = [k for k in ("rank", "condition", "family", "mean_d_auroc_all", "mean_d_auroc_plausible", "mean_pfr_all", "pfr_at_plausible_max", "min_level_flip_alert",
                                "frac_images_ever_flipped", "mfs_median_label", "worst_auroc", "worst_auroc_plausible") if k in cond]
            st.dataframe(cond[show].round(3), hide_index=True, width="stretch")
            st.caption("MFS = minimum flip severity: the lowest level at which an image's prediction flips (median over images; '>5' = most never flip within the tested levels).")
    with t3:
        v = st.selectbox("Value", ["d_auroc", "pfr", "conf_drift"], format_func=METRICS.get, key="au_heat")
        st.plotly_chart(charts.condition_heatmap(s, v, f"{METRICS[v]} by condition and level", diverging=(v != "pfr"), watermark=wm), width="stretch")
    with t4:
        if fam is not None:
            st.markdown("**Worst case per distortion family**")
            st.dataframe(fam.round(3), hide_index=True, width="stretch")
        st.markdown("**Per condition and level**")
        st.dataframe(s.round(3), hide_index=True, width="stretch")
        flagged = s[s["fidelity_flag"]] if "fidelity_flag" in s else pd.DataFrame()
        if len(flagged):
            theme.banner(f"{len(flagged)} condition/level combinations are fidelity-flagged (the distortion may have destroyed diagnostic information); "
                         "degradation there is not evidence of model fragility.")
        figs = sorted((out / "figures").glob("*.png")) if (out / "figures").exists() else []
        for f in figs[:4]:
            st.image(str(f), caption=f.name, width="stretch")
    theme.limitations(man.get("assumptions", []) if isinstance(man.get("assumptions"), list) else [], title="Assumptions of this audit")


def _launch(w, run, thr_choice: str) -> None:
    with st.expander("Run a new dataset-level audit", expanded=False):
        rob = state.rob_cfg()
        c1, c2 = st.columns(2)
        split = c1.selectbox("Split", ["val", "test"], key="aul_split")
        final = c2.checkbox("Final report (required for test; audit test once)", key="aul_final") if split == "test" else False
        names = st.multiselect("Conditions (empty = all)", [c.name for c in rob.conditions], key="aul_conds")
        c1, c2, c3 = st.columns(3)
        nb = c1.number_input("Bootstrap replicates", 0, 2000, 200, 50, key="aul_nb")
        seed = c2.number_input("Seed", 0, 10**6, int(rob.seed), key="aul_seed")
        mp = c3.number_input("Max patients (0 = all)", 0, 10000, 0, key="aul_mp", help="Quick runs only; sub-sampled audits are flagged in the manifest")
        if st.button("Start audit", type="primary", key="aul_go", disabled=(split == "test" and not final)):
            out = w.root / "audit" / f"{run.name}_{split}_s{int(seed)}"
            state.start_job("audit", f"audit {run.name} {split}", ops.audit_argv(w, run.checkpoint_path, out, split=split, threshold=thr_choice, conditions=names or None,
                                                                                 n_boot=int(nb), seed=int(seed), final_report=final, max_patients=int(mp) or None), str(out))
        state.jobs_panel("audit", key="jobs_audit")
