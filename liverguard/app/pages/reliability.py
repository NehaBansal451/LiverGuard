"""Page 6: Reliability Center - calibration, uncertainty, selective prediction and failure cases from a scripts/reliability.py run."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from liverguard.app import charts, ops, state, theme

SCORES = {"entropy": "predictive entropy", "mc_std": "MC-Dropout std", "tta_std": "TTA std", "mc_mi": "MC mutual information"}


def render() -> None:
    w = state.setup("Reliability Center", "Is the model's confidence worth anything? Calibration, uncertainty and abstention, measured on held-out patients.")
    t_view, t_unc, t_fail, t_run = st.tabs(["Calibration", "Uncertainty & abstention", "Failure cases", "Run analysis"])
    run = state.select_run("reliability", "rel_run", "Reliability run")
    with t_view:
        if run is None:
            theme.empty_state("No reliability analysis yet", "Use the 'Run analysis' tab to compute calibration and uncertainty for a trained checkpoint.")
        else:
            _calibration(run)
    with t_unc:
        if run is not None:
            _uncertainty(run)
    with t_fail:
        if run is not None:
            _failures(run)
    with t_run:
        _launch(w)
    theme.limitations({"Calibration": "ECE depends on binning and is noisy on small sets; the bootstrap intervals (patient-level) show how noisy. A model can be calibrated on average and still be wrong on a subgroup.",
                       "Uncertainty": "MC-Dropout here perturbs only the classification head (backbone BN stays in eval), so it underestimates epistemic uncertainty. TTA variance measures sensitivity to the chosen augmentations.",
                       "Confidence": "A probability is the model's output, not clinical certainty. Abstention curves describe this dataset and split only."})
    state.finish()


def _wm(run) -> str | None:
    return None if run.valid_as_result else run.watermark


def _calibration(run) -> None:
    out = Path(run.out_dir)
    theme.watermark_banner(_wm(run))
    man = state.manifest(run)
    st.caption(f"Split **{run.split}** · threshold {(man.get('threshold') or {}).get('value')} ({(man.get('threshold') or {}).get('source')}) · temperature scaling {(man.get('temperature') or {}).get('mode')}, in-sample T={theme.fmt((man.get('temperature') or {}).get('in_sample_T'), 2)} · scope: {man.get('scope', '')}")
    t_in = (man.get("temperature") or {}).get("in_sample_T")
    if isinstance(t_in, (int, float)) and (t_in >= 19.5 or t_in <= 0.06):
        theme.banner(f"The fitted temperature T={t_in:.2f} sits at the edge of the search range [0.05, 20]: NLL is best at an extreme rescale, i.e. the model's probabilities carry little usable "
                     "information on this split (typical of a near-chance model). The 'calibrated' numbers are not meaningful.")
    cs = state.csv(out / "calibration_summary.csv")
    if cs is not None:
        cols = st.columns(4)
        r = cs.iloc[0]
        for c, (lab, k) in zip(cols, [("ECE", "ece_width"), ("Brier", "brier"), ("NLL", "nll"), ("Mean confidence", "mean_confidence")]):
            theme.kpi(c, lab, theme.fmt(r[k], 4), "uncalibrated probabilities")
        st.dataframe(cs.round(5), hide_index=True, width="stretch")
    a, b = st.columns(2)
    bins = state.csv(out / "reliability_bins.csv")
    pi = state.csv(out / "per_image_reliability.csv")
    if bins is not None:
        a.plotly_chart(charts.reliability_diagram(bins, _wm(run)), width="stretch")
    if pi is not None:
        b.plotly_chart(charts.confidence_histogram(pi, _wm(run)), width="stretch")
    os_ = state.csv(out / "outcome_summary.csv")
    if os_ is not None:
        st.markdown("**Uncertainty by outcome** (TP/TN/FP/FN)")
        st.dataframe(os_.round(4), hide_index=True, width="stretch")


def _uncertainty(run) -> None:
    out = Path(run.out_dir)
    theme.watermark_banner(_wm(run))
    rc = state.csv(out / "risk_coverage.csv")
    ss = state.csv(out / "selective_summary.csv")
    if rc is not None:
        st.plotly_chart(charts.risk_coverage_fig(rc, ss, _wm(run)), width="stretch")
    if ss is not None:
        st.markdown("**Selective-prediction summary** (AURC = area under risk-coverage; excess = share above the oracle ranking)")
        st.dataframe(ss.round(4), hide_index=True, width="stretch")
    eu = state.csv(out / "error_by_uncertainty.csv")
    if eu is not None:
        st.markdown("**Error rate by uncertainty quantile**")
        st.dataframe(eu.round(4), hide_index=True, width="stretch")
    pi = state.csv(out / "per_image_reliability.csv")
    if pi is not None:
        score = st.selectbox("Rank images by", [k for k in SCORES if k in pi], format_func=SCORES.get, key="rel_rank")
        top = pi.sort_values(score, ascending=False).head(int(st.slider("Show most uncertain", 5, 100, 20, key="rel_topn")))
        st.dataframe(top[["image_id", "patient_key", "label", "p", "confidence", score, "tta_flip", "correct"]].round(4), hide_index=True, width="stretch")


def _failures(run) -> None:
    out = Path(run.out_dir)
    theme.watermark_banner(_wm(run))
    fc = state.csv(out / "failure_cases.csv")
    if fc is None or fc.empty:
        st.success("No misclassified images in this run." if fc is not None else "failure_cases.csv not found.")
    else:
        st.markdown(f"**{len(fc)} misclassified images** (confident errors flagged)")
        st.dataframe(fc.round(4), hide_index=True, width="stretch")
    g = out / "figures" / "failure_gallery.png"
    if g.exists():
        st.image(str(g), caption="Failure gallery from the run", width="stretch")
    pl = state.csv(out / "patient_level.csv")
    if pl is not None:
        st.markdown("**Patient-level view** (images of one patient are not independent)")
        only = st.checkbox("Only patients with a wrong or disagreeing image", key="rel_pl_only")
        st.dataframe(pl[(~pl["patient_correct"]) | pl["images_disagree"] | (pl["n_img_errors"] > 0)] if only else pl, hide_index=True, width="stretch")


def _launch(w) -> None:
    run = state.select_checkpoint("rel_model")
    if run is None:
        return
    c1, c2 = st.columns(2)
    split = c1.selectbox("Split", ["val", "test"], key="rel_split")
    final = c2.checkbox("Final report (required for test; analyse test once)", key="rel_final") if split == "test" else False
    thr = state.threshold_choice("rel_thr")
    c1, c2 = st.columns(2)
    nb = c1.number_input("Bootstrap replicates", 0, 2000, 200, 50, key="rel_nb")
    mc = c2.number_input("MC-Dropout samples", 2, 100, 20, key="rel_mc")
    if st.button("Start reliability analysis", type="primary", key="rel_go", disabled=(split == "test" and not final)):
        out = w.root / "reliability" / f"{run.name}_{split}"
        state.start_job("reliability", f"reliability {run.name} {split}", ops.reliability_argv(w, run.checkpoint_path, out, split=split, threshold=thr,
                                                                                               n_boot=int(nb), mc_samples=int(mc), final_report=final), str(out))
    state.jobs_panel("reliability", key="jobs_rel")
