"""Page 9: Research Reports - experiment history, manifest viewer and downloadable PDF reports built from registered runs."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from liverguard.app import state, theme
from liverguard.reports.pdf import build_report, run_to_dict

KINDS = {"train": "Training", "audit": "Robustness audit", "reliability": "Reliability", "explain": "Explainability",
         "m6_validation": "Robust-training validation", "m6_final": "Robust-training test re-audit"}


def render() -> None:
    w = state.setup("Research Reports", "Every experiment on record, and PDF reports assembled only from what the runs actually wrote.")
    reg = state.registry()
    t_hist, t_build, t_past = st.tabs(["Experiment history", "Build a report", "Past reports"])
    with t_hist:
        _history(reg)
    with t_build:
        _build(w, reg)
    with t_past:
        _past(reg)
    state.finish()


def _history(reg) -> None:
    runs = reg.runs()
    if not runs:
        theme.empty_state("No experiments registered", "Run something, or point LIVERGUARD_WORKSPACE at a workspace with runs and press 'Rescan' in the sidebar.")
        return
    c1, c2, c3 = st.columns(3)
    kinds = c1.multiselect("Kind", list(KINDS), format_func=KINDS.get, key="rh_kind")
    only_valid = c2.checkbox("Only valid results (hide smoke / null runs)", key="rh_valid")
    q = c3.text_input("Name contains", key="rh_q")
    rows = [r for r in runs if (not kinds or r.kind in kinds) and (not only_valid or r.valid_as_result) and q.lower() in r.name.lower()]
    df = pd.DataFrame([{"id": r.id, "kind": KINDS.get(r.kind, r.kind), "name": r.name, "arch": r.arch, "split": r.split, "seed": r.seed,
                        "valid result": r.valid_as_result, "created": r.created_utc[:19]} for r in rows])
    st.caption(f"{len(rows)} of {len(runs)} runs")
    st.dataframe(df, hide_index=True, width="stretch")
    if rows:
        i = st.selectbox("Inspect run", range(len(rows)), format_func=lambda k: f"#{rows[k].id} {KINDS.get(rows[k].kind)} · {rows[k].name}", key="rh_sel")
        r = rows[i]
        theme.watermark_banner(None if r.valid_as_result else r.watermark)
        a, b = st.columns(2)
        a.markdown("**Headline numbers (copied from the manifest)**")
        a.json(json.loads(r.metrics_json or "{}"))
        b.markdown("**Locations**")
        b.code(f"manifest: {r.manifest_path}\noutput:   {r.out_dir}\ncheckpoint sha256: {r.checkpoint_sha256 or 'n/a'}\nsplit id: {r.split_id or 'n/a'}", language="text")
        with st.expander("Full manifest (configuration, environment, limitations)"):
            st.json(state.manifest(r))
        figs = sorted((Path(r.out_dir) / "figures").glob("*.png"))[:6] if (Path(r.out_dir) / "figures").exists() else []
        if figs:
            cols = st.columns(min(3, len(figs)))
            for k, f in enumerate(figs):
                cols[k % len(cols)].image(str(f), caption=f.name, width="stretch")


def _build(w, reg) -> None:
    runs = reg.runs()
    if not runs:
        theme.empty_state("Nothing to report on yet")
        return
    title = st.text_input("Report title", "LiverGuard experiment report", key="rp_title")
    sel = st.multiselect("Runs to include", [r.id for r in runs], format_func=lambda i: next(f"#{r.id} {KINDS.get(r.kind)} · {r.name}" + ("" if r.valid_as_result else "  [NOT A RESULT]") for r in runs if r.id == i), key="rp_runs")
    note = st.text_area("Notes (appear in the report; plain text)", key="rp_note")
    chosen = [r for r in runs if r.id in sel]
    bad = [r for r in chosen if not r.valid_as_result]
    if bad:
        theme.banner(f"{len(bad)} selected run(s) are smoke/synthetic/null runs. The PDF will say so on its first page and in every section.")
    if st.button("Generate PDF report", type="primary", disabled=not chosen, key="rp_go"):
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_") or "report"
        path = w.reports_dir / f"{name}_{pd.Timestamp.now('UTC').strftime('%Y%m%d_%H%M%S')}.pdf"
        try:
            with st.spinner("Rendering report…"):
                build_report(path, title, [run_to_dict(r) for r in chosen], note)
            reg.add_report(title, path, [r.id for r in chosen], bool(bad))
            st.session_state["rp_last"] = str(path)
        except Exception as exc:  # surface the real error; nothing is faked
            st.error(f"Report generation failed: {exc}")
    last = st.session_state.get("rp_last")
    if last and Path(last).exists():
        st.success(f"Report written: {Path(last).name} ({Path(last).stat().st_size / 1024:.0f} KB)")
        st.download_button("Download PDF", Path(last).read_bytes(), file_name=Path(last).name, mime="application/pdf", key="rp_dl")


def _past(reg) -> None:
    recs = reg.reports()
    if not recs:
        theme.empty_state("No reports generated yet")
        return
    for r in recs:
        p = Path(r.path)
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{r.title}** · {r.created:%Y-%m-%d %H:%M} · runs {r.run_ids} · {r.size_bytes / 1024:.0f} KB"
                    + (" · " + theme.badge("contains non-results", "warn") if r.contains_non_results else ""), unsafe_allow_html=True)
        if p.exists():
            c2.download_button("PDF", p.read_bytes(), file_name=p.name, mime="application/pdf", key=f"pr_{r.id}")
        else:
            c2.caption("file missing")
