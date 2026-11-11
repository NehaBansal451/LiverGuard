"""LiverGuard Streamlit entry point: streamlit run liverguard/app/app.py

Workspace: LIVERGUARD_WORKSPACE (default ./artifacts), LIVERGUARD_INDEX_CSV, LIVERGUARD_ROBUSTNESS_CONFIG.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from liverguard.app import state, theme  # noqa: E402
from liverguard.app.pages import (compare, dashboard, dataset, explain, image_audit, model_lab, reliability, reports, robust_lab)  # noqa: E402

PAGES = {
    "Overview": [("Research Dashboard", dashboard.render, ":material/dashboard:", "dashboard"),
                 ("Dataset Explorer", dataset.render, ":material/database:", "dataset")],
    "Build": [("Model Laboratory", model_lab.render, ":material/model_training:", "model-lab")],
    "Attack & diagnose": [("Image Audit Studio", image_audit.render, ":material/image_search:", "image-audit"),
                          ("Explainability Studio", explain.render, ":material/visibility:", "explainability"),
                          ("Reliability Center", reliability.render, ":material/verified:", "reliability")],
    "Fix & re-test": [("Robust Training Lab", robust_lab.render, ":material/build:", "robust-training"),
                      ("Model Comparison", compare.render, ":material/compare_arrows:", "comparison")],
    "Documentation": [("Research Reports", reports.render, ":material/description:", "reports")],
}


def main() -> None:
    st.set_page_config(
        page_title="LiverGuard",
        page_icon=":material/shield:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Light is the default. Streamlit keeps this selection in session state
    # while the user navigates between the research pages.
    if "lg_theme" not in st.session_state:
        st.session_state["lg_theme"] = "Light"

    with st.sidebar:
        st.markdown(
            '<div class="lg-brand">LiverGuard</div>'
            '<div class="lg-brand-sub">Liver ultrasound AI reliability audit</div>',
            unsafe_allow_html=True,
        )
        selected_theme = st.selectbox(
            "Appearance",
            options=["Light", "Dark"],
            key="lg_theme",
            help="Choose the appearance used across LiverGuard pages.",
        )

    theme.apply_theme(selected_theme)

    nav = st.navigation({
        sec: [
            st.Page(fn, title=title, icon=icon, url_path=url, default=(title == "Research Dashboard"))
            for title, fn, icon, url in items
        ]
        for sec, items in PAGES.items()
    })

    with st.sidebar:
        w = state.ws()
        counts = state.registry().counts()
        st.caption(f"Workspace: `{w.root}`")
        st.caption(" · ".join(f"{k} {v}" for k, v in counts.items()) or "No runs indexed")
        if st.button("Rescan workspace", key="sb_rescan", width="stretch"):
            result = state.rescan()
            st.toast(
                f"Indexed {result.get('indexed', 0)} manifest(s), "
                f"{len(result.get('errors', []))} error(s)"
            )
        st.markdown(f'<div class="lg-disc">{theme.DISCLAIMER}</div>', unsafe_allow_html=True)

    nav.run()


main()
