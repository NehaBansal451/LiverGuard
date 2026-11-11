"""Page 2: Dataset Explorer - works on the real image index written by scripts/verify_dataset.py and the files it points to."""
from __future__ import annotations

import json
from pathlib import Path

import plotly.express as px
import streamlit as st

from liverguard.app import ops, state, theme


def render() -> None:
    w = state.setup("Dataset Explorer", "Browse SMC-LUD images, check class/patient structure, compare Clean and Caliper images, inspect data quality")
    idx = state.index_df()
    if idx is None:
        theme.empty_state("No image index found", f"Expected {w.index_csv}. Run scripts/verify_dataset.py --data-root <SMC-LUD> --check-images (see Verify below).")
        _verify_panel(w)
        state.finish()
        return
   idx = idx.copy()
idx["split"] = idx["split"].fillna("unknown").astype(str)
idx["variant"] = idx["variant"].fillna("unknown").astype(str)
o = ops.dataset_overview(idx)
    t_over, t_browse, t_domain, t_quality, t_verify = st.tabs(["Overview", "Browse images", "Clean vs Caliper", "Data quality", "Verify / re-index"])

    with t_over:
        c = st.columns(4)
        theme.kpi(c[0], "Images", f"{o['n_images']:,}")
        theme.kpi(c[1], "Patients", f"{o['n_patients']:,}")
        theme.kpi(c[2], "Median images / patient", f"{o['images_per_patient']['50%']:.0f}", f"max {o['images_per_patient']['max']:.0f}")
        theme.kpi(c[3], "Variants", " / ".join(f"{k} {v:,}" for k, v in o["by_variant"].items()))
        a, b = st.columns(2)
        cs = idx.groupby(["split", "class_label"]).size().reset_index(name="images")
        a.plotly_chart(px.bar(cs, x="split", y="images", color="class_label", barmode="group", title="Images by split and class",
                              color_discrete_sequence=theme.SERIES), width="stretch")
        ps = idx.groupby(["split", "class_label"])["patient_key"].nunique().reset_index(name="patients")
        b.plotly_chart(px.bar(ps, x="split", y="patients", color="class_label", barmode="group", title="Patients by split and class",
                              color_discrete_sequence=theme.SERIES), width="stretch")
        per = idx.groupby("patient_key").size().reset_index(name="images")
        st.plotly_chart(px.histogram(per, x="images", nbins=30, title="Images per patient", color_discrete_sequence=[theme.SERIES[0]]), width="stretch")
        leak = idx.groupby(["variant", "patient_key"])["split"].nunique()
        n_leak = int((leak > 1).sum())
        st.markdown(theme.badge("no patient in >1 split (per variant)" if n_leak == 0 else f"{n_leak} patient(s) span >1 split within a variant", "ok" if n_leak == 0 else "bad"),
                    unsafe_allow_html=True)
        rep = Path(w.dataset_dir) / "dataset_report.json"
        if rep.exists():
            d = json.loads(rep.read_text())
            with st.expander("Dataset verification report (scripts/verify_dataset.py)"):
                st.markdown(f"Overall: {'**OK**' if d.get('ok') else '**errors found**'} · split_id `{d.get('split_id')}` · data root `{d.get('data_root')}`")
                for r in d.get("reports", []):
                    st.markdown(f"**{r['title']}** — {'ok' if r.get('ok') else 'problems'}")
                    for i in r.get("issues", [])[:20]:
                        st.markdown(f"- {i if isinstance(i, str) else json.dumps(i)}")
                st.markdown("**Known documentation discrepancies**")
                for x in d.get("known_documentation_discrepancies", []):
                    st.markdown(f"- {x}")

    with t_browse:
        f1, f2, f3, f4 = st.columns(4)
        splits = f1.multiselect("Split", sorted(idx["split"].unique()), default=["train"] if "train" in set(idx["split"]) else None)
        classes = f2.multiselect("Class", sorted(idx["class_label"].unique()), default=sorted(idx["class_label"].unique()))
        variants = f3.multiselect("Variant", sorted(idx["variant"].unique()), default=sorted(idx["variant"].unique()))
        pat = f4.text_input("Patient id contains", "")
        d = idx[idx["class_label"].isin(classes) & idx["variant"].isin(variants)]
        if splits:
            d = d[d["split"].isin(splits)]
        if pat:
            d = d[d["patient_key"].astype(str).str.contains(pat, case=False)]
        st.caption(f"{len(d):,} images match" + (" · test split images are for viewing only" if "test" in splits else ""))
        per_page, cols_n = 12, 4
        pages = max(1, -(-len(d) // per_page))
        pg = st.number_input("Page", 1, pages, 1, key="ds_page")
        chunk = d.iloc[(pg - 1) * per_page: pg * per_page]
        grid = st.columns(cols_n)
        for i, row in enumerate(chunk.itertuples()):
            with grid[i % cols_n]:
                if Path(row.path).exists():
                    st.image(row.path, width="stretch", caption=f"{row.class_label} · patient {row.patient_key} · {row.variant} · {row.split}")
                else:
                    st.warning(f"missing file: {row.path}")
        if len(chunk):
            sel = st.selectbox("Metadata of image", chunk["image_id"].tolist())
            st.json({k: (v if not hasattr(v, "item") else v.item()) for k, v in idx[idx["image_id"] == sel].iloc[0].to_dict().items()})

    with t_domain:
        st.markdown("Clean images carry no on-screen annotations; Caliper images have measurement markers drawn on them. Domain-shift analyses must keep each patient inside one split (patient-complete pairs).")
        var = idx.groupby(["variant", "split"]).size().reset_index(name="images")
        st.plotly_chart(px.bar(var, x="variant", y="images", color="split", title="Images per variant and split", color_discrete_sequence=theme.SERIES), width="stretch")
        pv = idx.groupby("patient_key")["variant"].nunique()
        both = pv[pv > 1].index
        st.markdown(f"**{len(both):,} patients** have images in more than one variant.")
        if len(both):
            pick = st.selectbox("Patient with several variants", sorted(map(str, both))[:500])
            g = idx[idx["patient_key"].astype(str) == pick].sort_values("variant")
            cc = st.columns(min(4, len(g)) or 1)
            for i, r in enumerate(g.head(8).itertuples()):
                if Path(r.path).exists():
                    cc[i % len(cc)].image(r.path, caption=f"{r.variant} · {r.split}", width="stretch")
            splits_per = idx[idx["patient_key"].isin(both)].groupby("patient_key")["split"].nunique()
            st.markdown(theme.badge(f"{int((splits_per > 1).sum())} multi-variant patients span >1 split", "warn" if (splits_per > 1).any() else "ok"), unsafe_allow_html=True)
        else:
            theme.empty_state("This index has one variant only", "The Clean-vs-Caliper comparison needs both variants indexed.")

    with t_quality:
        st.caption("Measured on the image files (a seeded random sample for speed).")
        n = st.slider("Images to measure", 20, min(1000, len(idx)), min(200, len(idx)), 10) if len(idx) > 20 else len(idx)
        if st.button("Measure image quality", type="primary"):
            with st.spinner("Reading images…"):
                st.session_state["qual"] = ops.image_stats(idx["path"].tolist(), max_n=n)
        q = st.session_state.get("qual")
        if q is None:
            sp = Path(w.dataset_dir) / "image_stats.csv"
            if sp.exists():
                st.caption("Showing image_stats.csv written by verify_dataset.py")
                q = state.csv(sp)
        if q is not None and len(q):
            if "ok" in q:
                bad = int((~q["ok"].astype(bool)).sum())
                st.markdown(theme.badge(f"{bad} unreadable file(s)", "bad" if bad else "ok"), unsafe_allow_html=True)
            num = q.select_dtypes("number")
            if {"width", "height"} <= set(q.columns):
                a, b = st.columns(2)
                a.plotly_chart(px.scatter(q, x="width", y="height", title="Image dimensions", color_discrete_sequence=[theme.SERIES[0]]), width="stretch")
                col = "mean" if "mean" in q else num.columns[0]
                b.plotly_chart(px.histogram(q, x=col, nbins=30, title="Mean grey level", color_discrete_sequence=[theme.SERIES[1]]), width="stretch")
            st.dataframe(q.describe().T.round(2), width="stretch")
    with t_verify:
        _verify_panel(w)
    state.finish()


def _verify_panel(w) -> None:
    st.markdown("Run the dataset verification (indexing, metadata validation, patient-level split check, duplicates) as a background job.")
    root = st.text_input("SMC-LUD data root", key="ds_root")
    c1, c2 = st.columns(2)
    chk, hsh = c1.checkbox("Decode every image (--check-images)", True), c2.checkbox("Hash images for duplicates (--hash)", True)
    if st.button("Start verification", type="primary", disabled=not root):
        if not Path(root).is_dir():
            st.error(f"Not a directory: {root}")
        else:
            import sys
            argv = [sys.executable, w.script("verify_dataset.py"), "--data-root", root, "--out-dir", str(w.dataset_dir)] + (["--check-images"] if chk else []) + (["--hash"] if hsh else [])
            state.start_job("verify", "dataset verification", argv, str(w.dataset_dir))
    state.jobs_panel("verify", key="jobs_verify")
