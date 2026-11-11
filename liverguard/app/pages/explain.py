"""Page 5: Explainability Studio - Grad-CAM, occlusion, ROI/background masking on one image, plus dataset-level explainability runs."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st

from liverguard.app import charts, ops, state, theme
from liverguard.app.pages.image_audit import pick_image
from liverguard.explain import masking

LIMITS = {
    "Grad-CAM": "Low-resolution map of where the last conv layer's evidence for one class lies. It is not proof of medical reasoning and can resemble image structure even for an untrained network (see the randomisation check in the dataset-level tab).",
    "Occlusion": "Replaces a patch with the image mean and measures the probability drop. Resolution = patch size; the grey patch is out-of-distribution; one patch cannot reveal redundant evidence.",
    "ROI / background masking": "Everything outside (or inside) the region is filled with the image mean, which creates out-of-distribution inputs; a region that 'carries information' is not shown to be what a clinician would use. The centre proxy is NOT the liver or the lesion.",
}


def render() -> None:
    w = state.setup("Explainability Studio", "Where does the model look, and does the prediction survive removing the region it should rely on?")
    run = state.select_checkpoint("ex_model")
    if run is None:
        state.finish()
        return
    model, meta, dev = state.model_bundle(run.checkpoint_path)
    size = int(meta["image_size"])
    t_one, t_cmp, t_ds = st.tabs(["Single image", "Side-by-side comparison", "Dataset-level analysis"])
    with t_one:
        picked = pick_image("ex", size)
        if picked is None:
            _dataset_tab(w, run, t_ds)
            state.finish()
            return
        img, image_id, label, desc = picked
        st.session_state["ex_picked"] = (image_id, desc)
        st.caption(desc + (f" · true label {'HCC' if label == 1 else 'Hemangioma'}" if label is not None else ""))
        g, o, r = st.tabs(["Grad-CAM", "Occlusion sensitivity", "ROI vs background"])
        with g:
            tgt = st.radio("Class explained", ["predicted class", "HCC", "Hemangioma"], horizontal=True, key="ex_tgt")
            alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.45, 0.05, key="ex_alpha")
            t = {"predicted class": "predicted", "HCC": 1, "Hemangioma": 0}[tgt]
            cam, probs = ops.gradcam(model, img, size, dev, t)
            a, b, c = st.columns(3)
            a.image(img, caption=f"Original · p(HCC) {probs[1]:.3f}", width="stretch")
            b.image(ops.overlay(img, cam, 1.0), caption=f"Grad-CAM heat map ({tgt})", width="stretch")
            c.image(ops.overlay(img, cam, alpha), caption="Overlay", width="stretch")
            if cam.max() == 0:
                st.info("The map is empty: the last conv layer carries no positive evidence for this class on this image.")
            st.caption(f"Map normalised per image (peak = 1); mass in the top 20% of pixels: {np.sort(cam.ravel())[::-1][: cam.size // 5].sum() / max(cam.sum(), 1e-9):.0%}.")
            theme.limitations({"Grad-CAM": LIMITS["Grad-CAM"]})
        with o:
            c1, c2, c3 = st.columns(3)
            patch = c1.slider("Patch size (px at model resolution)", max(4, size // 16), size // 2, max(8, size // 4), 2, key="ex_patch")
            stride = c2.slider("Stride", max(2, size // 32), patch, max(4, patch // 2), 2, key="ex_stride")
            tcls = c3.radio("Class", ["HCC", "Hemangioma"], horizontal=True, key="ex_ocls")
            n_pass = (-(-(size - patch) // stride) + 1) ** 2
            st.caption(f"{n_pass} forward passes.")
            if st.button("Compute occlusion map", key="ex_occ_btn"):
                with st.spinner("Occluding patches…"):
                    st.session_state["ex_occ"] = (image_id, ops.occlusion(model, img, size, dev, 1 if tcls == "HCC" else 0, patch, stride), tcls)
            occ = st.session_state.get("ex_occ")
            if occ and occ[0] == image_id:
                (m, base), cl = occ[1], occ[2]
                a, b = st.columns(2)
                a.plotly_chart(charts.occlusion_heat(m, title=f"Occlusion ({cl})"), width="stretch")
                pos = np.clip(m, 0, None)
                b.image(ops.overlay(img, pos / max(pos.max(), 1e-9), 0.5), caption=f"Regions whose removal lowers p({cl}) (base p {base:.3f})", width="stretch")
            theme.limitations({"Occlusion": LIMITS["Occlusion"]})
        with r:
            roi, src = _roi(img, image_id, size)
            if roi is not None:
                st.image(ops.mask_figure_image(img, roi), caption=f"Region of interest ({src}); outside darkened", width=260)
                n_ctl = st.slider("Random-placement controls", 1, 10, 5, key="ex_nctl")
                if st.button("Run ROI vs background experiment", key="ex_roi_btn"):
                    with st.spinner("Predicting on masked versions…"):
                        st.session_state["ex_roi"] = (image_id, ops.roi_experiment(model, dev, img, roi, 0, n_ctl), src)
                res = st.session_state.get("ex_roi")
                if res and res[0] == image_id:
                    st.plotly_chart(charts.roi_bars(res[1], 0.5), width="stretch")
                    st.dataframe(res[1].round(3), hide_index=True, width="stretch")
                    st.caption("Compare each region with its matched-area control: only a difference between them says something about the region itself.")
                    if "proxy" in res[2]:
                        theme.banner("The ROI here is a centred geometric proxy, not the liver or the lesion. Supply annotation masks for anatomical experiments.")
            theme.limitations({"ROI / background masking": LIMITS["ROI / background masking"]})
    with t_cmp:
        _compare(model, dev, size, run)
    with t_ds:
        _dataset_tab(w, run, None)
    state.finish()


def _roi(img, image_id: str, size: int):
    mode = st.radio("Region source", ["Centre proxy (geometric)", "Upload a mask", "Annotation directory"], horizontal=True, key="ex_roi_src")
    if mode.startswith("Centre"):
        area = st.slider("Area fraction", 0.1, 0.8, 0.4, 0.05, key="ex_area")
        return masking.proxy_roi(size, area), f"centre proxy, {area:.0%} of the image"
    if mode.startswith("Upload"):
        f = st.file_uploader("Binary mask image (non-zero = region)", type=["png", "jpg", "bmp"], key="ex_mask_up")
        if f is None:
            return None, ""
        try:
            return ops.load_uploaded_mask(io.BytesIO(f.getvalue()), size), "uploaded mask"
        except ValueError as exc:
            st.error(str(exc))
            return None, ""
    d = st.text_input("Directory of masks (named by the image's relative path)", key="ex_mask_dir")
    if not d:
        return None, ""
    try:
        m = masking.load_annotation_mask(d, image_id, size)
    except ValueError as exc:
        st.error(str(exc))
        return None, ""
    if m is None:
        st.warning("No mask found for this image in that directory.")
        return None, ""
    return m, "annotation mask"


def _compare(model, dev, size: int, run) -> None:
    pk = st.session_state.get("ex_picked")
    if not pk:
        theme.empty_state("Pick an image on the 'Single image' tab first")
        return
    st.caption(f"Image: {pk[1]}")
    st.markdown("Left to right: original · Grad-CAM (predicted class) · occlusion (HCC) · ROI only · background only (centre-proxy ROI).")
    area = st.slider("Proxy ROI area", 0.1, 0.8, 0.4, 0.05, key="cmp_area")
    picked = pick_image("cmpx", size)
    if picked is None:
        return
    img, image_id, _, desc = picked
    if st.button("Build comparison", type="primary", key="cmp_btn"):
        with st.spinner("Computing Grad-CAM, occlusion and masked predictions…"):
            cam, probs = ops.gradcam(model, img, size, dev, "predicted")
            occ, base = ops.occlusion(model, img, size, dev, 1, max(8, size // 4), max(4, size // 8))
            roi = masking.proxy_roi(size, area)
            v = masking.region_variants(img, roi, np.random.default_rng(0), n_control=1)
            p = ops.p_hcc(model, [img, v["roi_only"][0], v["background_only"][0]], dev)
            st.session_state["cmp_res"] = (image_id, cam, occ, base, v["roi_only"][0], v["background_only"][0], p)
    res = st.session_state.get("cmp_res")
    if res and res[0] == image_id:
        _, cam, occ, base, ro, bo, p = res
        cols = st.columns(5)
        cols[0].image(img, caption=f"original p(HCC) {p[0]:.3f}", width="stretch")
        cols[1].image(ops.overlay(img, cam, 0.5), caption="Grad-CAM", width="stretch")
        pos = np.clip(occ, 0, None)
        cols[2].image(ops.overlay(img, pos / max(pos.max(), 1e-9), 0.5), caption=f"occlusion (max Δp {pos.max():.2f})", width="stretch")
        cols[3].image(ro, caption=f"ROI only p(HCC) {p[1]:.3f}", width="stretch")
        cols[4].image(bo, caption=f"background only p(HCC) {p[2]:.3f}", width="stretch")
        theme.banner("Centre-proxy ROI: not the liver or the lesion. Agreement between methods is suggestive, disagreement is informative; none of them proves what the model 'uses'.")
        theme.watermark_banner(None if run.valid_as_result else run.watermark)


def _dataset_tab(w, run, _unused) -> None:
    st.markdown("Dataset-level explainability: Grad-CAM / masking / CAM-deletion over a whole split with matched random controls (scripts/explain.py).")
    er = state.select_run("explain", "ex_run", "Existing explainability run")
    if er is not None:
        theme.watermark_banner(None if er.valid_as_result else er.watermark)
        out = Path(er.out_dir)
        man = state.manifest(er)
        st.markdown(f"Split **{er.split}** · region source: {man.get('roi_source', 'n/a')}")
        ms = state.csv(out / "masking_summary.csv")
        if ms is not None:
            st.plotly_chart(charts.masking_bar(ms, None if er.valid_as_result else "x"), width="stretch")
            st.dataframe(ms.round(3), hide_index=True, width="stretch")
        cf = state.csv(out / "cam_faithfulness.csv")
        if cf is not None:
            st.markdown("**Grad-CAM deletion / insertion vs matched random regions** (evidence drop = p(class) before − after)")
            st.dataframe(cf.round(3), hide_index=True, width="stretch")
        rz = man.get("gradcam_randomization") or {}
        if rz:
            st.markdown("**Grad-CAM vs a randomised copy of the network** (high correlation = map mostly reflects the image, not learned weights)")
            st.json(rz)
        cases = out / "figures" / "case_panels.png"
        if cases.exists():
            st.image(str(cases), caption="Case panels: most-wrong errors and randomly drawn correct cases (no cherry-picking)", width="stretch")
        lim = man.get("limitations")
        if isinstance(lim, dict):
            theme.limitations(lim)
    else:
        theme.empty_state("No dataset-level explainability run yet")
    with st.expander("Run a new dataset-level analysis"):
        split = st.selectbox("Split", ["val", "test"], key="exr_split")
        final = st.checkbox("Final report (required for the test split; use once)", key="exr_final") if split == "test" else False
        thr = state.threshold_choice("exr_thr")
        nb = st.number_input("Bootstrap replicates", 0, 2000, 200, 50, key="exr_nb")
        md = st.text_input("Annotation mask directory (optional; otherwise centre-proxy ROI)", key="exr_md")
        if st.button("Start analysis", key="exr_go", type="primary", disabled=(split == "test" and not final)):
            out = w.root / "explain" / f"{run.name}_{split}"
            state.start_job("explain", f"explain {run.name} {split}", ops.explain_argv(w, run.checkpoint_path, out, split=split, threshold=thr, n_boot=int(nb),
                                                                                       mask_dir=md or None, final_report=final), str(out))
        state.jobs_panel("explain", key="jobs_explain")
