"""Page 4: Image Audit Studio - distort one image, run the real model on both versions, compare."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from liverguard.app import charts, ops, state, theme
from liverguard.app.pages import audit_results
from liverguard.robustness.engine import resolve_threshold


def pick_image(key: str, size: int):
    """Shared image picker (dataset or upload). Returns (PIL image at model size, image_id, true label or None, description) or None."""
    src = st.radio("Image source", ["Dataset image", "Upload"], horizontal=True, key=key + "_src")
    if src == "Upload":
        f = st.file_uploader("Ultrasound image (PNG/JPEG)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key=key + "_up")
        if f is None:
            theme.empty_state("Upload an image to begin", "Uploaded images are processed in memory and never stored.")
            return None
        data = f.getvalue()
        try:
            img = ops.open_rgb(__import__("io").BytesIO(data), size)
        except Exception as exc:                                           # noqa: BLE001
            st.error(f"Could not read this file as an image: {exc}")
            return None
        return img, "upload:" + hashlib.sha1(data).hexdigest()[:12], None, f"uploaded file {f.name}"
    idx = state.index_df()
    if idx is None:
        theme.empty_state("No dataset index", "Upload an image instead, or run scripts/verify_dataset.py.")
        return None
    c = st.columns(4)
    split = c[0].selectbox("Split", sorted(idx["split"].unique()), key=key + "_sp")
    if split == "test":
        c[0].caption("Viewing a test image is fine; nothing on this page selects or tunes a model.")
    cls = c[1].selectbox("Class", sorted(idx["class_label"].unique()), key=key + "_cl")
    d = idx[(idx["split"] == split) & (idx["class_label"] == cls)]
    if d.empty:
        theme.empty_state("No images for this selection")
        return None
    pat = c[2].selectbox("Patient", sorted(d["patient_key"].astype(str).unique()), key=key + "_pt")
    d = d[d["patient_key"].astype(str) == pat]
    row = d.iloc[c[3].selectbox("Image", range(len(d)), format_func=lambda i: f"{d.iloc[i]['variant']} · {d.iloc[i]['image_filename']}", key=key + "_im")]
    if not Path(row["path"]).exists():
        st.error(f"Image file not found: {row['path']}")
        return None
    return ops.open_rgb(row["path"], size), str(row["image_id"]), int(row["label"]), f"{row['class_label']} · patient {pat} · {row['variant']} · {split}"


def render() -> None:
    state.setup("Image Audit Studio", "Apply a controlled distortion to one image and see whether the model's answer changes")
    run = state.select_checkpoint("ia_model")
    if run is None:
        state.finish()
        return
    model, meta, dev = state.model_bundle(run.checkpoint_path)
    size = int(meta["image_size"])
    thr_choice = state.threshold_choice("ia_thr")
    try:
        thr, thr_src = resolve_threshold(thr_choice, Path(run.checkpoint_path))
    except (ValueError, FileNotFoundError) as exc:
        st.error(f"Threshold unavailable: {exc}")
        state.finish()
        return
    st.caption(f"Operating threshold {thr:.3f} (source: {thr_src}). Class 1 = HCC; p(HCC) = softmax probability.")

    t_one, t_ds = st.tabs(["Single image", "Dataset-level audit"])
    with t_one:
        _single(run, model, dev, size, thr)
    with t_ds:
        audit_results.render(state.ws(), run, thr_choice)
    state.finish()


def _single(run, model, dev, size, thr) -> None:
    picked = pick_image("ia", size)
    if picked is None:
        return
    img, image_id, label, desc = picked
    _train_image_warning(run, desc)
    rob = state.rob_cfg()
    c1, c2, c3 = st.columns([2, 2, 1])
    cond_name = c1.selectbox("Distortion", [c.name for c in rob.conditions], key="ia_cond")
    cond = next(c for c in rob.conditions if c.name == cond_name)
    level = c2.slider("Severity level", 1, len(cond.params), 1, key="ia_lvl", help="Levels above the plausible maximum are stress tests (a-priori, not clinically validated)")
    seed = c3.number_input("Seed", 0, 10**6, int(rob.seed), key="ia_seed")
    tier = rob.tier(level)
    st.markdown(theme.badge(f"level {level}: {tier}", "ok" if tier == "plausible" else "warn") + f"parameter = {cond.params[level - 1]:g} ({cond.func})", unsafe_allow_html=True)

    dist = ops.distort(img, cond, level, seed=int(seed), image_id=image_id)
    cmp = ops.compare_prediction(model, dev, img, dist, thr)
    fid = ops.fidelity_report(img, dist, cond, rob)
    theme.watermark_banner(None if run.valid_as_result else run.watermark)
    diff = np.abs(np.asarray(img, np.float32) - np.asarray(dist, np.float32)).mean(2)
    diff_img = Image.fromarray(np.clip(diff * (255 / max(diff.max(), 1)), 0, 255).astype(np.uint8))
    a, b, d = st.columns(3)
    a.image(img, caption=f"Original · {desc}", width="stretch")
    b.image(dist, caption=f"{cond_name}, level {level}", width="stretch")
    d.image(diff_img, caption=f"|difference| (stretched; max {diff.max():.0f}/255)", width="stretch")

    if cmp["flip"]:
        theme.banner(f"PREDICTION FLIP: {cmp['pred_original']} → {cmp['pred_distorted']} (p(HCC) {cmp['p_original']:.3f} → {cmp['p_distorted']:.3f}).", "bad")
    else:
        st.success(f"Prediction unchanged: {cmp['pred_original']} (p(HCC) {cmp['p_original']:.3f} → {cmp['p_distorted']:.3f}).")
    m = st.columns(4)
    m[0].metric("p(HCC) original", f"{cmp['p_original']:.3f}")
    m[1].metric("p(HCC) distorted", f"{cmp['p_distorted']:.3f}", f"{cmp['p_distorted'] - cmp['p_original']:+.3f}")
    m[2].metric("Confidence drift", f"{cmp['confidence_drift']:+.3f}", help="Change in max(p, 1-p)")
    m[3].metric("True label", ("HCC" if label == 1 else "Hemangioma") if label is not None else "unknown (upload)")
    if label is not None:
        ok0, ok1 = (cmp["p_original"] >= thr) == bool(label), (cmp["p_distorted"] >= thr) == bool(label)
        st.caption(f"Original prediction {'correct' if ok0 else 'wrong'} · distorted prediction {'correct' if ok1 else 'wrong'}")

    with st.expander("Does this distortion still leave the image diagnosable? (fidelity proxies)"):
        st.markdown(theme.badge("flagged: information may be destroyed" if fid["at_risk"] else "no automatic flag", "bad" if fid["at_risk"] else "ok"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{k: round(float(v), 3) for k, v in fid.items() if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)}]), hide_index=True)
        if fid["geometric_skips_pixel_metrics"]:
            st.caption("Rotation / crop move content, so PSNR and correlation (unaligned pixels) are not used; whether the lesion is still in view needs a human check.")
        st.caption("These are screening heuristics. Label preservation is an assumption that only expert review can confirm.")

    st.markdown("#### Severity sweep for this image")
    names = st.multiselect("Conditions", [c.name for c in rob.conditions], default=[c.name for c in rob.conditions][:4], key="ia_sweep_c")
    if st.button("Run sweep (real inference on every level)", disabled=not names):
        with st.spinner("Running the model on all distorted copies…"):
            st.session_state["ia_sweep"] = (image_id, ops.severity_sweep(model, dev, img, image_id, [c for c in rob.conditions if c.name in names], int(seed), thr), thr)
    sw = st.session_state.get("ia_sweep")
    if sw and sw[0] == image_id:
        df = sw[1]
        st.plotly_chart(charts.sweep_fig(df, sw[2], f"p(HCC) vs severity · {desc}", None if run.valid_as_result else "x"), width="stretch")
        first = df[df["flip"]].groupby("condition")["level"].min()
        st.dataframe(pd.DataFrame({"condition": [c for c in df["condition"].unique()],
                                   "first flipping level": [int(first[c]) if c in first.index else "none" for c in df["condition"].unique()]}), hide_index=True)
    theme.limitations(["One image proves nothing about robustness: use the dataset-level audit (Robust Training Lab / audit CLI) for rates and CIs.",
                       "Distortions are synthetic approximations applied after resizing; speckle is a simple multiplicative model.",
                       "A flip at a stress level is not a clinical finding: diagnostic information may be gone."])


def _train_image_warning(run, desc: str) -> None:
    """A model's own training images say nothing about robustness or accuracy: warn instead of letting that pass silently."""
    if desc.rstrip().endswith("train"):
        theme.banner("This image is in the TRAIN split: the model has seen it (and its patient's other images) during training, so its behaviour here is not evidence of generalisation.")
    cur = state.index_split_id()
    if run.split_id and cur and run.split_id != cur:
        theme.banner(f"The model was trained on split {run.split_id} but the current index defines split {cur}: the split labels shown for dataset images may not match this model's train/val/test partition.", "bad")
