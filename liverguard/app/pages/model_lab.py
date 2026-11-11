"""Page 3: Model Laboratory - configure (validated by the real TrainConfig), start training as a background job, follow it, inspect results."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from liverguard.app import charts, ops, state, theme


def render() -> None:
    w = state.setup("Model Laboratory", "Configure and train EfficientNet-B0 / ResNet-50 on the official patient-level split; inspect learning curves, confusion matrix and checkpoints")
    t_train, t_results, t_ckpt = st.tabs(["Train", "Results", "Checkpoints"])

    with t_train:
        if state.index_df() is None:
            theme.empty_state("No image index", f"Training needs {w.index_csv} (scripts/verify_dataset.py).")
        c1, c2, c3 = st.columns(3)
        arch = c1.selectbox("Architecture", ["efficientnet_b0", "resnet50", "tiny_cnn"], help="tiny_cnn is for pipeline smoke tests only")
        kind = c2.selectbox("Experiment kind", ["baseline", "smoke"], help="'smoke' runs are watermarked and never valid as results")
        seed = c3.number_input("Seed", 0, 10**6, 42)
        d1, d2, d3, d4 = st.columns(4)
        epochs = d1.number_input("Epochs", 1, 200, 30)
        freeze = d2.number_input("Frozen-backbone epochs", 0, 50, 2)
        lr = d3.number_input("Learning rate", 1e-6, 1.0, 3e-4, format="%.6f")
        bs = d4.number_input("Batch size", 2, 512, 32)
        e1, e2, e3, e4 = st.columns(4)
        size = e1.selectbox("Image size", [64, 128, 224, 256, 320], index=2)
        pretrained = e2.checkbox("ImageNet pretrained", True, help="If weights cannot be loaded the run fails; it never silently starts from random weights.")
        control = e3.selectbox("Label control", ["none", "patient_shuffle"], help="patient_shuffle = null control: must score ~0.5 AUROC")
        device = e4.selectbox("Device", ["auto", "cpu", "cuda"])
        weights = st.text_input("Local pretrained weights file (offline use)", "")
        name = st.text_input("Run name", f"{arch}_s{seed}")
        eval_test = st.checkbox("Evaluate the TEST split when training ends", False, help="Leave off while experimenting: every look at test metrics can leak into your choices. "
                                "Turn on only for the final baseline run; the final test audit (with --final-report) does not need it.")
        with st.expander("More settings"):
            m1, m2, m3 = st.columns(3)
            patience = m1.number_input("Early-stopping patience", 1, 50, 8)
            wd = m2.number_input("Weight decay", 0.0, 1.0, 1e-4, format="%.6f")
            boot = m3.number_input("Bootstrap replicates (CI)", 0, 5000, 1000, 100)
        if st.button("Start training", type="primary", disabled=not (state.index_df() is not None)):
            params = dict(arch=arch, experiment_kind=kind, label_control=control, pretrained=pretrained, epochs=int(epochs), freeze_epochs=int(freeze), lr=float(lr),
                          batch_size=int(bs), image_size=int(size), seed=int(seed), device=device, patience=int(patience), weight_decay=float(wd),
                          bootstrap_samples=int(boot), pretrained_weights_path=weights or None, evaluate_test=bool(eval_test))
            try:
                ops.check_run_name(name)
                if (w.runs_dir / name / "manifest.json").exists():
                    raise ValueError(f"A finished run named {name!r} already exists; choose another name (runs are never overwritten).")
                if pretrained and not weights and arch != "tiny_cnn":
                    st.info("No local weights given: torchvision will try to download ImageNet weights; if that fails the run stops with an error.")
                cfg_path = ops.train_config_file(w, params, name)
            except Exception as exc:                                       # noqa: BLE001 - pydantic validation message for the user
                st.error(f"Invalid configuration: {exc}")
            else:
                out = w.runs_dir / name
                state.start_job("train", f"train {name}", ops.train_argv(w, cfg_path, out, seed=int(seed), label_control=control,
                                                                       weights=weights or None), str(out))
        st.markdown("#### Training jobs")
        _progress()
        state.jobs_panel("train", key="jobs_train")

    with t_results:
        run = state.select_run("train", "ml_run", "Training run")
        if run is None:
            theme.empty_state("No finished training run", "Start one on the Train tab; finished runs are indexed automatically after 'Index new results'.")
        else:
            _results(run)

    with t_ckpt:
        runs = state.registry().runs("train")
        if not runs:
            theme.empty_state("No checkpoints")
        else:
            rows = []
            for r in runs:
                p = Path(r.checkpoint_path) if r.checkpoint_path else None
                rows.append({"run": r.name, "arch": r.arch, "seed": r.seed, "split_id": r.split_id, "created": r.created_utc[:16],
                             "valid result": "yes" if r.valid_as_result else "NO", "file": str(p) if p else "", "exists": bool(p and p.exists()),
                             "size (MB)": round(p.stat().st_size / 1e6, 1) if p and p.exists() else None})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    state.finish()


def _progress() -> None:
    """Epoch progress parsed from the job log of the most recent running training job."""
    run_jobs = [j for j in state.registry().jobs(10) if j.kind == "train" and j.status == "running"]
    if not run_jobs:
        return
    j = run_jobs[0]
    m = None
    for line in reversed(ops.tail(j.log_path, 200).splitlines()):
        m = re.search(r"epoch (\d+)/(\d+) \[(\w+)\] train_loss=([\d.]+) val_loss=([\d.]+) val_auroc=([\d.]+)", line)
        if m:
            break
    if m:
        e, n = int(m.group(1)), int(m.group(2))
        st.progress(e / n, text=f"{j.label}: epoch {e}/{n} ({m.group(3)}) · train loss {m.group(4)} · val loss {m.group(5)} · val AUROC {m.group(6)}")
    else:
        st.progress(0.0, text=f"{j.label}: starting (loading data / weights)…")


def _results(run) -> None:
    man = state.manifest(run)
    theme.watermark_banner(None if run.valid_as_result else run.watermark)
    res = man.get("results", {})
    tr = man.get("training", {})
    c = st.columns(4)
    theme.kpi(c[0], "Architecture", run.arch, str(man.get("model", {}).get("weights_source", "")))
    theme.kpi(c[1], "Best epoch", f"{tr.get('best_epoch')} / {tr.get('epochs_run')}", f"selected on {tr.get('selected_on')}")
    for col, sp in zip(c[2:], ("val", "test")):
        r = res.get(sp)
        theme.kpi(col, f"{sp} AUROC", theme.fmt(r["auroc"]) if r else "not evaluated", f"n={r['n']} images" if r else "test split is evaluated once, at the end")
    out = Path(run.out_dir)
    h = state.csv(out / "history.csv")
    a, b = st.columns([3, 2])
    if h is not None:
        a.plotly_chart(charts.training_curves(h, None if run.valid_as_result else "x"), width="stretch")
    split = b.radio("Split for confusion matrix / ROC", [s for s in ("val", "test") if res.get(s)], horizontal=True, key="ml_split")
    r = res.get(split) if split else None
    if r and r.get("confusion_matrix"):
        b.plotly_chart(charts.confusion_matrix_fig(r["confusion_matrix"], title=f"Confusion matrix ({split}, threshold {r['threshold']:.2f})"), width="stretch")
    preds = state.csv(out / "predictions.csv")
    if preds is not None and split:
        pp = preds[preds["split"] == split]
        if pp["label"].nunique() == 2:
            st.plotly_chart(charts.roc_fig(pp["label"].to_numpy(), pp["p_hcc"].to_numpy(), r["auroc"] if r else None, f"ROC ({split})"), width="stretch")
    rows = []
    for sp in ("val", "test"):
        r = res.get(sp)
        if r:
            ci = (r.get("ci95_patient_bootstrap") or {}).get("auroc")
            rows.append({"split": sp, "n": r["n"], "AUROC": r["auroc"], "AUROC 95% CI (patients)": f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a", "accuracy": r["accuracy"],
                         "sensitivity": r["sensitivity"], "specificity": r["specificity"], "precision": r["precision"], "F1": r["f1"], "Brier": r["brier"]})
    if rows:
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    for k in ("val_at_youden", "test_at_val_youden"):
        if res.get(k):
            st.caption(f"{k}: sensitivity {theme.fmt(res[k].get('sensitivity'))}, specificity {theme.fmt(res[k].get('specificity'))} at threshold {theme.fmt(res[k].get('threshold'))} (fitted on validation only)")
    with st.expander("Configuration and data split"):
        st.json({"config": man.get("config"), "dataset": man.get("dataset"), "environment": man.get("environment")})
    theme.limitations(["Single seed per run: differences smaller than seed-to-seed variation are not findings.",
                       "The test split is evaluated once after validation-based model selection; never tune on it.",
                       "Images are resized to a square, which distorts the aspect ratio."])
