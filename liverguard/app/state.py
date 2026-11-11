"""Streamlit glue: cached registry/model/CSV access and shared page chrome. Business logic lives in ops.py and the ML modules."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from liverguard.app import ops, theme
from liverguard.app.config import Workspace, get_workspace
from liverguard.db import Registry
from liverguard.robustness.config import RobustnessConfig, load_robustness_config


def ws() -> Workspace:
    return get_workspace()


@st.cache_resource
def _registry(db_path: str) -> Registry:
    return Registry(db_path)


def registry() -> Registry:
    return _registry(str(ws().db_path))


def rescan() -> dict:
    res = registry().scan(ws().root)
    st.session_state["scan"] = res
    return res


def setup(title: str, subtitle: str) -> Workspace:
    """Common page start: theme, header, one workspace scan per browser session."""
    theme.apply_theme()
    if "scan" not in st.session_state:
        rescan()
    theme.page_header(title, subtitle)
    errs = st.session_state.get("scan", {}).get("errors", [])
    if errs:
        with st.expander(f"{len(errs)} manifest(s) could not be indexed"):
            for p, m in errs:
                st.code(f"{p}\n{m}")
    return ws()


def finish() -> None:
    theme.footer()


@st.cache_data(show_spinner=False)
def read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def csv(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    return read_csv(str(p), p.stat().st_mtime) if p.exists() else None


@st.cache_data(show_spinner=False)
def _json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


def manifest(run) -> dict:
    p = Path(run.manifest_path)
    return _json(str(p), p.stat().st_mtime) if p.exists() else {}


def index_df() -> pd.DataFrame | None:
    return csv(ws().index_csv)


@st.cache_resource(show_spinner="Loading model…")
def _model(ckpt: str, mtime: float, device: str):
    return ops.load_model(ckpt, device)


def model_bundle(ckpt: str, device: str = "auto"):
    p = Path(ckpt)
    return _model(str(p), p.stat().st_mtime, device)


def rob_cfg() -> RobustnessConfig:
    return load_robustness_config(ws().robustness_config)


def checkpoint_runs() -> list:
    """Training runs whose checkpoint file exists."""
    return [r for r in registry().runs("train") if r.checkpoint_path and Path(r.checkpoint_path).exists()]


def run_label(r) -> str:
    tag = "" if r.valid_as_result else "  [NOT A RESULT]"
    return f"{r.name} · {r.arch or r.kind} · seed {r.seed if r.seed is not None else '?'} · {r.created_utc[:10]}{tag}"


def select_checkpoint(key: str, label: str = "Model checkpoint"):
    """Selectbox over trained checkpoints; returns the Run or None (with an explanatory empty state)."""
    runs = checkpoint_runs()
    if not runs:
        theme.empty_state("No trained checkpoint in this workspace", "Train a model on the Model Laboratory page (or point LIVERGUARD_WORKSPACE at a workspace that has runs).")
        return None
    i = st.selectbox(label, range(len(runs)), format_func=lambda k: run_label(runs[k]), key=key)
    run = runs[i]
    theme.watermark_banner(None if run.valid_as_result else run.watermark)
    return run


def select_run(kind: str, key: str, label: str, *, split: str | None = None):
    runs = [r for r in registry().runs(kind) if split is None or r.split == split]
    if not runs:
        return None
    i = st.selectbox(label, range(len(runs)), format_func=lambda k: f"{runs[k].name} · {runs[k].split or ''} · {runs[k].created_utc[:16]}"
                     + ("" if runs[k].valid_as_result else "  [NOT A RESULT]"), key=key)
    return runs[i]


def threshold_choice(key: str) -> str:
    c = st.radio("Decision threshold", ["0.5 (fixed)", "val_youden (fitted on validation by the training run)", "custom"], key=key, horizontal=True)
    if c.startswith("custom"):
        return str(st.number_input("Custom threshold", 0.01, 0.99, 0.5, 0.01, key=key + "_v"))
    return "0.5" if c.startswith("0.5") else "val_youden"


def start_job(kind: str, label: str, argv: list[str], out_dir: str = "") -> None:
    job = registry().start_job(kind, label, argv, cwd=ws().repo, log_dir=ws().jobs_dir, out_dir=out_dir)
    st.session_state["last_job"] = job.id
    st.success(f"Started background job #{job.id} ({label}). Follow it in the 'Jobs' panel on this page; results appear in the registry when it finishes.")


def jobs_panel(kind: str | None = None, key: str = "jobs") -> None:
    """Live status, log tail and cancel button for background jobs (real processes, real logs)."""
    jobs = [j for j in registry().jobs(30) if kind is None or j.kind == kind]
    if not jobs:
        st.caption("No jobs started yet.")
        return
    tone = {"running": "warn", "succeeded": "ok", "failed": "bad", "cancelled": "info"}
    df = pd.DataFrame([{"job": j.id, "kind": j.kind, "label": j.label, "status": j.status, "exit": j.returncode,
                        "started": j.started.strftime("%m-%d %H:%M:%S")} for j in jobs])
    st.dataframe(df, hide_index=True, width="stretch")
    sel = st.selectbox("Job log", [j.id for j in jobs], format_func=lambda i: f"#{i} · {next(j.label for j in jobs if j.id == i)}", key=key + "_sel",
                       index=0)
    job = next(j for j in jobs if j.id == sel)
    c1, c2, c3 = st.columns([1, 1, 4])
    if c1.button("Refresh", key=key + "_r"):
        st.rerun()
    if job.status == "running" and c2.button("Cancel", key=key + "_c"):
        registry().cancel_job(job.id)
        st.rerun()
    c3.markdown(theme.badge(job.status, tone.get(job.status, "info")) + f"`{job.command[:140]}`", unsafe_allow_html=True)
    log = ops.tail(job.log_path, 40)
    st.code(log or "(no output yet)", language="text")
    if job.status in ("succeeded", "failed") and st.button("Index new results into the registry", key=key + "_scan"):
        rescan()
        st.rerun()


@st.cache_data(show_spinner=False)
def _split_id(path: str, mtime: float) -> str | None:
    from liverguard.data.splits import split_manifest_id
    from liverguard.ml.data import load_split_frame
    try:
        return split_manifest_id(load_split_frame(path, variant="clean"))
    except Exception:  # noqa: BLE001 - an unusable index just means "unknown"
        return None


def index_split_id() -> str | None:
    """split_id of the clean-variant split defined by the current index (what training runs record), or None if unusable."""
    p = ws().index_csv
    return _split_id(str(p), p.stat().st_mtime) if p.exists() else None
