"""App tests: registry/jobs, ops, charts, PDF and Streamlit AppTest workflows against a real (tiny, synthetic) workspace.

The workspace is produced by the real CLIs through the app's own job runner, so launching, indexing and rendering are all exercised.
Everything here is synthetic pipeline verification, never a finding.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from liverguard.app import charts, ops
from liverguard.app.config import Workspace
from liverguard.db.registry import Registry
from liverguard.reports.pdf import build_report, run_to_dict
from liverguard.robustness.config import load_robustness_config
from tests.test_robustness_engine import world  # noqa: F401

REPO = Path(__file__).resolve().parents[1]


def _wait(reg: Registry, job_id: int, timeout: float = 240) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = reg.refresh_job(job_id)
        if j.status != "running":
            return j.status
        time.sleep(0.5)
    raise TimeoutError


@pytest.fixture(scope="module")
def ws_env(world, tmp_path_factory):  # noqa: F811
    import os
    root = world["base"]
    os.environ["LIVERGUARD_WORKSPACE"] = str(root)
    os.environ["LIVERGUARD_INDEX_CSV"] = str(world["index"])
    w = Workspace(root=root, index_csv=Path(world["index"]), robustness_config=REPO / "configs" / "robustness.yaml", repo=REPO)
    reg = Registry(w.db_path)
    ck = str(world["ckpt"])
    jobs = [("audit", ops.audit_argv(w, ck, root / "a_val", conditions=["brightness_dark", "gaussian_noise"], n_boot=0, max_patients=10)),
            ("reliability", ops.reliability_argv(w, ck, root / "r_val", n_boot=0, mc_samples=3)),
            ("explain", ops.explain_argv(w, ck, root / "e_val", n_boot=0))]
    for kind, argv in jobs:
        j = reg.start_job(kind, kind, argv, cwd=REPO, log_dir=w.jobs_dir, out_dir=str(root))
        assert _wait(reg, j.id) == "succeeded", ops.tail(j.log_path, 30)
    reg.scan(root)
    return w, reg, world


def test_registry_indexes_all_kinds_and_is_idempotent(ws_env):
    w, reg, _ = ws_env
    kinds = reg.counts()
    assert {"train", "audit", "reliability", "explain"} <= set(kinds)
    n = len(reg.runs())
    reg.scan(w.root)
    assert len(reg.runs()) == n
    assert all(not r.valid_as_result and r.watermark for r in reg.runs())  # synthetic runs are never valid results


def test_registry_reports_bad_manifest_without_crashing(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "manifest.json").write_text("{not json")
    res = Registry(tmp_path / "db.sqlite").scan(tmp_path)
    assert len(res["errors"]) == 1


def test_job_failure_and_cancel(tmp_path):
    reg = Registry(tmp_path / "db.sqlite")
    j = reg.start_job("t", "fails", ["python", "-c", "import sys; sys.exit(3)"], cwd=tmp_path, log_dir=tmp_path)
    assert _wait(reg, j.id, 30) == "failed" and reg.refresh_job(j.id).returncode == 3
    k = reg.start_job("t", "sleeps", ["python", "-c", "import time; time.sleep(60)"], cwd=tmp_path, log_dir=tmp_path)
    reg.cancel_job(k.id)
    assert reg.refresh_job(k.id).status == "cancelled"


def test_ops_distort_compare_and_explain(ws_env):
    w, reg, world = ws_env  # noqa: F811
    run = next(r for r in reg.runs("train"))
    model, meta, dev = ops.load_model(run.checkpoint_path)
    size = int(meta["image_size"])
    idx = pd.read_csv(world["index"])
    path = idx[idx.split == "val"].iloc[0]
    img = ops.open_rgb(path["path"], size)
    cond = next(c for c in load_robustness_config(n_boot=0).conditions if c.name == "brightness_dark")
    d1 = ops.distort(img, cond, 3, seed=1, image_id="a")
    d2 = ops.distort(img, cond, 3, seed=1, image_id="a")
    assert np.array_equal(np.asarray(d1), np.asarray(d2)) and not np.array_equal(np.asarray(d1), np.asarray(img))  # reproducible, and it does something
    with pytest.raises(ValueError):
        ops.distort(img, cond, 99, seed=1, image_id="a")
    cmp = ops.compare_prediction(model, dev, img, d1, 0.5)
    assert cmp["flip"] == (cmp["pred_original"] != cmp["pred_distorted"]) and abs(cmp["confidence_drift"] - (cmp["confidence_distorted"] - cmp["confidence_original"])) < 1e-12
    cam, probs = ops.gradcam(model, img, size, dev, "predicted")
    assert cam.shape == (size, size) and 0 <= cam.min() and cam.max() <= 1 and abs(probs.sum() - 1) < 1e-4
    occ, base = ops.occlusion(model, img, size, dev, 1, max(4, size // 4), max(2, size // 8))
    assert occ.ndim == 2 and 0 <= base <= 1
    from liverguard.explain.masking import proxy_roi
    df = ops.roi_experiment(model, dev, img, proxy_roi(size, 0.4), 0, 2)
    assert {"variant", "p_hcc"} <= set(df.columns) and len(df) >= 3


def test_pipeline_status_is_derived_not_asserted():
    st = {s["stage"]: s["status"] for s in ops.pipeline_status([], False)}
    assert set(st.values()) == {"not run"}
    st = {s["stage"]: s["status"] for s in ops.pipeline_status([("train", False), ("audit", True)], True)}
    assert st["BUILD"] == "pipeline check only" and st["ATTACK"] == "done" and st["DIAGNOSE"] == "pipeline check only" and st["RE-TEST"] == "not run"


def test_charts_build_from_real_outputs(ws_env):
    w, reg, _ = ws_env
    a = next(r for r in reg.runs("audit"))
    s = pd.read_csv(Path(a.out_dir) / "summary_by_level.csv")
    conds = list(s["condition"].unique())
    for m in ("d_auroc", "pfr", "conf_drift"):
        fig = charts.degradation_curves(s, m, conds, watermark="NOT A RESULT")
        assert len(fig.data) == len(conds) and any("NOT A RESULT" in (x.text or "") for x in fig.layout.annotations)
    r = next(r for r in reg.runs("reliability"))
    out = Path(r.out_dir)
    charts.reliability_diagram(pd.read_csv(out / "reliability_bins.csv"))
    charts.confidence_histogram(pd.read_csv(out / "per_image_reliability.csv"))
    charts.risk_coverage_fig(pd.read_csv(out / "risk_coverage.csv"), pd.read_csv(out / "selective_summary.csv"))
    charts.occlusion_heat(np.zeros((4, 4)))  # all-zero map must not divide by zero


def test_pdf_report_states_non_results(ws_env, tmp_path):
    _, reg, _ = ws_env
    runs = [run_to_dict(r) for r in reg.runs()[:4]]
    p = build_report(tmp_path / "r.pdf", "t", runs, "a <note> & more")
    assert p.exists() and p.read_bytes()[:4] == b"%PDF"
    with pytest.raises(ValueError):
        build_report(tmp_path / "e.pdf", "t", [])


# ---------------------------------------------------------------- Streamlit workflows
def _page(mod: str) -> AppTest:
    return AppTest.from_string(f"from liverguard.app.pages import {mod}; {mod}.render()", default_timeout=180).run()


@pytest.mark.parametrize("mod", ["dashboard", "dataset", "model_lab", "image_audit", "explain", "reliability", "robust_lab", "compare", "reports"])
def test_every_page_renders_without_exception(ws_env, mod):
    at = _page(mod)
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_pages_show_non_result_warning_for_synthetic_runs(ws_env):
    at = _page("reliability")
    assert any("NOT A" in m.value.upper() or "NOT A RESULT" in m.value.upper() for m in at.markdown)


def test_empty_workspace_shows_empty_states(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVERGUARD_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("LIVERGUARD_INDEX_CSV", str(tmp_path / "none.csv"))
    for mod in ["dashboard", "dataset", "model_lab", "image_audit", "explain", "reliability", "robust_lab", "compare", "reports"]:
        at = _page(mod)
        assert not at.exception, (mod, [str(e.value) for e in at.exception])


def test_reports_workflow_generates_pdf(ws_env):
    at = _page("reports")
    at.tabs[1]  # 'Build a report'
    ms = next(m for m in at.multiselect if m.key == "rp_runs")
    ms.set_value(ms.options[:2]).run()
    next(b for b in at.button if b.key == "rp_go").click().run()
    assert not at.exception and not at.error
    assert any("Report written" in s.value for s in at.success)


def test_test_split_analysis_is_locked_without_final_report(ws_env):
    at = _page("reliability")
    sb = next(s for s in at.selectbox if s.key == "rel_split")
    sb.set_value("test").run()
    go = next(b for b in at.button if b.key == "rel_go")
    assert go.disabled
    next(c for c in at.checkbox if c.key == "rel_final").check().run()
    assert not next(b for b in at.button if b.key == "rel_go").disabled


def test_robust_final_is_locked_before_validation(ws_env):
    at = _page("robust_lab")
    assert next(b for b in at.button if b.key == "rt_final").disabled


def _count(at, kind: str) -> int:
    def walk(e):
        return int(e.type == kind) + sum(walk(c) for c in getattr(e, "children", {}).values())
    return walk(at.main)


def test_explainability_workflows_produce_images_and_charts(ws_env):
    at = _page("explain")
    assert _count(at, "image") >= 3  # original, heat map, overlay
    before = _count(at, "plotly_chart")
    next(b for b in at.button if b.key == "ex_occ_btn").click().run()
    assert not at.exception and _count(at, "plotly_chart") == before + 1
    next(b for b in at.button if b.key == "ex_roi_btn").click().run()
    assert not at.exception and _count(at, "plotly_chart") == before + 2


def test_image_audit_shows_original_distorted_and_sweep(ws_env):
    at = _page("image_audit")
    assert _count(at, "image") >= 3
    lvl = next(s for s in at.slider if s.key == "ia_lvl")
    lvl.set_value(5).run()
    assert not at.exception and _count(at, "image") >= 3


def test_model_lab_rejects_invalid_training_config(ws_env):
    from liverguard.app import ops
    w, _, _ = ws_env
    with pytest.raises(Exception):
        ops.train_config_file(w, {"arch": "not_an_arch", "epochs": -1}, "bad")


# ---------------------------------------------------------------- audit-review regression tests
def test_run_names_cannot_escape_the_workspace():
    for bad in ("../x", "a/b", ".hidden", "", "a b", "x" * 200):
        with pytest.raises(ValueError):
            ops.check_run_name(bad)
    assert ops.check_run_name("effnet_b0-s42.v2") == "effnet_b0-s42.v2"


def test_registry_prunes_runs_whose_manifest_was_deleted(tmp_path):
    d = tmp_path / "audit_x"
    d.mkdir()
    (d / "audit_manifest.json").write_text(json.dumps({"created_utc": "2026-01-01", "valid_as_result": False, "watermark": "w", "split": "val"}))
    reg = Registry(tmp_path / "db.sqlite")
    assert reg.scan(tmp_path)["added"] == 1 and len(reg.runs()) == 1
    (d / "audit_manifest.json").unlink()
    assert reg.scan(tmp_path)["removed"] == 1 and reg.runs() == []


def test_cancelling_an_already_finished_job_keeps_its_real_outcome(tmp_path):
    reg = Registry(tmp_path / "db.sqlite")
    j = reg.start_job("t", "quick", ["python", "-c", "pass"], cwd=tmp_path, log_dir=tmp_path)
    time.sleep(2)  # finished, but nobody has refreshed the row yet
    assert reg.cancel_job(j.id).status == "succeeded"


def test_dataset_level_audit_tab_shows_real_curves_and_guards_test(ws_env):
    w, reg, _ = ws_env
    at = _page("image_audit")
    assert not at.exception, [str(e.value) for e in at.exception]
    assert _count(at, "plotly_chart") >= 2  # degradation curves + ranking/heatmap built from the audit CSVs
    sb = next(s for s in at.selectbox if s.key == "aul_split")
    sb.set_value("test").run()
    assert next(b for b in at.button if b.key == "aul_go").disabled
    next(c for c in at.checkbox if c.key == "aul_final").check().run()
    assert not next(b for b in at.button if b.key == "aul_go").disabled


def test_train_from_the_ui_end_to_end(ws_env):
    """Start a tiny real training from the Model Laboratory page, wait for the job, and find the run indexed with test NOT evaluated."""
    w, reg, _ = ws_env
    at = _page("model_lab")
    pick = lambda label: next(x for x in list(at.selectbox) + list(at.number_input) + list(at.text_input) + list(at.checkbox) if x.label == label)  # noqa: E731
    pick("Architecture").set_value("tiny_cnn").run()  # the default run name follows the architecture, so settle it before typing a name
    pick("Experiment kind").set_value("smoke")
    pick("ImageNet pretrained").uncheck()
    pick("Image size").set_value(64)
    pick("Epochs").set_value(1)
    pick("Frozen-backbone epochs").set_value(0)
    pick("Bootstrap replicates (CI)").set_value(0)
    at.run()
    pick("Run name").set_value("ui_e2e")
    assert pick("Evaluate the TEST split when training ends").value is False  # default must not touch test
    at.run()
    n_jobs = len(reg.jobs(500))
    next(b for b in at.button if b.label == "Start training").click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert len(reg.jobs(500)) == n_jobs + 1  # the click really started a job
    job = max(reg.jobs(500), key=lambda j: j.id)
    assert _wait(reg, job.id) == "succeeded", ops.tail(job.log_path, 30)
    reg.scan(w.root)
    run = next(r for r in reg.runs("train") if r.name == "ui_e2e")
    m = json.loads(Path(run.manifest_path).read_text())
    assert "test" not in m["results"] and "Test split NOT evaluated" in m["test_set_policy"]
    # a second run under the same name is refused rather than overwritten
    at = _page("model_lab")
    pick = lambda label: next(x for x in list(at.selectbox) + list(at.number_input) + list(at.text_input) + list(at.checkbox) if x.label == label)  # noqa: E731
    pick("Architecture").set_value("tiny_cnn").run()
    pick("Run name").set_value("ui_e2e").run()
    next(b for b in at.button if b.label == "Start training").click().run()
    assert any("already exists" in e.value for e in at.error)


def test_invalid_run_name_is_rejected_in_the_ui(ws_env):
    at = _page("model_lab")
    next(t for t in at.text_input if t.label == "Run name").set_value("../escape").run()
    next(b for b in at.button if b.label == "Start training").click().run()
    assert any("Run name" in e.value for e in at.error)


def test_dataset_verification_from_the_ui_builds_the_index(world, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("LIVERGUARD_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("LIVERGUARD_INDEX_CSV", str(tmp_path / "dataset" / "image_index.csv"))
    at = _page("dataset")  # no index yet: the page offers the verification panel instead of crashing
    assert not at.exception
    root = str(world["dcfg"].data_root)
    next(t for t in at.text_input if t.key == "ds_root").set_value(root).run()
    next(b for b in at.button if b.label == "Start verification").click().run()
    assert not at.exception and not at.error
    reg = Registry(Workspace(root=tmp_path, index_csv=tmp_path / "dataset" / "image_index.csv", robustness_config=REPO / "configs" / "robustness.yaml", repo=REPO).db_path)
    job = max(reg.jobs(50), key=lambda j: j.id)
    assert _wait(reg, job.id) == "succeeded", ops.tail(job.log_path, 30)
    idx = pd.read_csv(tmp_path / "dataset" / "image_index.csv")
    assert {"train", "val", "test"} <= set(idx["split"]) and len(idx) > 100
    at = _page("dataset")  # now the explorer renders from the real index
    assert not at.exception and not at.error
    assert _count(at, "plotly_chart") >= 1


def test_nonexistent_data_root_is_reported(ws_env):
    at = _page("dataset")
    next(t for t in at.text_input if t.key == "ds_root").set_value("/definitely/not/here").run()
    next(b for b in at.button if b.label == "Start verification").click().run()
    assert any("Not a directory" in e.value for e in at.error)


def test_registry_reads_training_seed_and_audit_subsample_flag(ws_env):
    _, reg, _ = ws_env
    train = next(r for r in reg.runs("train"))
    assert train.seed is not None  # seed lives under config.seed in training manifests (was shown as "seed ?")
    a = next(r for r in reg.runs("audit"))
    man = json.loads(Path(a.manifest_path).read_text())
    sub = man["subsampled"]
    assert sub["complete"] == (not sub["max_patients"])  # the UI must read this flag, not the truthiness of the dict (always true)
    from liverguard.app.pages.audit_results import subsample_note
    assert ("sub-sampled" in subsample_note(man)) == (not sub["complete"])
    assert "all patients" in subsample_note({"subsampled": {"max_patients": None, "complete": True}})
    assert "sub-sampled" in subsample_note({"subsampled": {"max_patients": 5, "complete": False}})
