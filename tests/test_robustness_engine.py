"""Measures (hand-computed), engine sanity (identity / blackout), reproducibility, split guards, outputs, CLI."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from liverguard.data.audit import run_audit, write_audit
from liverguard.data.config import DocumentedReference, load_config
from liverguard.ml import trainer as T
from liverguard.ml.config import TrainConfig
from liverguard.robustness import measures as M
from liverguard.robustness.config import Condition, RobustnessConfig, load_robustness_config
from liverguard.robustness.engine import resolve_threshold, run_robustness_audit, subsample_patients
from liverguard.robustness.review import REVIEW_COLUMNS, summarize_review
from tests.synthetic import build_tree

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ measures: hand-computed
def test_flip_breakdown_matches_hand_computation():
    y = np.array([1, 1, 0, 0])
    pc, pd_ = np.array([.9, .4, .2, .6]), np.array([.3, .7, .4, .2])     # preds clean 1,0,0,1  dist 0,1,0,0
    fb = M.flip_breakdown(y, pc, pd_, 0.5)
    assert fb["pfr"] == 0.75                                              # images 0,1,3 flip
    assert fb["flip_to_wrong"] == 0.25 and fb["flip_to_right"] == 0.5     # img0 right->wrong; img1, img3 wrong->right
    assert fb["conf_drift"] == pytest.approx(((.7 - .9) + (.7 - .6) + (.6 - .8) + (.8 - .6)) / 4)   # = -0.025
    assert fb["mean_abs_dp"] == pytest.approx((.6 + .3 + .2 + .4) / 4)
    assert M.prediction_flips(pc, pc, 0.5).sum() == 0                     # identical -> no flips


def test_first_flip_level_and_mfs_with_censoring():
    fm = np.array([[0, 1, 0], [0, 0, 0], [1, 1, 1]], bool)
    first = M.first_flip_level(fm)
    assert first.tolist() == [2, 4, 1]                                    # never-flipping image censored at L+1
    s = M.mfs_summary(first, 3)
    assert s["frac_ever_flipped"] == pytest.approx(2 / 3) and s["mfs_median"] == 2 and s["mfs_median_label"] == "2"
    assert s["cdf"] == pytest.approx([1 / 3, 2 / 3, 2 / 3])
    majority_censored = M.mfs_summary(np.array([4, 4, 2]), 3)
    assert np.isnan(majority_censored["mfs_median"]) and majority_censored["mfs_median_label"] == ">3"


def _toy_arrays():
    y = np.tile([1, 0], 20)                                               # 20 patients x 2 images
    groups = np.repeat(np.arange(20), 2)
    p = np.where(y == 1, 0.9, 0.1)
    cfg = RobustnessConfig(seed=0, plausible_max_level=2, n_boot=0, flip_alert=0.05, conditions=[
        Condition(name="a", family="fa", func="brightness", params=[1, 1, 1]),
        Condition(name="b", family="fb", func="brightness", params=[1, 1, 1])])
    P = {"a": np.stack([p, p, p]), "b": np.stack([p, 1 - p, np.full_like(p, 0.5)])}   # b: ok, inverted, constant
    z = lambda: {k: np.zeros((3, 40)) for k in ("psnr_db", "correlation", "std_ratio", "new_saturated_fraction")}  # noqa: E731
    FID = {"a": {**z(), "at_risk": np.zeros((3, 40), bool)}, "b": {**z(), "at_risk": np.zeros((3, 40), bool)}}
    FID["b"]["at_risk"][2] = True
    return y, groups, p, cfg, P, FID


def test_summarize_ranking_worst_case_flags_and_any_flip():
    y, g, p, cfg, P, FID = _toy_arrays()
    t = M.summarize(y, g, p, cfg.conditions, P, FID, cfg, 0.5)
    b = t.summary[t.summary.condition == "b"].set_index("level")
    assert t.clean["auroc"] == 1.0
    assert b.loc[1, "pfr"] == 0 and b.loc[2, "pfr"] == 1 and b.loc[3, "pfr"] == 0.5     # constant 0.5 => all called HCC
    assert b.loc[2, "auroc"] == 0 and b.loc[3, "auroc"] == 0.5 and b.loc[3, "d_auroc"] == -0.5
    assert b.loc[2, "tier"] == "plausible" and b.loc[3, "tier"] == "stress"
    assert bool(b.loc[3, "fidelity_flag"]) and not bool(b.loc[2, "fidelity_flag"])
    assert list(t.conditions["condition"]) == ["b", "a"] and t.conditions["rank"].tolist() == [1, 2]
    rb = t.conditions.set_index("condition")
    assert rb.loc["b", "mean_d_auroc_all"] == pytest.approx(-0.5) and rb.loc["a", "mean_d_auroc_all"] == 0
    assert rb.loc["b", "min_level_flip_alert"] == 2 and np.isnan(rb.loc["a", "min_level_flip_alert"])
    assert rb.loc["b", "mfs_median_label"] == "2" and rb.loc["a", "mfs_median_label"] == ">3"
    assert t.worst_case["all_levels"]["auroc"] == {"value": 0.0, "condition": "b", "level": 2, "tier": "plausible"}
    assert t.worst_case["plausible_only"]["auroc"]["condition"] == "b"
    assert t.any_flip["any_condition_plausible_levels"] == 1.0                           # level 2 inverts everyone
    assert t.families.set_index("family").loc["fb", "worst_auroc"] == 0


def test_paired_bootstrap_is_seeded_degenerate_for_identity_and_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = np.tile([1, 0], 30)
    g = np.repeat(np.arange(30), 2)
    pc = np.clip(0.5 + (y - 0.5) * 0.5 + rng.normal(0, 0.25, 60), 0, 1)
    noisy = np.clip(pc + rng.normal(0, 0.2, 60), 0, 1)
    P = {"same": np.stack([pc, pc]), "noisy": np.stack([noisy, noisy])}
    lvl, cnd, cln = M.paired_bootstrap(y, g, pc, P, 0.5, 1, 200, 0.95, seed=5)
    assert lvl[("same", 1)]["d_auroc"] == [0.0, 0.0] and lvl[("same", 1)]["pfr"] == [0.0, 0.0]   # identity: exactly zero
    lo, hi = lvl[("noisy", 1)]["d_auroc"]
    point = M.level_metrics(y, noisy, 0.5)["auroc"] - M.level_metrics(y, pc, 0.5)["auroc"]
    assert lo <= point <= hi and hi - lo > 0
    assert lvl == M.paired_bootstrap(y, g, pc, P, 0.5, 1, 200, 0.95, seed=5)[0]            # seeded
    assert M.paired_bootstrap(y, g, pc, P, 0.5, 1, 0, 0.95, 0) == ({}, {}, {})


def test_review_summary_counts_and_ignores_blank(tmp_path):
    rows = [("i1", "jpeg", 5, "yes"), ("i2", "jpeg", 5, "no"), ("i3", "jpeg", 5, "yes"), ("i4", "jpeg", 5, "unsure"),
            ("i5", "jpeg", 5, ""), ("i1", "crop", 5, "no")]
    df = pd.DataFrame([{"image_id": i, "condition": c, "level": lv, "tier": "stress", "fidelity_at_risk": False,
                        "reviewer_label_preserved": a, "reviewer_notes": ""} for i, c, lv, a in rows])
    assert list(df.columns) == REVIEW_COLUMNS
    df.to_csv(tmp_path / "r.csv", index=False)
    r = summarize_review(tmp_path / "r.csv").set_index("condition")
    assert r.loc["jpeg", "share_preserved"] == pytest.approx(2 / 3) and r.loc["jpeg", "n_unsure"] == 1
    assert r.loc["jpeg", "n_unanswered"] == 1 and r.loc["crop", "share_preserved"] == 0


# ------------------------------------------------------------------ engine fixtures
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("rob")
    build_tree(base / "d", n_per_class=40, signal=True)
    dcfg = load_config(data_root=base / "d")
    dcfg.documented_reference = DocumentedReference()
    index = write_audit(run_audit(dcfg), base / "audit")["index"]
    tcfg = TrainConfig(arch="tiny_cnn", pretrained=False, experiment_kind="smoke", image_size=32, batch_size=16, epochs=12,
                       freeze_epochs=1, lr=3e-3, num_workers=0, bootstrap_samples=20, patience=12, dropout=0.1, amp=False)
    man = T.run_experiment(tcfg, index, base / "run")
    return {"index": index, "ckpt": base / "run" / "best.pt", "run": base / "run", "manifest": man, "base": base,
            "dcfg": dcfg}


def _audit(world, out, cfg=None, **kw):
    cfg = cfg or load_robustness_config(n_boot=0)
    return run_robustness_audit(world["ckpt"], world["index"], world["base"] / out, cfg=cfg, make_plots=kw.pop("make_plots", False),
                                review_n=kw.pop("review_n", 2), **kw)


def _custom(conds, **kw) -> RobustnessConfig:
    return RobustnessConfig(seed=0, plausible_max_level=1, n_boot=0, conditions=conds, **kw)


# ------------------------------------------------------------------ engine sanity
def test_identity_condition_changes_nothing_and_blackout_destroys_everything(world):
    cfg = _custom([Condition(name="identity", family="brightness", func="brightness", params=[1.0]),
                   Condition(name="blackout", family="brightness", func="brightness", params=[0.0])])
    r = _audit(world, "sanity", cfg)
    pi = r.per_image
    ident = pi[pi.condition == "identity"]
    assert np.allclose(ident.p_dist, ident.p_clean, atol=1e-12) and ident.flip.sum() == 0
    assert np.allclose(ident.conf_drift, 0)
    s = r.tables.summary.set_index("condition")
    assert s.loc["identity", "d_auroc"] == 0 and s.loc["identity", "pfr"] == 0 and s.loc["identity", "mean_abs_dp"] < 1e-12
    black = pi[pi.condition == "blackout"]
    assert black.p_dist.nunique() == 1                                    # every image becomes the same image
    assert s.loc["blackout", "auroc"] == 0.5 and s.loc["blackout", "fidelity_flag"]
    assert r.tables.clean["auroc"] > 0.9                                  # the underlying model does work on clean data


def test_clean_level_equals_training_run_predictions(world):
    r = _audit(world, "consistency", _custom([Condition(name="identity", family="brightness", func="brightness", params=[1.0])]))
    trained = pd.read_csv(world["run"] / "predictions.csv")
    trained = trained[trained.split == "val"].set_index("image_id")["p_hcc"]
    mine = r.per_image.drop_duplicates("image_id").set_index("image_id")["p_clean"]
    assert set(mine.index) == set(trained.index)
    assert np.allclose(mine.sort_index(), trained.sort_index(), atol=1e-5)


# ------------------------------------------------------------------ reproducibility
def test_runs_are_reproducible_order_free_and_seed_sensitive(world):
    cfg = load_robustness_config(n_boot=30)
    a, b = _audit(world, "rep_a", cfg), _audit(world, "rep_b", cfg)
    pd.testing.assert_frame_equal(a.per_image, b.per_image)
    pd.testing.assert_frame_equal(a.tables.summary, b.tables.summary)    # including the bootstrap CIs
    assert a.tables.summary["d_auroc_lo"].notna().any()
    only = _audit(world, "rep_sub", cfg.select(["gaussian_noise"]))
    full = a.per_image[a.per_image.condition == "gaussian_noise"].reset_index(drop=True)
    pd.testing.assert_frame_equal(only.per_image.reset_index(drop=True), full)    # subset audit == same numbers
    other = _audit(world, "rep_seed", load_robustness_config(n_boot=0, seed=1))
    key = ["image_id", "condition", "level"]
    m = a.per_image.merge(other.per_image, on=key, suffixes=("_a", "_o"))
    noise = m[m.condition == "gaussian_noise"]
    assert not np.allclose(noise.p_dist_a, noise.p_dist_o)                # stochastic distortion depends on seed
    det = m[m.condition == "brightness_dark"]
    assert np.allclose(det.p_dist_a, det.p_dist_o)                        # deterministic one does not


# ------------------------------------------------------------------ outputs, manifest, plots
def test_outputs_manifest_review_sheet_and_figures(world):
    r = _audit(world, "full", load_robustness_config(n_boot=20), make_plots=True, review_n=2)
    for key in ("per_image", "summary", "conditions", "families", "manifest", "review_template"):
        assert r.paths[key].stat().st_size > 0, key
    figs = [p for k, p in r.paths.items() if k.startswith("fig_")]
    assert len(figs) == 4 + 2 and all(p.stat().st_size > 5000 for p in figs)
    m = json.loads(r.paths["manifest"].read_text())
    assert m["valid_as_result"] is False and "SMOKE TEST" in m["watermark"]          # propagated from the checkpoint
    assert m["split"] == "val" and "DIAGNOSTIC" in m["audit_scope"] and len(m["checkpoint"]["sha256"]) == 64
    assert m["checkpoint"]["split_id"] == m["dataset"]["split_id"] == world["manifest"]["dataset"]["split_id"]
    assert m["seed"] == 0 and m["threshold"] == {"value": 0.5, "source": "fixed"}
    assert any("Label preservation" in a for a in m["assumptions"]) and "worst_case" in m and "any_flip" in m
    assert set(m["artifacts"]) >= {"per_image", "fig_degradation", "review_template"}
    assert len(r.tables.summary) == 12 * 5 and r.tables.summary.groupby("condition")["level"].nunique().eq(5).all()
    rv = pd.read_csv(r.paths["review_template"], dtype=str, keep_default_na=False)
    assert list(rv.columns) == REVIEW_COLUMNS and len(rv) == 2 * 60 and (rv.reviewer_label_preserved == "").all()
    pi = pd.read_csv(r.paths["per_image"])
    assert {"p_clean", "p_dist", "flip", "conf_drift", "tier", "at_risk"} <= set(pi.columns)
    assert len(pi) == r.tables.summary["n"].iloc[0] * 60
    assert (pi.groupby(["condition", "level"]).size() == r.tables.summary["n"].iloc[0]).all()


# ------------------------------------------------------------------ guards
def test_test_split_requires_explicit_final_report(world):
    cfg = _custom([Condition(name="identity", family="brightness", func="brightness", params=[1.0])])
    with pytest.raises(ValueError, match="final_report"):
        _audit(world, "t0", cfg, split="test")
    r = _audit(world, "t1", cfg, split="test", final_report=True)
    assert r.manifest["split"] == "test" and "FINAL REPORT" in r.manifest["audit_scope"]
    with pytest.raises(ValueError, match="split must be"):
        _audit(world, "t2", cfg, split="train")


def test_checkpoint_trained_on_a_different_split_is_refused(world):
    from liverguard.data.splits import build_patient_splits
    from liverguard.ml.data import load_split_frame
    df = load_split_frame(world["index"], variant="clean")
    other = build_patient_splits(df, world["dcfg"], seed=11)
    sf = world["base"] / "other_split.csv"
    pd.DataFrame({"image_id": df["image_id"], "split": other}).to_csv(sf, index=False)
    cfg = _custom([Condition(name="identity", family="brightness", func="brightness", params=[1.0])])
    with pytest.raises(ValueError, match="split_id"):
        _audit(world, "mismatch", cfg, split_file=sf)


def test_patient_subsample_is_whole_patients_class_balanced_and_seeded(world):
    from liverguard.ml.data import load_split_frame
    part = load_split_frame(world["index"], variant="clean").query("split == 'val'")
    a, b = subsample_patients(part, 8, 0), subsample_patients(part, 8, 0)
    assert a.equals(b) and not a.equals(subsample_patients(part, 8, 3))
    assert a["label"].nunique() == 2 and a["patient_key"].nunique() <= 10
    full_counts = part.groupby("patient_key").size()
    assert (a.groupby("patient_key").size() == full_counts[a["patient_key"].unique()]).all()   # patients kept whole
    r = _audit(world, "sub", _custom([Condition(name="identity", family="brightness", func="brightness", params=[1.0])]),
               max_patients=8)
    assert r.manifest["subsampled"]["complete"] is False


def test_threshold_resolution(world, tmp_path):
    thr = world["manifest"]["results"]["val_youden_threshold"]
    assert resolve_threshold("val_youden", world["ckpt"]) == (pytest.approx(thr), "val_youden from manifest.json")
    assert resolve_threshold("0.3", world["ckpt"]) == (0.3, "fixed")
    lone = tmp_path / "best.pt"
    lone.write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        resolve_threshold("val_youden", lone)


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError, match="Unknown condition"):
        load_robustness_config().select(["sharpen"])


# ------------------------------------------------------------------ CLI
def _cli(*args):
    return subprocess.run([sys.executable, "scripts/audit_robustness.py", *map(str, args)], cwd=REPO, capture_output=True, text=True)


def test_cli_success_and_clean_errors(world, tmp_path):
    base = ["--checkpoint", world["ckpt"], "--index-csv", world["index"]]
    ok = _cli(*base, "--out-dir", tmp_path / "o", "--n-boot", "0", "--conditions", "jpeg,gaussian_noise", "--no-plots")
    assert ok.returncode == 0, ok.stderr[-600:]
    assert "NOT A RESULT" in ok.stdout and "DIAGNOSTIC" in ok.stdout and "most disruptive" in ok.stdout
    assert (tmp_path / "o" / "audit_manifest.json").exists()
    guard = _cli(*base, "--out-dir", tmp_path / "g", "--split", "test", "--no-plots")
    assert guard.returncode == 2 and "final_report" in guard.stderr and "Traceback" not in guard.stderr
    assert _cli("--checkpoint", tmp_path / "nope.pt", "--index-csv", world["index"], "--out-dir", tmp_path / "m").returncode == 2
    bad = _cli(*base, "--out-dir", tmp_path / "b", "--conditions", "sharpen")
    assert bad.returncode == 2 and "Unknown condition" in bad.stderr
