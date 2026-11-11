"""M4: calibration, uncertainty, selective prediction, failure analysis - hand-computed values first, then the engine end to end."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from liverguard.reliability import calibration as cal
from liverguard.reliability import failure, selective
from liverguard.reliability import uncertainty as unc
from liverguard.reliability.engine import run_reliability_analysis

from tests.test_robustness_engine import world  # noqa: F401  (module fixture: synthetic tree + smoke-trained checkpoint)

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ calibration metrics (hand computed)
def test_brier_and_nll_hand_computed():
    y, p = [1, 0, 1, 0], [0.9, 0.2, 0.6, 0.5]
    assert cal.brier_score(y, p) == pytest.approx((0.01 + 0.04 + 0.16 + 0.25) / 4)
    assert cal.nll(y, p) == pytest.approx(-(np.log(0.9) + np.log(0.8) + np.log(0.6) + np.log(0.5)) / 4)


def test_ece_hand_computed_two_bins():
    # confidences 0.9 x4 (3 correct) and 0.6 x4 (2 correct). bins of width 0.05 -> separate bins
    y = [1, 1, 1, 0, 1, 1, 0, 0]
    p = [0.9, 0.9, 0.9, 0.9, 0.6, 0.6, 0.6, 0.6]       # preds all 1; correct: first three, then 5th,6th
    ece = cal.expected_calibration_error(y, p, n_bins=10, strategy="width")
    # bin(0.9): acc 3/4, conf .9 -> .15 ; bin(0.6): acc 2/4, conf .6 -> .10 ; weights .5/.5
    assert ece == pytest.approx(0.5 * 0.15 + 0.5 * 0.10)


def test_perfectly_calibrated_and_overconfident_extremes():
    assert cal.expected_calibration_error([1, 1, 0, 0], [1.0, 1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-9)
    # always 99% sure but right half the time -> ECE ~ 0.49
    y = [1, 0] * 10
    assert cal.expected_calibration_error(y, [0.99] * 20) == pytest.approx(0.49)


def test_reliability_table_counts_sum_to_n_for_both_strategies():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 97)
    y = (rng.uniform(0, 1, 97) < p).astype(int)
    for s in ("width", "mass"):
        assert cal.reliability_table(y, p, 10, s)["n"].sum() == 97


def test_temperature_scaling_recovers_known_temperature_and_keeps_ranking():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, 4000)
    y = (rng.uniform(size=4000) < cal.sigmoid(z)).astype(int)      # true logits are z
    p_over = cal.sigmoid(3.0 * z)                                   # model is 3x over-confident
    T = cal.fit_temperature(y, p_over)
    assert T == pytest.approx(3.0, rel=0.15)
    q = cal.apply_temperature(p_over, T)
    assert cal.expected_calibration_error(y, q) < cal.expected_calibration_error(y, p_over)
    assert np.array_equal(np.argsort(p_over, kind="stable"), np.argsort(q, kind="stable"))   # ranking (AUROC) unchanged


def test_temperature_single_class_returns_identity():
    assert cal.fit_temperature([1, 1, 1], [0.9, 0.8, 0.7]) == 1.0


def test_crossfit_never_splits_a_patient_and_is_deterministic():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(20), 3)
    y = np.tile([0, 1, 1], 20)
    p = rng.uniform(0.05, 0.95, 60)
    a, Ts = cal.crossfit_temperature(y, p, groups, k=4, seed=3)
    b, _ = cal.crossfit_temperature(y, p, groups, k=4, seed=3)
    assert np.array_equal(a, b) and len(Ts) == 4
    # a patient's three images are all transformed with the same T -> the logit ratio is constant within a patient
    for g in range(20):
        m = groups == g
        ratio = cal.logit(a[m]) / cal.logit(p[m])
        assert np.allclose(ratio, ratio[0], atol=1e-6)


def test_cluster_bootstrap_resamples_patients_and_zero_disables():
    y, p = np.array([1, 0] * 10), np.linspace(0.1, 0.9, 20)
    g = np.repeat(np.arange(10), 2)
    lo, hi = cal.cluster_bootstrap(y, p, g, cal.brier_score, n_boot=100, seed=0)
    assert lo <= cal.brier_score(y, p) + 0.05 and hi >= lo
    assert all(np.isnan(cal.cluster_bootstrap(y, p, g, cal.brier_score, n_boot=0)))


# ------------------------------------------------------------------ uncertainty
def test_binary_entropy_values_and_rank_equivalence_with_confidence():
    assert unc.binary_entropy([0.5])[0] == pytest.approx(1.0)
    assert unc.binary_entropy([1.0])[0] == pytest.approx(0.0, abs=1e-9)
    p = np.random.default_rng(0).uniform(0, 1, 200)
    assert np.array_equal(np.argsort(unc.binary_entropy(p), kind="stable"), np.argsort(-unc.confidence(p), kind="stable"))


class _HeadDrop(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Flatten(), nn.Linear(12, 8), nn.ReLU())
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(8, 2))

    def forward(self, x):
        return self.fc(self.features(x))


def test_mc_dropout_is_seeded_varies_and_leaves_model_in_eval():
    torch.manual_seed(0)
    m = _HeadDrop().eval()
    x = torch.randn(5, 3, 2, 2)
    a, b = unc.mc_dropout_probs(m, x, 30, seed=1), unc.mc_dropout_probs(m, x, 30, seed=1)
    assert np.array_equal(a, b) and a.shape == (30, 5)
    assert a.std(0).min() > 0                                       # dropout really is active
    assert not m.fc[0].training and not m.training
    # deterministic path with dropout off: eval forward is unaffected by MC sampling
    ref = torch.softmax(m(x), 1)[:, 1].detach().numpy()
    assert np.allclose(torch.softmax(m(x), 1)[:, 1].detach().numpy(), ref)


def test_mc_fast_path_matches_full_passes_when_dropout_only_in_head():
    m = _HeadDrop().eval()
    x = torch.randn(4, 3, 2, 2)
    fast = unc.mc_dropout_probs(m, x, 12, seed=5, fast=True)
    full = unc.mc_dropout_probs(m, x, 12, seed=5, fast=False)
    assert np.allclose(fast, full, atol=1e-6)


def test_mc_summary_mutual_information_zero_without_variation():
    s = unc.mc_summary(np.full((10, 4), 0.7))
    assert np.allclose(s["mc_std"], 0) and np.allclose(s["mc_mi"], 0, atol=1e-9)
    s2 = unc.mc_summary(np.array([[0.1, 0.9]] * 5 + [[0.9, 0.1]] * 5))
    assert (s2["mc_mi"] > 0.5).all()                                # disagreeing samples -> high epistemic term


def test_tta_summary_flags_decision_changes():
    P = np.array([[0.9, 0.2], [0.8, 0.2], [0.4, 0.1]])             # 3 views x 2 images
    s = unc.tta_summary(P, 0.5)
    assert list(s["tta_flip"]) == [True, False] and s["tta_std"][0] > s["tta_std"][1]


# ------------------------------------------------------------------ selective prediction
def test_risk_coverage_hand_computed_and_oracle_bounds():
    err = np.array([0, 0, 1, 0, 1])
    u = np.array([0.1, 0.2, 0.9, 0.3, 0.8])                         # order kept: idx 0,1,3,4,2
    rc = selective.risk_coverage(err, u)
    assert list(rc["risk"].round(4)) == [0.0, 0.0, 0.0, 0.25, 0.4]
    assert selective.aurc(rc) == pytest.approx(np.mean([0, 0, 0, 0.25, 0.4]))
    assert selective.oracle_aurc(err) <= selective.aurc(rc) <= 1
    assert selective.random_aurc(err) == pytest.approx(0.4)
    assert selective.risk_at_coverage(rc, 1.0) == pytest.approx(0.4)


def test_risk_coverage_ties_are_deterministic():
    a = selective.risk_coverage([1, 0, 1, 0], [0.5] * 4)
    b = selective.risk_coverage([1, 0, 1, 0], [0.5] * 4)
    assert a.equals(b)


# ------------------------------------------------------------------ failure analysis
def test_label_failures_outcomes_and_confident_errors():
    df = pd.DataFrame({"label": [1, 1, 0, 0], "p": [0.95, 0.2, 0.05, 0.97]})
    out = failure.label_failures(df, 0.5, confident_at=0.9)
    assert list(out.outcome) == ["TP", "FN", "TN", "FP"]
    assert list(out.confident_error) == [False, False, False, True]
    assert out.loc[3, "error_rank"] == 1 and np.isnan(out.loc[0, "error_margin"])


def test_patient_level_aggregation_and_disagreement():
    df = pd.DataFrame({"patient_key": ["a", "a", "b"], "label": [1, 1, 0], "p": [0.9, 0.3, 0.2]})
    df = failure.label_failures(df.assign(entropy=0.0), 0.5)
    pl = failure.patient_level(df, 0.5).set_index("patient_key")
    assert pl.loc["a", "mean_p"] == pytest.approx(0.6) and bool(pl.loc["a", "images_disagree"])
    assert bool(pl.loc["a", "patient_correct"]) and not bool(pl.loc["b", "images_disagree"])


# ------------------------------------------------------------------ engine
def test_engine_outputs_manifest_and_limitations(world, tmp_path):
    r = run_reliability_analysis(world["ckpt"], world["index"], tmp_path / "rel", n_boot=20, mc_samples=6, make_plots=True, seed=0)
    m = r.manifest
    assert m["valid_as_result"] is False and "SMOKE" in m["watermark"]
    assert m["scope"].startswith("DIAGNOSTIC") and len(m["limitations"]) >= 6
    names = {c["probabilities"] for c in m["calibration"]}
    assert "uncalibrated" in names and any("cross-fitted" in n for n in names) and any("optimistic" in n for n in names)
    for f in ("per_image_reliability", "calibration_summary", "reliability_bins", "risk_coverage", "selective_summary",
              "failure_cases", "outcome_summary", "patient_level", "manifest", "fig_reliability", "fig_risk_coverage"):
        assert r.paths[f].exists(), f
    pi = r.per_image
    assert ((pi.p >= 0) & (pi.p <= 1)).all() and (pi.entropy >= 0).all() and (pi.entropy <= 1 + 1e-9).all()
    assert set(pi.outcome) <= {"TP", "TN", "FP", "FN"}
    assert json.loads(r.paths["manifest"].read_text())["dataset"]["n_images"] == len(pi)


def test_engine_is_reproducible(world, tmp_path):
    kw = dict(n_boot=0, mc_samples=5, make_plots=False, seed=3)
    a = run_reliability_analysis(world["ckpt"], world["index"], tmp_path / "a", **kw)
    b = run_reliability_analysis(world["ckpt"], world["index"], tmp_path / "b", **kw)
    pd.testing.assert_frame_equal(a.per_image, b.per_image)
    pd.testing.assert_frame_equal(a.calibration.drop(columns=["ece_ci95", "brier_ci95"]), b.calibration.drop(columns=["ece_ci95", "brier_ci95"]))


def test_engine_probabilities_match_training_run_val_predictions(world, tmp_path):
    r = run_reliability_analysis(world["ckpt"], world["index"], tmp_path / "c", n_boot=0, mc_samples=3, make_plots=False)
    pred = pd.read_csv(world["run"] / "predictions.csv")
    val = pred[pred["split"] == "val"].set_index("image_id")["p_hcc"]
    assert len(val) == len(r.per_image) > 0
    got = r.per_image.set_index("image_id")["p"].reindex(val.index)
    assert np.allclose(got.to_numpy(), val.to_numpy(), atol=1e-4)        # same model, same images, same probabilities


def test_test_split_needs_final_report_and_uses_val_temperature(world, tmp_path):
    with pytest.raises(ValueError, match="final_report"):
        run_reliability_analysis(world["ckpt"], world["index"], tmp_path / "t", split="test", make_plots=False)
    r = run_reliability_analysis(world["ckpt"], world["index"], tmp_path / "t2", split="test", final_report=True, n_boot=0,
                                 mc_samples=3, make_plots=False)
    assert r.manifest["scope"].startswith("FINAL") and r.manifest["temperature"]["mode"] == "val_fit_test_apply"
    assert r.manifest["temperature"]["n_val_images"] > 0


def _cli(*args):
    return subprocess.run([sys.executable, "scripts/reliability.py", *map(str, args)], cwd=REPO, capture_output=True, text=True)


def test_cli_exit_codes(world, tmp_path):
    base = ["--checkpoint", world["ckpt"], "--index-csv", world["index"]]
    ok = _cli(*base, "--out-dir", tmp_path / "cli", "--n-boot", 0, "--mc-samples", 3, "--no-plots")
    assert ok.returncode == 0 and "NOT A RESULT" in ok.stdout
    bad = _cli(*base, "--out-dir", tmp_path / "x", "--split", "test")
    assert bad.returncode == 2 and "final_report" in bad.stderr
    assert _cli("--checkpoint", "missing.pt", "--index-csv", world["index"], "--out-dir", tmp_path / "y").returncode == 2
