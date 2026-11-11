"""M6: interventions, robust augmentation, training hooks, weakness diagnosis, paired comparison, selection protocol."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

import liverguard.improve.augment as aug_mod
from liverguard.data.audit import run_audit, write_audit
from liverguard.data.config import DocumentedReference, load_config
from liverguard.improve import compare as C
from liverguard.improve.ablation import Workspace, model_name
from liverguard.improve.augment import RobustAugment
from liverguard.improve.interventions import INTERVENTIONS, conditions_of, intervention_for
from liverguard.improve.weakness import diagnose, write_report
from liverguard.ml import trainer as T
from liverguard.ml.config import TrainConfig
from liverguard.ml.data import UltrasoundDataset
from liverguard.reliability import calibration as cal
from liverguard.robustness.config import load_robustness_config
from tests.synthetic import build_tree

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ interventions
def test_every_intervention_condition_exists_in_the_audit_config_and_is_targetable():
    names = {c.name for c in load_robustness_config().conditions}
    for iv, cs in INTERVENTIONS.items():
        assert set(cs) <= names, iv
    assert set(INTERVENTIONS["combined"]) == names                       # combined covers everything the audit attacks
    for c in names:
        assert intervention_for(c) in INTERVENTIONS and c in conditions_of(intervention_for(c))
    with pytest.raises(ValueError, match="Unknown intervention"):
        conditions_of("nope")


# ------------------------------------------------------------------ augmentation
def _img(seed=0, size=32):
    return Image.fromarray(np.random.default_rng(seed).integers(20, 235, (size, size, 3), dtype=np.uint8))


def test_baseline_and_zero_probability_leave_the_image_untouched():
    im = _img()
    assert RobustAugment("baseline")(im, "k") is im
    torch.manual_seed(0)
    assert np.array_equal(np.asarray(RobustAugment("noise", p=1e-12)(im, "k")), np.asarray(im))


def test_augmentation_changes_the_image_never_mutates_the_original_and_is_seed_deterministic():
    im, before = _img(), np.asarray(_img()).copy()
    torch.manual_seed(3)
    a = np.asarray(RobustAugment("combined", p=1.0)(im, "k"))
    assert not np.array_equal(a, before) and np.array_equal(np.asarray(im), before)
    torch.manual_seed(3)
    b = np.asarray(RobustAugment("combined", p=1.0)(im, "k"))
    assert np.array_equal(a, b)
    torch.manual_seed(4)
    assert not np.array_equal(a, np.asarray(RobustAugment("combined", p=1.0)(im, "k")))


def test_sampled_severity_stays_inside_the_plausible_tier_and_ids_are_domain_separated(monkeypatch):
    seen = []
    real = aug_mod.apply_distortion

    def spy(img, func, param, *, seed, image_id, condition):
        seen.append((condition, param, image_id))
        return real(img, func, param, seed=seed, image_id=image_id, condition=condition)
    monkeypatch.setattr(aug_mod, "apply_distortion", spy)
    cfg = load_robustness_config()
    by = {c.name: c for c in cfg.conditions}
    a = RobustAugment("combined", p=1.0, rob_cfg=cfg)
    torch.manual_seed(0)
    for i in range(40):
        a(_img(i), f"val_set/Clean/HCC/H001/{i}.png")
    assert len(seen) == 40 * 12
    for cond, param, image_id in seen:
        lo, hi = sorted([by[cond].params[0], by[cond].params[cfg.plausible_max_level - 1]])
        assert lo - 1e-9 <= param <= hi + 1e-9, (cond, param)           # never a stress-tier value
        assert image_id.startswith("train/")                           # keyed apart from the audit's image ids
    # severities are continuous (not just the discrete audit levels)
    assert len({round(p, 6) for c, p, _ in seen if c == "gaussian_noise"}) > 20


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError):
        RobustAugment("noise", p=0)
    with pytest.raises(ValueError, match="Unknown intervention"):
        RobustAugment("nonsense")
    cfg = load_robustness_config().select(["jpeg"])
    with pytest.raises(ValueError, match="lacks conditions"):
        RobustAugment("noise", rob_cfg=cfg)


# ------------------------------------------------------------------ data + trainer hooks
@pytest.fixture(scope="module")
def m6(tmp_path_factory):
    base = tmp_path_factory.mktemp("m6")
    build_tree(base / "d", n_per_class=40, signal=True)
    dcfg = load_config(data_root=base / "d")
    dcfg.documented_reference = DocumentedReference()
    index = write_audit(run_audit(dcfg), base / "audit")["index"]
    cfg = TrainConfig(arch="tiny_cnn", pretrained=False, experiment_kind="smoke", image_size=32, batch_size=16, epochs=8,
                      freeze_epochs=1, lr=3e-3, num_workers=0, bootstrap_samples=0, patience=8, dropout=0.1, amp=False)
    return {"base": base, "index": index, "cfg": cfg}


def test_dataset_applies_robust_augmentation_only_when_given(m6):
    from liverguard.ml.data import load_split_frame
    fr = load_split_frame(m6["index"])
    part = fr[fr["split"] == "train"].head(6).reset_index(drop=True)
    plain = UltrasoundDataset(part, 32, False, True)
    torch.manual_seed(0)
    rob = UltrasoundDataset(part, 32, False, True, robust=RobustAugment("combined", p=1.0))
    assert all(torch.equal(plain[i][0], UltrasoundDataset(part, 32, False, True)[i][0]) for i in range(3))
    assert any(not torch.equal(plain[i][0], rob[i][0]) for i in range(6))
    assert all(plain[i][1] == rob[i][1] for i in range(6))                  # labels never change


def test_training_with_an_intervention_records_it_and_can_skip_the_test_split(m6, tmp_path):
    cfg = m6["cfg"].model_copy(update={"intervention": "noise", "evaluate_test": False, "epochs": 3})
    man = T.run_experiment(cfg, m6["index"], tmp_path / "r")
    assert "test" not in man["results"] and "test_at_val_youden" not in man["results"] and "val" in man["results"]
    assert "NOT evaluated at training time" in man["test_set_policy"]
    assert not (tmp_path / "r" / "confusion_matrix_test.png").exists()
    pred = pd.read_csv(tmp_path / "r" / "predictions.csv")
    assert set(pred["split"]) == {"val"}                                    # no test prediction was ever written
    meta = torch.load(tmp_path / "r" / "best.pt", weights_only=True)["meta"]
    assert meta["intervention"] == "noise" and meta["intervention_conditions"] == INTERVENTIONS["noise"]
    assert man["config"]["intervention"] == "noise"


def test_default_training_is_unchanged_baseline_with_test_evaluation(m6, tmp_path):
    man = T.run_experiment(m6["cfg"].model_copy(update={"epochs": 2}), m6["index"], tmp_path / "b")
    assert "test" in man["results"] and man["config"]["intervention"] == "baseline"


# ------------------------------------------------------------------ weakness diagnosis (hand-built audit tables)
def _audit_dir(path: Path, rows: list[dict], split="val", valid=True):
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path / "summary_by_level.csv", index=False)
    (path / "audit_manifest.json").write_text(json.dumps({"split": split, "valid_as_result": valid, "watermark": None,
                                                         "dataset": {"n_images": 100, "n_patients": 50}}))
    return path


def _row(cond, level, d_auroc=0.0, hi=0.01, pfr=0.0, pfr_lo=0.0, fid=False, tier=None):
    return {"condition": cond, "level": level, "tier": tier or ("plausible" if level <= 3 else "stress"), "d_auroc": d_auroc,
            "d_auroc_lo": d_auroc - 0.02, "d_auroc_hi": hi, "pfr": pfr, "pfr_lo": pfr_lo, "fidelity_flag": fid}


def _toy_rows():
    r = []
    for l in (1, 2, 3, 4, 5):
        r.append(_row("brightness_dark", l, d_auroc=-0.10 if l >= 2 else 0.0, hi=-0.04 if l >= 2 else 0.01))   # AUROC weakness
        r.append(_row("gamma_high", l, pfr=0.2, pfr_lo=0.1))                                                       # flip weakness
        r.append(_row("gaussian_noise", l, d_auroc=-0.5 if l >= 4 else 0.0, hi=-0.4 if l >= 4 else 0.01))         # stress-only failure
        r.append(_row("jpeg", l))                                                                                  # fine
        r.append(_row("rotation", l, d_auroc=-0.3, hi=-0.2, pfr=0.3, pfr_lo=0.2, fid=(l <= 3)))                   # only fidelity-flagged
        r.append(_row("contrast_low", l, d_auroc=-0.04, hi=0.02))                                                  # drop but CI includes 0
    return r


def test_diagnosis_applies_the_documented_rule(tmp_path):
    d = diagnose([_audit_dir(tmp_path / "a", _toy_rows())])
    assert set(d["weaknesses"]) == {"brightness_dark", "gamma_high"}
    byc = {c["condition"]: c for c in d["conditions"]}
    assert not byc["gaussian_noise"]["weakness"]                           # stress-tier failures never create a weakness
    assert byc["rotation"]["excluded_fidelity"] and not byc["rotation"]["weakness"]
    assert not byc["contrast_low"]["weakness"]                             # CI includes 0
    assert d["targeted"] == ["brightness", "gamma"] or d["targeted"] == ["gamma", "brightness"]
    assert d["plan"][0] == "baseline" and "combined" in d["plan"]
    ctl = d["negative_control"]
    assert ctl and not (set(INTERVENTIONS[ctl]) & set(d["weaknesses"]))    # control trains only on non-weaknesses
    md = write_report(d, tmp_path / "out").read_text()
    assert "WEAKNESS" in md and (tmp_path / "out" / "plan.json").exists()


def test_diagnosis_majority_rule_and_split_guard_and_no_weakness_case(tmp_path):
    clean_rows = [_row(c, l) for c in ("brightness_dark", "gamma_high", "gaussian_noise", "jpeg", "rotation", "contrast_low")
                  for l in (1, 2, 3, 4, 5)]
    a = _audit_dir(tmp_path / "a", _toy_rows())
    b = _audit_dir(tmp_path / "b", clean_rows)
    c = _audit_dir(tmp_path / "c", clean_rows)
    d = diagnose([a, b, c])                                                # flagged in 1 of 3 audits -> not a weakness
    assert "brightness_dark" not in d["weaknesses"]
    none = diagnose([b])
    assert none["weaknesses"] == [] and none["plan"] == ["baseline"] + ([none["negative_control"]] if none["negative_control"] else [])
    assert "nothing to fix" in none["note"]
    with pytest.raises(ValueError, match="different conditions"):
        diagnose([a, _audit_dir(tmp_path / "x", [_row("jpeg", l) for l in (1, 2, 3)])])
    with pytest.raises(ValueError, match="VALIDATION"):
        diagnose([_audit_dir(tmp_path / "t", clean_rows, split="test")])
    with pytest.raises(ValueError):
        diagnose([])
    nm = diagnose([_audit_dir(tmp_path / "nm", clean_rows, valid=False)])
    assert nm["valid_as_result"] is False
    assert "NOT A RESULT" in write_report(nm, tmp_path / "o2").read_text()


# ------------------------------------------------------------------ comparison maths
def test_fast_auroc_and_ece_agree_with_reference_implementations():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=200) < 0.4).astype(int)
    S = np.vstack([rng.normal(size=200) + y, np.round(rng.uniform(size=200), 1), rng.uniform(size=200)])   # incl. heavy ties
    for k in range(3):
        assert C.auroc_rows(y, S)[k] == pytest.approx(roc_auc_score(y, S[k]))
    P = np.clip(S, 0.01, 0.99)
    for k in range(3):
        assert C.ece_rows(y, P)[k] == pytest.approx(cal.expected_calibration_error(y, P[k], 10, "width"))
    assert np.isnan(C.auroc_rows(np.ones(5, dtype=int), np.random.rand(1, 5))[0])


def _eval(name, iv, seed, y, groups, p_clean, P, tiers, conds, trained=(), thr=0.5):
    keys = [(c, 1 + sum(1 for k in range(i) if conds[k] == c)) for i, c in enumerate(conds)]
    ids = np.array([f"i{i:03d}" for i in range(len(y))])
    return C.ModelEval(name, iv, seed, frozenset(trained), thr, ids, y, groups, p_clean, keys, np.array(tiers), np.array(conds), P)


def _world(n=120, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    groups = np.repeat(np.arange(n // 2).astype(str), 2)
    signal = np.where(y == 1, 0.8, 0.2)
    return y, groups, signal, rng


def test_metrics_at_hand_checked_and_trained_untrained_split():
    y, g, sig, rng = _world()
    P = np.vstack([sig, 1 - sig, sig, sig])                                # cond A (perfect), cond B (inverted), A level 2, stress
    m = _eval("m", "noise", 0, y, g, sig, P, ["plausible", "plausible", "plausible", "stress"], ["A", "B", "A", "A"], trained=["A"])
    r = C.metrics_at(m)
    assert r["clean_auroc"] == 1.0 and r["clean_accuracy"] == 1.0 and r["clean_sensitivity"] == 1.0
    assert r["plaus_auroc"] == pytest.approx((1 + 0 + 1) / 3) and r["plaus_worst_auroc"] == 0.0
    assert r["plaus_auroc_trained"] == 1.0 and r["plaus_auroc_untrained"] == 0.0
    assert r["plaus_flip_rate"] == pytest.approx((0 + 1 + 0) / 3)          # B flips every decision
    assert r["stress_auroc"] == 1.0 and r["clean_brier"] == pytest.approx(0.04)


def test_paired_bootstrap_zero_for_identical_models_and_positive_for_a_better_model():
    y, g, sig, rng = _world(160)
    good = np.vstack([sig, sig, sig])
    bad = np.vstack([rng.uniform(size=len(y)) for _ in range(3)])
    tiers, conds = ["plausible"] * 3, ["A", "B", "C"]
    ref = _eval("b", "baseline", 0, y, g, sig, bad, tiers, conds)
    same = _eval("s", "noise", 0, y, g, sig, bad, tiers, conds, trained=["A"])
    d = C.paired_bootstrap(ref, same, 60, 0).set_index("metric")
    undefined = {"stress_auroc", "stress_flip_rate"}                      # no stress-tier conditions in this toy world
    assert set(d.index[d["diff"].isna()]) == undefined
    d = d.drop(index=list(undefined))
    assert (d["diff"].abs() < 1e-12).all() and (d["lo"].abs() < 1e-12).all() and (d["hi"].abs() < 1e-12).all()
    better = _eval("g", "noise", 0, y, g, sig, good, tiers, conds, trained=["A"])
    d = C.paired_bootstrap(ref, better, 80, 0).set_index("metric")
    assert d.loc["plaus_auroc", "diff"] > 0.2 and d.loc["plaus_auroc", "lo"] > 0 and d.loc["clean_auroc", "diff"] == 0
    assert C.verdict(pd.DataFrame([d.loc["plaus_auroc"]]), "plaus_auroc") == "improved"
    assert C.verdict(pd.DataFrame([d.loc["plaus_flip_rate"]]), "plaus_flip_rate") in {"improved", "inconclusive"}


def test_verdict_directions_and_missing_ci():
    f = lambda lo, hi: pd.DataFrame({"diff": [(lo + hi) / 2], "lo": [lo], "hi": [hi]})   # noqa: E731
    assert C.verdict(f(0.01, 0.05), "plaus_auroc") == "improved" and C.verdict(f(-0.05, -0.01), "plaus_auroc") == "worse"
    assert C.verdict(f(-0.05, -0.01), "plaus_flip_rate") == "improved"       # lower is better
    assert C.verdict(f(-0.01, 0.05), "plaus_auroc") == "inconclusive"
    two = pd.DataFrame({"diff": [0.03, 0.01], "lo": [0.01, -0.01], "hi": [0.05, 0.03]})
    assert C.verdict(two, "plaus_auroc") == "inconclusive"                   # must hold in EVERY seed
    assert C.verdict(pd.DataFrame({"diff": [0.1], "lo": [np.nan], "hi": [np.nan]}), "plaus_auroc") == "no_ci"


def test_comparison_refuses_different_images_or_conditions():
    y, g, sig, rng = _world(40)
    P = np.vstack([sig])
    a = _eval("a", "baseline", 0, y, g, sig, P, ["plausible"], ["A"])
    b = _eval("b", "noise", 0, y, g, sig, P, ["plausible"], ["A"])
    b.y = b.y[::-1].copy()
    with pytest.raises(ValueError, match="identical images"):
        C.check_same_images([a, b])
    c = _eval("c", "noise", 0, y, g, sig, P, ["plausible"], ["Z"])
    c.keys = [("Z", 1)]
    with pytest.raises(ValueError, match="different condition"):
        C.check_same_images([a, c])
    with pytest.raises(ValueError, match="no baseline"):
        C.compare_to_baseline([b], n_boot=0)


def _summary(rows):
    out = []
    for iv, clean_d, clean_v, pl_d, pl_v in rows:
        for m, d, v in (("clean_auroc", clean_d, clean_v), ("plaus_auroc", pl_d, pl_v)):
            out.append({"intervention": iv, "metric": m, "mean_diff": d, "verdict": v})
    return pd.DataFrame(out)


def test_selection_rule_picks_best_allowed_candidate_and_falls_back_to_baseline():
    nc = {"a": 2, "b": 1, "c": 3, "d": 1}
    s = _summary([("a", 0.0, "inconclusive", 0.05, "improved"), ("b", -0.03, "inconclusive", 0.20, "improved"),   # b loses too much clean
                  ("c", 0.0, "worse", 0.10, "improved"),                                                         # clean verdict worse
                  ("d", 0.0, "inconclusive", 0.01, "inconclusive")])                                              # not improved
    sel = C.select_final(s, nc, max_clean_drop=0.02)
    assert sel["chosen"] == "a"
    tie = _summary([("a", 0.0, "inconclusive", 0.05, "improved"), ("b", 0.0, "inconclusive", 0.05, "improved")])
    assert C.select_final(tie, nc)["chosen"] == "b"                                                              # fewer conditions wins
    none = _summary([("a", 0.0, "inconclusive", 0.05, "inconclusive")])
    assert C.select_final(none, nc)["chosen"] == "baseline"


# ------------------------------------------------------------------ protocol end to end (synthetic; NOT results)
@pytest.fixture(scope="module")
def ws(m6):
    return Workspace(m6["base"] / "w", m6["cfg"], m6["index"], [1], rob_cfg=load_robustness_config(n_boot=0), n_boot=30)


def test_protocol_test_split_is_locked_until_selection_is_frozen(ws):
    d = ws.diagnose()
    assert d["plan"][0] == "baseline" and (ws.dir / "diagnosis" / "weakness_report.md").exists()
    with pytest.raises(ValueError, match="final_report"):
        ws.final(final_report=False)
    with pytest.raises(FileNotFoundError, match="selection.json"):
        ws.final(final_report=True)
    assert not (ws.dir / "audits" / "test").exists()                        # nothing touched the test split so far
    man = json.loads((ws.model_dir("baseline_s1") / "manifest.json").read_text())
    assert "test" not in man["results"] and man["config"]["evaluate_test"] is False


def test_validate_freezes_selection_then_final_reports_test(ws):
    ws.train_variants(["baseline", "noise"])
    res = ws.validate(["baseline", "noise"])
    sel = json.loads((ws.dir / "validation" / "selection.json").read_text())
    assert sel["chosen"] in {"baseline", "noise"} and sel["selected_on"] == "validation split only"
    assert set(sel["checkpoint_sha256"]) == {"baseline_s1", "noise_s1"} and "SMOKE" in sel["watermark"]
    assert not (ws.dir / "audits" / "test").exists()                        # validation never evaluated test
    for f in ("point_metrics", "paired_diffs_per_run", "paired_diffs_summary", "per_condition_auroc"):
        assert (ws.dir / "validation" / f"{f}.csv").exists()
    for f in ("tradeoff_clean_vs_robust", "paired_differences", "ablation_matrix", "degradation_baseline_vs_best"):
        assert (ws.dir / "validation" / "figures" / f"{f}.png").exists()
    rep = (ws.dir / "validation" / "report.md").read_text()
    assert "NOT A RESULT" in rep and "Limitations" in rep and "Pre-declared rule" in rep
    assert set(res["points"]["intervention"]) == {"baseline", "noise"}
    # the final stage now runs, uses the frozen selection and records its hash
    out = ws.final(final_report=True)
    fin = json.loads((ws.dir / "final_test" / "final_manifest.json").read_text())
    assert out["selection"]["chosen"] == sel["chosen"] and fin["selection_file_sha256"]
    assert (ws.dir / "audits" / "test" / "baseline_s1" / "audit_manifest.json").exists()
    assert json.loads((ws.dir / "audits" / "test" / "noise_s1" / "audit_manifest.json").read_text())["split"] == "test"
    assert "frozen on validation" in (ws.dir / "final_test" / "report.md").read_text()


def test_validation_is_reproducible(ws):
    a = pd.read_csv(ws.dir / "validation" / "paired_diffs_summary.csv")
    ws.validate(["baseline", "noise"])
    b = pd.read_csv(ws.dir / "validation" / "paired_diffs_summary.csv")
    pd.testing.assert_frame_equal(a, b)


def test_final_refuses_if_a_checkpoint_changed_after_selection(ws):
    ck = ws.model_dir("noise_s1") / "best.pt"
    original = ck.read_bytes()
    try:
        ck.write_bytes(original + b"\0")
        with pytest.raises(ValueError, match="changed after selection"):
            ws.final(final_report=True)
    finally:
        ck.write_bytes(original)


def test_cli_protocol_errors_exit_2(m6, tmp_path):
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text("arch: tiny_cnn\npretrained: false\nexperiment_kind: smoke\nimage_size: 32\nepochs: 1\nnum_workers: 0\namp: false\n"
                       "bootstrap_samples: 0\n")
    r = subprocess.run([sys.executable, "scripts/robust_train.py", "final", "--workdir", str(tmp_path / "w"), "--config", str(cfgfile),
                        "--index-csv", str(m6["index"])], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 2 and "final_report" in r.stderr
    r = subprocess.run([sys.executable, "scripts/robust_train.py", "final", "--final-report", "--workdir", str(tmp_path / "w"),
                        "--config", str(cfgfile), "--index-csv", str(m6["index"])], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 2 and "selection.json" in r.stderr


def test_report_flags_gain_that_is_only_a_better_clean_model(tmp_path):
    """plaus_auroc 'improved' but the degradation gap did not improve -> the report must not present it as robustness."""
    from liverguard.improve import ablation as A
    from liverguard.ml.config import TrainConfig
    from liverguard.robustness.config import load_robustness_config

    def row(iv, metric, d, lo, hi, verdict):
        return {"intervention": iv, "metric": metric, "n_seeds": 3, "mean_diff": d, "sd_diff": 0.0, "min_diff": d, "max_diff": d,
                "ci_lo_min_seed": lo, "ci_hi_max_seed": hi, "higher_is_better": True, "verdict": verdict}
    summ = pd.DataFrame([row("noise", "plaus_auroc", 0.05, 0.02, 0.08, "improved"), row("noise", "clean_auroc", 0.05, -0.01, 0.1, "inconclusive"),
                         row("noise", "plaus_minus_clean_auroc", -0.01, -0.05, 0.03, "inconclusive")])
    cols = ["clean_auroc", "clean_accuracy", "clean_ece", "clean_brier", "plaus_auroc", "plaus_minus_clean_auroc", "plaus_worst_auroc",
            "plaus_flip_rate", "plaus_mean_abs_dp", "plaus_ece", "plaus_brier", "stress_auroc"]
    pts = pd.DataFrame([{"model": "noise_s1", "intervention": "noise", "seed": 1, "n_images": 10, "n_patients": 5, **{c: 0.5 for c in cols}}])
    sel = {"chosen": "noise", "rule": {}, "candidates": [], "negative_control": None}
    out = tmp_path / "r.md"
    A.write_report(out, "TEST", {"points": pts, "summary": summ, "per_run": pd.DataFrame()}, sel,
                   TrainConfig(arch="tiny_cnn", pretrained=False, experiment_kind="smoke"), load_robustness_config(None), [1, 2, 3])
    txt = out.read_text()
    assert "Attribution caveat" in txt and "NOT shown to be robustness" in txt
    summ.loc[summ.metric == "plaus_minus_clean_auroc", "verdict"] = "improved"
    A.write_report(out, "TEST", {"points": pts, "summary": summ, "per_run": pd.DataFrame()}, sel,
                   TrainConfig(arch="tiny_cnn", pretrained=False, experiment_kind="smoke"), load_robustness_config(None), [1, 2, 3])
    assert "Attribution caveat" not in out.read_text()
