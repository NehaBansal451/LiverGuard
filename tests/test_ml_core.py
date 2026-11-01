"""Metrics, config, data loading (patient separation) and model construction."""
import numpy as np
import pandas as pd
import pytest
import torch

from liverguard.ml import models as M
from liverguard.ml.config import TrainConfig, load_train_config
from liverguard.ml.data import (LeakageError, UltrasoundDataset, apply_patient_label_shuffle,
                                assert_patient_separation, compute_class_weights, load_split_frame, make_loader)
from liverguard.ml.metrics import cluster_bootstrap_ci, compute_metrics, youden_threshold
from tests.synthetic import build_tree

# ---------------------------------------------------------------- metrics
def test_metrics_hand_computed():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([.9, .8, .4, .3, .6, .2, .1, .05])
    m = compute_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == [[3, 1], [2, 2]]          # [[tn, fp], [fn, tp]]
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.75
    assert m["precision"] == pytest.approx(2 / 3) and m["f1"] == pytest.approx(4 / 7)
    # AUROC by hand: 14 of the 4x4 = 16 (positive, negative) pairs are ranked correctly
    assert m["accuracy"] == 0.625 and m["auroc"] == pytest.approx(14 / 16)


def test_metrics_single_class_is_nan_not_error():
    m = compute_metrics(np.ones(5), np.linspace(.6, .9, 5))        # all positives, all predicted positive
    assert np.isnan(m["auroc"]) and np.isnan(m["specificity"]) and m["sensitivity"] == 1.0


def test_cluster_bootstrap_resamples_patients_and_is_seeded():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 3)
    y = np.repeat(np.arange(40) % 2, 3)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 120), 0, 1)
    a = cluster_bootstrap_ci(y, p, groups, n_boot=200, seed=1)
    b = cluster_bootstrap_ci(y, p, groups, n_boot=200, seed=1)
    assert a == b and a["n_valid_replicates"] > 150
    lo, hi = a["auroc"]
    assert lo < compute_metrics(y, p)["auroc"] < hi
    # patient-level CI must be wider than a naive image-level one (images are correlated within patient)
    naive = cluster_bootstrap_ci(y, p, np.arange(120), n_boot=200, seed=1)["auroc"]
    assert hi - lo >= (naive[1] - naive[0]) * 0.9
    assert all(np.isnan(v) for v in cluster_bootstrap_ci(y, p, groups, n_boot=0)["auroc"])


def test_youden_threshold_fitted_on_scores_not_fixed_at_half():
    # rank-perfect but low-probability scores (the miscalibrated case): 0.5 would call everything negative
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([.07, .08, .09, .09, .12, .15, .18, .20])
    thr = youden_threshold(y, p)
    assert 0.09 < thr <= 0.12
    m = compute_metrics(y, p, thr)
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0
    assert compute_metrics(y, p, 0.5)["sensitivity"] == 0.0
    assert youden_threshold(np.ones(4), np.array([.1, .2, .3, .4])) == 0.5      # single class -> neutral default


# ---------------------------------------------------------------- config
def test_config_validation_and_watermarks():
    assert TrainConfig().valid_as_result and TrainConfig().watermark is None
    s = TrainConfig(experiment_kind="smoke")
    assert not s.valid_as_result and "NOT A RESULT" in s.watermark
    assert not TrainConfig(label_control="patient_shuffle").valid_as_result
    with pytest.raises(ValueError):
        TrainConfig(arch="vgg16")
    with pytest.raises(ValueError):
        TrainConfig(unknown_option=1)


def test_shipped_configs_load_and_override():
    for name in ("baseline_efficientnet_b0", "baseline_resnet50", "smoke_tiny"):
        load_train_config(f"configs/{name}.yaml")
    cfg = load_train_config("configs/baseline_resnet50.yaml", seed=7, epochs=None)
    assert cfg.seed == 7 and cfg.epochs == 30 and cfg.arch == "resnet50"


# ---------------------------------------------------------------- data / patient separation
@pytest.fixture
def index_csv(tmp_path, make_cfg):
    from liverguard.data.audit import run_audit, write_audit
    root = tmp_path / "d"
    build_tree(root, signal=True)
    return write_audit(run_audit(make_cfg(root)), tmp_path / "out")["index"]


def test_load_split_frame_clean_variant_and_disjoint(index_csv):
    df = load_split_frame(index_csv, variant="clean")
    assert set(df["variant"]) == {"clean"} and set(df["split"]) == {"train", "val", "test"}
    assert (df.groupby("patient_key")["split"].nunique() == 1).all()


def test_pooling_variants_with_a_cross_variant_leak_is_refused(tmp_path, make_cfg):
    from liverguard.data.audit import run_audit, write_audit
    root = tmp_path / "d"
    build_tree(root, cross_variant_leak=True)
    idx = write_audit(run_audit(make_cfg(root)), tmp_path / "o")["index"]
    load_split_frame(idx, variant="clean")                      # Clean alone is fine
    with pytest.raises(LeakageError):
        load_split_frame(idx, variant=None)                     # pooled: H001 straddles train/test


def test_within_variant_leak_is_refused(tmp_path, make_cfg):
    from liverguard.data.audit import run_audit, write_audit
    root = tmp_path / "d"
    build_tree(root, leak=True)
    idx = write_audit(run_audit(make_cfg(root)), tmp_path / "o")["index"]
    with pytest.raises(LeakageError, match="more than one split"):
        load_split_frame(idx, variant="clean")


def test_missing_split_is_refused_with_guidance(index_csv, tmp_path):
    df = pd.read_csv(index_csv)
    df["split"] = np.nan
    p = tmp_path / "nosplit.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="no split"):
        load_split_frame(p, variant="clean")


def test_split_file_overrides_split_and_must_stay_disjoint(index_csv, tmp_path):
    df = pd.read_csv(index_csv)
    from liverguard.data.splits import build_patient_splits
    from liverguard.data.config import load_config
    cfg = load_config()
    df["patient_key"] = df["patient_key"].astype(str)
    split = build_patient_splits(df.assign(label=df["label"].astype(int)), cfg, seed=3)
    sf = tmp_path / "s.csv"
    pd.DataFrame({"image_id": df["image_id"], "split": split}).to_csv(sf, index=False)
    assert load_split_frame(index_csv, split_file=sf, variant=None).shape[0] == len(df)
    h0 = df.index[df["patient_key"] == "H000"]                    # one patient, several images
    assert len(h0) >= 2
    bad = pd.DataFrame({"image_id": df["image_id"], "split": df["split"]})
    bad.loc[h0[0], "split"], bad.loc[h0[1], "split"] = "train", "test"   # split that patient across train/test
    bad.to_csv(sf, index=False)
    with pytest.raises(LeakageError):
        load_split_frame(index_csv, split_file=sf, variant=None)


def test_label_shuffle_is_patient_level_and_keeps_proportions(index_csv):
    df = load_split_frame(index_csv, variant="clean")
    sh = apply_patient_label_shuffle(df, seed=0)
    assert (sh.groupby("patient_key")["label"].nunique() == 1).all()
    assert sh.drop_duplicates("patient_key")["label"].sum() == df.drop_duplicates("patient_key")["label"].sum()
    assert (sh["label"] != df["label"]).any()
    assert sh["label"].equals(apply_patient_label_shuffle(df, seed=0)["label"])


def test_class_weights_inverse_frequency():
    w = compute_class_weights(np.array([0] * 30 + [1] * 10), 2)       # n / (k * count)
    assert w.tolist() == pytest.approx([40 / 60, 40 / 20])


def test_dataset_shapes_and_cache_equivalence(index_csv):
    df = load_split_frame(index_csv, variant="clean").head(6)
    a = UltrasoundDataset(df, 32, augment=False, cache=True)
    b = UltrasoundDataset(df, 32, augment=False, cache=False)
    x, y, i = a[2]
    assert x.shape == (3, 32, 32) and i == 2 and y in (0, 1)
    assert torch.equal(x, b[2][0])
    xb, yb, ib = next(iter(make_loader(a, batch_size=4, shuffle=False, num_workers=0, seed=0)))
    assert xb.shape == (4, 3, 32, 32) and ib.tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------- models
@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet50", "tiny_cnn"])
def test_forward_shape_and_freeze(arch):
    model, src = M.build_model(arch, pretrained=False)
    assert src == "random_init"
    assert model.eval()(torch.randn(2, 3, 64, 64)).shape == (2, 2)
    M.set_backbone_requires_grad(model, False)
    head = {id(p) for p in M.head_module(model).parameters()}
    assert all(p.requires_grad == (id(p) in head) for p in model.parameters())
    M.set_backbone_requires_grad(model, True)
    assert all(p.requires_grad for p in model.parameters())
    g = M.param_groups(model, 1e-3, 0.1)
    assert g[0]["lr"] == 1e-3 and g[1]["lr"] == pytest.approx(1e-4)
    assert sum(len(x["params"]) for x in g) == len(list(model.parameters()))


def test_pretrained_failure_is_loud_not_silent(monkeypatch):
    real = M.tvm.resnet50

    def fake(weights=None):
        if weights is not None:
            raise OSError("no network")          # what a blocked download looks like
        return real(weights=None)
    monkeypatch.setattr(M.tvm, "resnet50", fake)
    with pytest.raises(RuntimeError, match="Refusing to fall back"):
        M.build_model("resnet50", pretrained=True)


def test_local_weights_load_backbone_only_and_mismatch_is_detected(tmp_path):
    src, _ = M.build_model("resnet50", pretrained=False)
    ref = M.tvm.resnet50(weights=None)                    # a 1000-class torchvision-style state dict
    path = tmp_path / "r50.pth"
    torch.save(ref.state_dict(), path)
    model, source = M.build_model("resnet50", pretrained=True, weights_path=path)
    assert source.startswith("local:")
    assert torch.equal(model.layer1[0].conv1.weight, ref.layer1[0].conv1.weight)        # backbone loaded
    assert model.fc[1].weight.shape == (2, 2048)                                          # head stays 2-class
    wrong = tmp_path / "eff.pth"
    torch.save(M.tvm.efficientnet_b0(weights=None).state_dict(), wrong)
    with pytest.raises(RuntimeError, match="does not match"):
        M.build_model("resnet50", pretrained=True, weights_path=wrong)


def test_checkpoint_roundtrip(tmp_path):
    model, _ = M.build_model("tiny_cnn", pretrained=False)
    M.save_checkpoint(tmp_path / "c.pt", model, {"arch": "tiny_cnn", "dropout": 0.2, "val_auroc": {"auroc": 0.5}})
    loaded, meta = M.load_checkpoint(tmp_path / "c.pt")
    x = torch.randn(2, 3, 32, 32)
    assert meta["arch"] == "tiny_cnn" and torch.allclose(model.eval()(x), loaded(x))
