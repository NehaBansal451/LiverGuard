"""End-to-end training runs on SYNTHETIC data. These test the pipeline, not scientific performance."""
import json

import numpy as np
import pandas as pd
import pytest

from liverguard.data.audit import run_audit, write_audit
from liverguard.ml import trainer as T
from liverguard.ml.config import TrainConfig
from liverguard.ml.data import LeakageError
from liverguard.ml.metrics import compute_metrics, youden_threshold
from tests.synthetic import build_tree


@pytest.fixture(scope="module")
def signal_index(tmp_path_factory):
    from liverguard.data.config import DocumentedReference, load_config
    base = tmp_path_factory.mktemp("sig")
    build_tree(base / "d", n_per_class=40, signal=True)
    cfg = load_config(data_root=base / "d")
    cfg.documented_reference = DocumentedReference()
    return write_audit(run_audit(cfg), base / "out")["index"]


def _cfg(**kw) -> TrainConfig:
    d = dict(arch="tiny_cnn", pretrained=False, experiment_kind="smoke", image_size=32, batch_size=16,
             epochs=12, freeze_epochs=1, lr=3e-3, num_workers=0, bootstrap_samples=100, patience=12,
             dropout=0.1, amp=False)
    d.update(kw)
    return TrainConfig(**d)


def test_end_to_end_run_writes_all_artifacts_and_learns(signal_index, tmp_path):
    m = T.run_experiment(_cfg(), signal_index, tmp_path / "run")
    out = tmp_path / "run"
    for f in ("best.pt", "history.csv", "predictions.csv", "manifest.json", "training_curves.png",
              "confusion_matrix_test.png", "roc_test.png"):
        assert (out / f).stat().st_size > 0, f
    assert json.loads((out / "manifest.json").read_text())["dataset"]["patient_leakage_check"] == "passed"
    assert m["valid_as_result"] is False and "NOT A RESULT" in m["watermark"]
    assert m["results"]["val"]["auroc"] > 0.9 and m["results"]["test"]["auroc"] > 0.9   # the optimiser fits
    # predictions.csv agrees with the reported test metrics
    pr = pd.read_csv(out / "predictions.csv")
    t = pr[pr["split"] == "test"]
    assert compute_metrics(t["label"], t["p_hcc"])["auroc"] == pytest.approx(m["results"]["test"]["auroc"])
    # no patient shared between the splits the model saw
    assert (pr.groupby("patient_key")["split"].nunique() == 1).all()
    # the second operating point is fitted on VALIDATION predictions only
    v = pr[pr["split"] == "val"]
    assert m["results"]["val_youden_threshold"] == pytest.approx(youden_threshold(v["label"], v["p_hcc"]))
    assert m["results"]["test_at_val_youden"]["threshold"] == pytest.approx(m["results"]["val_youden_threshold"])
    hist = pd.read_csv(out / "history.csv")
    assert hist["phase"].iloc[0] == "frozen" and (hist["phase"].iloc[1:] == "finetune").all()


def test_same_seed_reproduces_and_different_seed_differs(signal_index, tmp_path):
    a = T.run_experiment(_cfg(epochs=3), signal_index, tmp_path / "a")["results"]["test"]["auroc"]
    b = T.run_experiment(_cfg(epochs=3), signal_index, tmp_path / "b")["results"]["test"]["auroc"]
    c = T.run_experiment(_cfg(epochs=3, seed=7), signal_index, tmp_path / "c")["results"]["test"]["auroc"]
    assert a == pytest.approx(b, abs=1e-6)
    assert a != pytest.approx(c, abs=1e-9)


def test_patient_shuffled_null_control_is_not_learnable(signal_index, tmp_path):
    """Labels permuted across patients: the toy signal no longer predicts the label, so performance on held-out
    patients must collapse. If this ever scores high, there is leakage or a label bug in the pipeline."""
    real = T.run_experiment(_cfg(), signal_index, tmp_path / "real")["results"]["test"]["auroc"]
    null = T.run_experiment(_cfg(label_control="patient_shuffle"), signal_index, tmp_path / "null")
    assert null["valid_as_result"] is False and "NULL CONTROL" in null["watermark"]
    assert real > 0.9 and null["results"]["test"]["auroc"] < 0.8


def test_leaky_split_is_refused_before_any_training(tmp_path, make_cfg):
    build_tree(tmp_path / "d", leak=True, signal=True)
    idx = write_audit(run_audit(make_cfg(tmp_path / "d")), tmp_path / "o")["index"]
    with pytest.raises(LeakageError):
        T.run_experiment(_cfg(epochs=1), idx, tmp_path / "run")
    assert not (tmp_path / "run" / "best.pt").exists()


def test_test_split_is_only_loaded_after_training(signal_index, tmp_path, monkeypatch):
    """The test images must not even be decoded until the best-validation checkpoint has been restored."""
    events = []
    orig = T.UltrasoundDataset.__init__

    def spy(self, frame, *a, **k):
        events.append(("dataset", sorted(frame["split"].unique())))
        orig(self, frame, *a, **k)
    monkeypatch.setattr(T.UltrasoundDataset, "__init__", spy)
    orig_run = T._run_epoch

    def spy_epoch(*a, **k):
        events.append(("epoch", None))
        return orig_run(*a, **k)
    monkeypatch.setattr(T, "_run_epoch", spy_epoch)
    T.run_experiment(_cfg(epochs=2), signal_index, tmp_path / "run")
    test_pos = events.index(("dataset", ["test"]))
    last_epoch = max(i for i, e in enumerate(events) if e[0] == "epoch")
    assert test_pos > last_epoch


@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet50"])
def test_real_architectures_train_end_to_end_from_scratch(arch, signal_index, tmp_path):
    cfg = _cfg(arch=arch, image_size=64, batch_size=16, epochs=2, freeze_epochs=1, lr=1e-3, bootstrap_samples=20)
    m = T.run_experiment(cfg, signal_index, tmp_path / arch)
    assert m["model"]["weights_source"] == "random_init"
    assert np.isfinite(m["results"]["test"]["auroc"]) and (tmp_path / arch / "best.pt").exists()
    from liverguard.ml.models import load_checkpoint
    model, meta = load_checkpoint(tmp_path / arch / "best.pt")
    assert meta["arch"] == arch and meta["positive_class"] == "HCC" and meta["split_id"] == m["dataset"]["split_id"]


def test_cli_smoke_run_and_clean_error_exit(signal_index, tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]

    def run(*extra, cfg="configs/smoke_tiny.yaml"):
        return subprocess.run([sys.executable, "scripts/train.py", "--config", cfg, "--index-csv",
                               str(signal_index), "--out-dir", str(tmp_path / "r"), *extra],
                              cwd=repo, capture_output=True, text=True)
    ok = run("--epochs", "2")
    assert ok.returncode == 0, ok.stderr[-800:]
    assert "SMOKE TEST" in ok.stdout and "test: AUROC" in ok.stdout
    # a missing weights file must end in a clean exit code 2 with a message, not a traceback
    bad = run("--pretrained-weights-path", str(tmp_path / "missing.pth"), cfg="configs/baseline_resnet50.yaml")
    assert bad.returncode == 2 and "ERROR" in bad.stderr and "Traceback" not in bad.stderr


def test_checkpoint_selection_breaks_auroc_ties_on_validation_loss():
    f = T.is_improvement
    assert f(0.90, 0.85, 0.7, 0.5, 0.0)                    # clearly better score
    assert not f(0.84, 0.85, 0.1, 0.5, 0.0)                # worse score never wins on loss alone
    assert f(1.0, 1.0, 0.30, 0.92, 0.0)                    # AUROC plateau: lower val loss wins
    assert not f(1.0, 1.0, 0.95, 0.92, 0.0)                # plateau, higher loss: keep the earlier epoch
    assert not f(0.851, 0.85, 0.1, 0.5, 0.01)              # min_delta still respected for strict gains
