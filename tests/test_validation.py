import pandas as pd

from liverguard.data.config import DocumentedReference
from liverguard.data.indexer import build_index
from liverguard.data.validation import inspect_images, validate_index
from tests.synthetic import LAYOUT_A, LAYOUT_B, build_tree, write_png


def _validate(cfg, **kw):
    res = build_index(cfg)
    stats = inspect_images(res.frame) if kw.pop("inspect", False) else None
    return res, validate_index(res, cfg, stats)


def test_clean_tree_has_no_errors(tmp_path, make_cfg):
    build_tree(tmp_path)
    _, rep = _validate(make_cfg(tmp_path), inspect=True)
    assert rep.ok and "no_metadata_csv" in rep.codes("warning")
    assert rep.stats["n_patients"] == 60 and rep.stats["patients_by_class"] == {"HCC": 30, "Hemangioma": 30}


def test_missing_patient_and_class_are_errors(tmp_path, make_cfg):
    build_tree(tmp_path)
    write_png(tmp_path / "stray.png", seed=999)
    _, rep = _validate(make_cfg(tmp_path))
    assert not rep.ok and {"missing_patient_id", "missing_class_label"} <= rep.codes("error")


def test_corrupt_and_zero_byte_files_detected_only_when_inspecting(tmp_path, make_cfg):
    build_tree(tmp_path)
    bad = tmp_path / "train_set/Clean/HCC/H000"
    (bad / "bad.png").write_bytes(b"this is not a png")
    (bad / "empty.png").write_bytes(b"")
    _, rep = _validate(make_cfg(tmp_path))
    assert "unreadable_images" not in rep.codes()
    _, rep = _validate(make_cfg(tmp_path), inspect=True)
    assert rep.get("unreadable_images").count == 2


def test_reference_comparison(tmp_path, make_cfg):
    build_tree(tmp_path)
    cfg = make_cfg(tmp_path)
    n = len(build_index(cfg).frame)
    cfg.documented_reference = DocumentedReference(total_images=n, images_by_class={"HCC": 1})
    _, rep = _validate(cfg)
    issue = rep.get("reference_mismatch")
    assert issue.count == 1 and "images[HCC]" in issue.examples[0]
    cfg.documented_reference = DocumentedReference(total_images=n)
    _, rep = _validate(cfg)
    assert "reference_match" in rep.codes("info")


def test_patient_with_two_classes_warns(tmp_path, make_cfg):
    build_tree(tmp_path)
    write_png(tmp_path / "train_set/Clean/Hemangioma/H000/x.png", seed=998)
    _, rep = _validate(make_cfg(tmp_path))
    assert rep.get("patient_multiple_classes").examples == ["H000"]


def test_csv_missing_required_column_is_error(tmp_path, make_cfg):
    build_tree(tmp_path, layout=LAYOUT_B)
    csv = tmp_path / "metadata.csv"
    pd.read_csv(csv).drop(columns="patient_id").to_csv(csv, index=False)
    _, rep = _validate(make_cfg(tmp_path))
    assert "csv_missing_columns" in rep.codes("error")


def test_files_without_metadata_warn(tmp_path, make_cfg):
    build_tree(tmp_path, layout=LAYOUT_B)
    write_png(tmp_path / "HCC/H000/extra.png", seed=997)
    _, rep = _validate(make_cfg(tmp_path))
    assert rep.get("files_without_metadata").count == 1


def test_markdown_and_json_render(tmp_path, make_cfg):
    build_tree(tmp_path, layout=LAYOUT_A)
    _, rep = _validate(make_cfg(tmp_path))
    assert "PASS" in rep.to_markdown() and '"issues"' in rep.to_json()
