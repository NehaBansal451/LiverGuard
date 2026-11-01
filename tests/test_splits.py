import pandas as pd
import pytest

from liverguard.data.config import DocumentedReference
from liverguard.data.indexer import build_index
from liverguard.data.splits import build_patient_splits, split_manifest_id, verify_patient_splits
from tests.synthetic import LAYOUT_B, build_tree


def _verify(cfg):
    frame = build_index(cfg).frame
    return frame, verify_patient_splits(frame, cfg)


def test_clean_split_has_no_leakage(tmp_path, make_cfg):
    build_tree(tmp_path)
    frame, rep = _verify(make_cfg(tmp_path))
    assert rep.ok and "no_patient_leakage" in rep.codes("info")
    assert rep.stats["per_split"]["train"]["patients"] == 42
    assert rep.stats["per_split"]["val"]["patients"] == 12
    assert rep.stats["per_split"]["test"]["patients"] == 6
    assert rep.stats["split_id"] == split_manifest_id(frame)
    assert rep.stats["patients_with_both_variants"] == 30


def test_split_id_is_stable_and_sensitive(tmp_path, make_cfg):
    build_tree(tmp_path)
    frame, _ = _verify(make_cfg(tmp_path))
    a = split_manifest_id(frame)
    assert a == split_manifest_id(frame.copy())
    changed = frame.copy()
    changed.loc[0, "split"] = "test" if changed.loc[0, "split"] != "test" else "train"
    assert split_manifest_id(changed) != a


def test_within_variant_leakage_is_an_error(tmp_path, make_cfg):
    build_tree(tmp_path, leak=True)
    _, rep = _verify(make_cfg(tmp_path))
    issue = rep.get("patient_leakage_clean")
    assert not rep.ok and issue.count == 1 and issue.examples == ["H000 [test+train]"]


def test_cross_variant_overlap_is_only_a_warning(tmp_path, make_cfg):
    build_tree(tmp_path, cross_variant_leak=True)
    _, rep = _verify(make_cfg(tmp_path))
    assert rep.ok and rep.get("cross_variant_split_overlap").count == 1
    assert "no_patient_leakage" not in rep.codes()


def test_no_split_information_is_reported_not_invented(tmp_path, make_cfg):
    build_tree(tmp_path, layout=LAYOUT_B)
    csv = tmp_path / "metadata.csv"
    pd.read_csv(csv).drop(columns=["split", "variant"]).to_csv(csv, index=False)
    _, rep = _verify(make_cfg(tmp_path))
    assert rep.stats["has_official_split"] is False and "no_split_information" in rep.codes("info")


def test_clean_split_reference_check(tmp_path, make_cfg):
    build_tree(tmp_path)
    cfg = make_cfg(tmp_path)
    frame = build_index(cfg).frame
    obs = int(((frame["split"] == "train") & (frame["variant"] == "clean") & (frame["class_label"] == "HCC")).sum())
    cfg.documented_reference = DocumentedReference(clean_split_images={"train": {"HCC": obs}})
    assert "split_reference_match" in verify_patient_splits(frame, cfg).codes("info")
    cfg.documented_reference = DocumentedReference(clean_split_images={"train": {"HCC": obs + 1}})
    assert "split_reference_mismatch" in verify_patient_splits(frame, cfg).codes("warning")


def test_rebuilt_split_is_patient_grouped_stratified_and_seeded(tmp_path, make_cfg):
    build_tree(tmp_path)
    cfg = make_cfg(tmp_path)
    frame = build_index(cfg).frame
    split = build_patient_splits(frame, cfg, seed=1)
    assert split.notna().all() and set(split) == {"train", "val", "test"}
    rebuilt = frame.assign(split=split)
    assert verify_patient_splits(rebuilt, cfg).ok
    assert (rebuilt.groupby("patient_key")["split"].nunique() == 1).all()   # no patient straddles splits
    patients = rebuilt.drop_duplicates("patient_key")
    frac = patients["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.7) < 0.1 and abs(frac["test"] - 0.1) < 0.06
    for s in ("train", "val", "test"):
        assert set(rebuilt.loc[rebuilt["split"] == s, "class_label"]) == {"HCC", "Hemangioma"}
    assert split.equals(build_patient_splits(frame, cfg, seed=1))
    assert not split.equals(build_patient_splits(frame, cfg, seed=2))


def test_rebuild_refuses_ambiguous_patients(tmp_path, make_cfg):
    build_tree(tmp_path)
    cfg = make_cfg(tmp_path)
    frame = build_index(cfg).frame
    frame.loc[frame["patient_key"] == "H000", "label"] = [0] + [1] * ((frame["patient_key"] == "H000").sum() - 1)
    with pytest.raises(ValueError, match="several class labels"):
        build_patient_splits(frame, cfg)
