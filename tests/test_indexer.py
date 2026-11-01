import pandas as pd
import pytest

from liverguard.data.config import DatasetConfig, load_config
from liverguard.data.indexer import build_index
from tests.synthetic import LAYOUT_A, LAYOUT_B, build_tree


def test_default_config_loads_and_maps_labels():
    cfg = load_config()
    assert cfg.classes == ["Hemangioma", "HCC"] and cfg.positive_class == "HCC"
    assert cfg.map_class(" hepatocellular carcinoma ") == "HCC"
    assert cfg.map_split("Train-Set") == "train"
    assert cfg.map_column("Filename") == "image_filename"
    assert cfg.map_class("nan") is None and cfg.map_class(None) is None
    assert len(cfg.documented_reference.known_discrepancies) >= 5   # documentation problems are recorded


def test_env_var_and_argument_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVERGUARD_DATA_ROOT", str(tmp_path / "env"))
    assert load_config().data_root == tmp_path / "env"
    assert load_config(data_root=tmp_path / "arg").data_root == tmp_path / "arg"


def test_bad_config_rejected():
    cfg = load_config().model_dump()
    with pytest.raises(ValueError):
        DatasetConfig(**{**cfg, "positive_class": "Cyst"})
    with pytest.raises(ValueError):
        DatasetConfig(**{**cfg, "split_fractions": {"train": 0.5, "val": 0.2, "test": 0.1}})


def test_layout_a_infers_everything_from_paths(tmp_path, make_cfg):
    info = build_tree(tmp_path, layout=LAYOUT_A)
    res = build_index(make_cfg(tmp_path))
    f = res.frame
    assert len(f) == len(info["records"]) and res.metadata_csv is None
    assert set(f["class_label"]) == {"HCC", "Hemangioma"}
    assert set(f["split"]) == {"train", "val", "test"}
    assert set(f["variant"]) == {"clean", "caliper"}
    assert f["patient_id"].notna().all()
    assert (f.loc[f["class_label"] == "HCC", "label"] == 1).all()
    assert (f.loc[f["class_label"] == "Hemangioma", "label"] == 0).all()
    truth = {r["patient"]: r["split"] for r in info["records"]}
    assert f.groupby("patient_id")["split"].first().to_dict() == truth
    assert not res.conflicts and not res.unknown_values


def test_layout_b_csv_disambiguates_repeated_filenames(tmp_path, make_cfg):
    build_tree(tmp_path, layout=LAYOUT_B)
    res = build_index(make_cfg(tmp_path))
    f = res.frame
    assert res.metadata_csv.name == "metadata.csv"
    assert f["image_filename"].duplicated().any()          # same basename in many patient dirs
    assert f["has_metadata"].all() and len(res.csv_unmatched) == 0
    assert f["split"].notna().all() and f["variant"].notna().all()
    assert "tumor_size_cm" in f.columns                    # extra metadata preserved
    assert not res.conflicts


def test_csv_aliases_unmatched_rows_and_unknown_values(tmp_path, make_cfg):
    build_tree(tmp_path, layout=LAYOUT_B)
    csv = tmp_path / "metadata.csv"
    df = pd.read_csv(csv)
    df.loc[len(df)] = ["H999", "ghost.png", "HCC", "train", "clean", "1.0"]
    df.loc[0, "class_label"] = "Cyst"
    df = df.rename(columns={"image_filename": "filename", "patient_id": "patient",
                            "class_label": "label", "split": "subset", "variant": "version"})
    df.to_csv(csv, index=False)
    res = build_index(make_cfg(tmp_path))
    assert len(res.csv_unmatched) == 1
    assert res.unknown_values["class_label"] == ["Cyst"]
    assert res.frame["split"].notna().all()                # alias columns were mapped


def test_csv_vs_path_conflict_is_recorded(tmp_path, make_cfg):
    build_tree(tmp_path, layout=LAYOUT_B)
    csv = tmp_path / "metadata.csv"
    df = pd.read_csv(csv)
    df.loc[0, "class_label"] = "Hemangioma"                # path says HCC
    df.to_csv(csv, index=False)
    res = build_index(make_cfg(tmp_path))
    assert res.conflicts["class_label"] == ["HCC/H000/0.png"]
    assert res.frame.loc[res.frame["image_id"] == "HCC/H000/0.png", "class_label"].item() == "Hemangioma"


def test_missing_root_raises(tmp_path, make_cfg):
    with pytest.raises(FileNotFoundError):
        build_index(make_cfg(tmp_path / "nope"))


def test_per_class_patient_scope_namespaces_ids(tmp_path, make_cfg):
    build_tree(tmp_path, layout=LAYOUT_A)
    f = build_index(make_cfg(tmp_path, patient_scope="per_class")).frame
    assert f["patient_key"].str.contains("::").all()
