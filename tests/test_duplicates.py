import shutil

import numpy as np
from PIL import Image

from liverguard.data.duplicates import compute_hashes, dhash, duplicate_report, near_duplicate_pairs, popcount
from liverguard.data.indexer import build_index
from tests.synthetic import build_tree


def _report(cfg, max_hamming=2):
    frame = build_index(cfg).frame
    return duplicate_report(frame, compute_hashes(frame), max_hamming)


def test_popcount():
    x = np.array([0, 1, 3, 2**63, 2**64 - 1], dtype=np.uint64)
    assert popcount(x).tolist() == [0, 1, 2, 1, 64]
    pairs, trunc = near_duplicate_pairs(np.array([0, 1, 2**40 + 3], dtype=np.uint64), max_hamming=1)
    assert pairs == [(0, 1, 1)] and not trunc


def test_dhash_is_16_hex_and_stable(tmp_path):
    build_tree(tmp_path, n_per_class=12)
    p = next(tmp_path.rglob("*.png"))
    assert dhash(p) == dhash(p) and len(dhash(p)) == 16


def test_unique_images_report_no_duplicates(tmp_path, make_cfg):
    build_tree(tmp_path)
    rep = _report(make_cfg(tmp_path))
    assert rep.ok and "no_duplicates" in rep.codes("info")


def test_exact_duplicate_across_splits_is_an_error(tmp_path, make_cfg):
    build_tree(tmp_path)
    src = tmp_path / "train_set/Clean/HCC/H000/0.png"
    dst = tmp_path / "test_set/Clean/HCC/H009/dup.png"
    shutil.copy(src, dst)
    rep = _report(make_cfg(tmp_path))
    assert not rep.ok
    assert rep.get("duplicate_cross_split").count == 1
    assert rep.get("duplicate_cross_patient").count == 1


def test_duplicate_with_conflicting_labels_is_an_error(tmp_path, make_cfg):
    build_tree(tmp_path)
    dst = tmp_path / "train_set/Clean/Hemangioma/M001/dup.png"
    shutil.copy(tmp_path / "train_set/Clean/HCC/H001/0.png", dst)
    rep = _report(make_cfg(tmp_path))
    assert "duplicate_label_conflict" in rep.codes("error")


def test_near_duplicate_across_splits_is_a_warning(tmp_path, make_cfg):
    build_tree(tmp_path)
    img = np.array(Image.open(tmp_path / "train_set/Clean/HCC/H000/0.png"))
    img[:2, :2] = np.clip(img[:2, :2].astype(int) + 7, 0, 255).astype(np.uint8)   # tiny edit, new bytes
    dst = tmp_path / "test_set/Clean/HCC/H009/near.png"
    Image.fromarray(img).save(dst)
    rep = _report(make_cfg(tmp_path))
    assert rep.ok                                              # near duplicates are warnings, not errors
    assert rep.get("near_duplicate_cross_split").count >= 1
