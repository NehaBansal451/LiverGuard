"""M5: Grad-CAM, occlusion, region masking, CAM faithfulness - known-answer toy models first, then the engine end to end."""
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
from torch import nn

from liverguard.explain import masking
from liverguard.explain.engine import run_explainability
from liverguard.explain.gradcam import GradCAM, batched_gradcam, randomization_check, randomized_copy
from liverguard.explain.occlusion import occlusion_map

from tests.test_robustness_engine import world  # noqa: F401

REPO = Path(__file__).resolve().parents[1]


class Toy(nn.Module):
    """logit_HCC = + mean(conv(x)), logit_hem = - mean(conv(x)) with conv = sum of channels (all positive for positive images)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1, bias=False)
        nn.init.constant_(self.conv.weight, 1.0)
        self.fc = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[-1.0], [1.0]]))

    def forward(self, x):
        return self.fc(self.conv(x).mean((2, 3)))


def _blob_image(r=slice(0, 8), c=slice(0, 8), size=32):
    x = torch.full((1, 3, size, size), 0.1)
    x[:, :, r, c] = 1.0
    return x


# ------------------------------------------------------------------ Grad-CAM
def test_gradcam_peaks_on_the_evidence_and_is_zero_for_the_opposing_class():
    gc = GradCAM(Toy())
    cam, probs = gc(_blob_image(), 1)
    assert cam.shape == (1, 32, 32) and cam.min() >= 0 and cam.max() == pytest.approx(1.0)
    assert cam[0, :8, :8].mean() > 5 * cam[0, 16:, 16:].mean()            # mass sits on the bright blob
    cam0, _ = gc(_blob_image(), 0)
    assert cam0.max() == 0                                                  # class 0 gets only negative evidence -> ReLU -> empty
    assert probs.shape == (1, 2) and probs.sum() == pytest.approx(1.0)
    gc.close()


def test_gradcam_predicted_target_matches_argmax_and_batches_agree():
    m = Toy().eval()
    X = torch.cat([_blob_image(), _blob_image(slice(24, 32), slice(24, 32))])
    cams, probs = batched_gradcam(m, X, "predicted", batch_size=1)
    full, _ = batched_gradcam(m, X, "predicted", batch_size=8)
    assert np.allclose(cams, full, atol=1e-6) and (probs.argmax(1) == 1).all()
    assert cams[1, 24:, 24:].mean() > cams[1, :8, :8].mean()


def test_gradcam_hook_does_not_break_other_passes_regression():
    """Found while building M5: the forward hook stayed on the model and crashed no-grad passes (occlusion)."""
    m = Toy().eval()
    gc = GradCAM(m)
    with torch.inference_mode():
        m(_blob_image())                                                     # must not raise
    cam, _ = gc(_blob_image(), 1)
    assert cam.max() > 0
    gc.close()
    m(_blob_image())


def test_gradcam_leaves_model_parameters_untouched():
    m = Toy().eval()
    before = [p.clone() for p in m.parameters()]
    GradCAM(m)(_blob_image(), 1)
    assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))


def test_randomised_copy_changes_weights_not_the_original_and_check_returns_per_image_values():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1, self.c2 = nn.Conv2d(3, 4, 3, padding=1), nn.Conv2d(4, 4, 3, padding=1)
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(torch.relu(self.c2(torch.relu(self.c1(x)))).mean((2, 3)))
    torch.manual_seed(0)
    m = Net().eval()
    w = m.c1.weight.clone()
    r = randomized_copy(m, seed=1)
    assert torch.equal(m.c1.weight, w) and not torch.equal(r.c1.weight, w)
    X = torch.rand(5, 3, 16, 16)
    s = randomization_check(m, X, np.ones(5, dtype=int), seed=1)
    assert s.shape == (5,) and np.all((np.nan_to_num(s) >= -1) & (np.nan_to_num(s) <= 1))
    assert np.array_equal(np.nan_to_num(s, nan=9), np.nan_to_num(randomization_check(m, X, np.ones(5, dtype=int), seed=1), nan=9))


# ------------------------------------------------------------------ occlusion
class Region(nn.Module):
    """Probability depends only on the top-left 8x8 block of channel 0."""

    def forward(self, x):
        s = (x[:, 0, :8, :8].mean((1, 2)) - 0.5) * 20
        return torch.stack([-s, s], 1)


def test_occlusion_localises_the_only_pixels_the_model_uses():
    x = _blob_image()
    occ, base = occlusion_map(Region(), x, target=1, patch=8, stride=8)
    assert base > 0.99 and occ.shape == (32, 32)
    assert occ[:8, :8].min() > 0.4                                          # covering the block collapses the probability
    assert np.abs(occ[8:, :]).max() < 1e-6 and np.abs(occ[:, 8:]).max() < 1e-6   # everything else is irrelevant


def test_occlusion_is_deterministic_and_covers_edges_with_non_dividing_stride():
    x = _blob_image(slice(20, 28), slice(20, 28))
    a, _ = occlusion_map(Toy().eval(), x, 1, patch=10, stride=7)
    b, _ = occlusion_map(Toy().eval(), x, 1, patch=10, stride=7)
    assert np.array_equal(a, b) and np.isfinite(a).all()


# ------------------------------------------------------------------ masking
def test_proxy_roi_area_and_centre():
    m = masking.proxy_roi(64, 0.4)
    assert m.mean() == pytest.approx(0.4, abs=0.02) and m[32, 32] and not m[0, 0]


def test_random_control_keeps_exact_area_and_shape():
    rng = np.random.default_rng(0)
    roi = masking.proxy_roi(48, 0.3)
    for _ in range(5):
        c = masking.random_translate(roi, rng)
        assert c.sum() == roi.sum()


def test_cam_region_has_exact_area_and_selects_highest_pixels():
    cam = np.arange(100, dtype=float).reshape(10, 10)
    reg = masking.cam_region(cam, 0.2)
    assert reg.sum() == 20 and cam[reg].min() >= cam[~reg].max()
    assert masking.cam_region(np.zeros((10, 10)), 0.2).sum() == 20        # constant map: still exact area (deterministic ties)


def test_fill_outside_keeps_inside_pixels_exactly_and_fills_with_mean():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    keep = masking.proxy_roi(16, 0.4)
    out = np.asarray(masking.fill_outside(Image.fromarray(a), keep))
    assert np.array_equal(out[keep], a[keep])
    assert (out[~keep] == np.round(a.reshape(-1, 3).mean(0)).astype(np.uint8)).all()
    assert (np.asarray(masking.fill_outside(Image.fromarray(a), keep, "black"))[~keep] == 0).all()


def test_roi_and_background_variants_partition_the_image():
    img = Image.fromarray(np.random.default_rng(2).integers(0, 255, (20, 20, 3), dtype=np.uint8))
    roi = masking.proxy_roi(20, 0.4)
    v = masking.region_variants(img, roi, np.random.default_rng(0), n_control=2)
    assert set(v) == {"roi_only", "background_only", "control_roi_only", "control_background_only"}
    assert len(v["control_roi_only"]) == 2 and len(v["roi_only"]) == 1
    a = np.asarray(img)
    assert np.array_equal(np.asarray(v["roi_only"][0])[roi], a[roi]) and np.array_equal(np.asarray(v["background_only"][0])[~roi], a[~roi])


def test_annotation_masks_are_found_by_relative_path_not_bare_filename(tmp_path):
    (tmp_path / "val_set/Clean/HCC/H001").mkdir(parents=True)
    m = np.zeros((10, 10), np.uint8); m[2:6, 2:6] = 255
    Image.fromarray(m).save(tmp_path / "val_set/Clean/HCC/H001/0.png")
    got = masking.load_annotation_mask(tmp_path, "val_set/Clean/HCC/H001/0.png", 20)
    assert got.shape == (20, 20) and got.dtype == bool and 0 < got.sum() < 400
    # a DIFFERENT patient's image with the same file name must not pick up this mask
    assert masking.load_annotation_mask(tmp_path, "val_set/Clean/HCC/H002/0.png", 20) is None
    assert masking.load_annotation_mask(None, "x.png", 20) is None


def test_empty_or_full_annotation_masks_are_rejected(tmp_path):
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / "a.png")
    Image.fromarray(np.full((8, 8), 255, np.uint8)).save(tmp_path / "b.png")
    for name in ("a.png", "b.png"):
        with pytest.raises(ValueError, match="empty or covers"):
            masking.load_annotation_mask(tmp_path, name, 8)


# ------------------------------------------------------------------ engine
def test_engine_outputs_controls_and_limitations(world, tmp_path):
    r = run_explainability(world["ckpt"], world["index"], tmp_path / "ex", n_boot=10, sanity_n=6, n_cases=1, seed=0)
    m = r.manifest
    assert m["valid_as_result"] is False and "SMOKE" in m["watermark"]
    assert "NOT the liver" in m["roi_source"] and m["scope"].startswith("DIAGNOSTIC")
    assert {"grad_cam", "occlusion", "masking", "roi", "cam_faithfulness"} <= set(m["limitations"])
    v = set(r.masking_summary["variant"])
    assert {"full", "roi_only", "background_only", "control_roi_only", "control_background_only",
            "cam_delete", "cam_keep", "control_cam_delete", "control_cam_keep"} <= v
    full = r.masking_summary.set_index("variant").loc["full"]
    assert full["flip_rate_vs_full"] == 0                                    # 'full' is compared with itself
    assert r.masking_summary["auroc"].between(0, 1).all()
    for f in ("masking_summary", "cam_faithfulness", "cases", "masking_per_image", "gradcam_randomization_check", "fig_masking",
              "fig_cases", "manifest"):
        assert r.paths[f].exists(), f
    assert r.cases["group"].str.contains("random correct|confident error").all()
    assert (r.cam_faithfulness["variant"].str.startswith("paired")).any()
    assert json.loads(r.paths["manifest"].read_text())["dataset"]["n_images"] == len(
        pd.read_csv(r.paths["cam_roi_ratio"]))


def test_engine_full_probabilities_equal_the_reliability_engine_probabilities(world, tmp_path):
    from liverguard.reliability.engine import run_reliability_analysis
    e = run_explainability(world["ckpt"], world["index"], tmp_path / "ex2", n_boot=0, sanity_n=2, n_cases=1, make_plots=False)
    rel = run_reliability_analysis(world["ckpt"], world["index"], tmp_path / "rel2", n_boot=0, mc_samples=2, make_plots=False)
    per = pd.read_csv(e.paths["masking_per_image"])
    full = per[per.variant == "full"].set_index("image_id")["p"]
    assert np.allclose(full.reindex(rel.per_image["image_id"]).to_numpy(), rel.per_image["p"].to_numpy(), atol=1e-6)


def test_engine_is_reproducible(world, tmp_path):
    kw = dict(n_boot=0, sanity_n=4, n_cases=1, make_plots=False, seed=5)
    a = run_explainability(world["ckpt"], world["index"], tmp_path / "a", **kw)
    b = run_explainability(world["ckpt"], world["index"], tmp_path / "b", **kw)
    pd.testing.assert_frame_equal(a.masking_summary, b.masking_summary)
    pd.testing.assert_frame_equal(a.cam_faithfulness, b.cam_faithfulness)
    pd.testing.assert_frame_equal(pd.read_csv(a.paths["masking_per_image"]), pd.read_csv(b.paths["masking_per_image"]))


def test_engine_uses_annotation_masks_when_given_and_rejects_a_useless_mask_dir(world, tmp_path):
    idx = pd.read_csv(world["index"])
    val = idx[(idx["split"] == "val") & (idx["variant"].str.lower() == "clean")]
    md = tmp_path / "masks"
    for rel in val["image_id"].head(10):
        (md / rel).parent.mkdir(parents=True, exist_ok=True)
        m = np.zeros((32, 32), np.uint8); m[8:24, 8:24] = 255
        Image.fromarray(m).save(md / rel)
    r = run_explainability(world["ckpt"], world["index"], tmp_path / "ann", mask_dir=md, n_boot=0, sanity_n=2, n_cases=1, make_plots=False)
    assert r.manifest["roi_source"].startswith("annotation masks (10/")
    with pytest.raises(ValueError, match="No annotation masks"):
        run_explainability(world["ckpt"], world["index"], tmp_path / "bad", mask_dir=tmp_path, n_boot=0, make_plots=False)


def test_test_split_guard(world, tmp_path):
    with pytest.raises(ValueError, match="final_report"):
        run_explainability(world["ckpt"], world["index"], tmp_path / "t", split="test", make_plots=False)


def _cli(*args):
    return subprocess.run([sys.executable, "scripts/explain.py", *map(str, args)], cwd=REPO, capture_output=True, text=True)


def test_cli_exit_codes(world, tmp_path):
    base = ["--checkpoint", world["ckpt"], "--index-csv", world["index"]]
    ok = _cli(*base, "--out-dir", tmp_path / "cli", "--n-boot", 0, "--n-cases", 1, "--no-plots")
    assert ok.returncode == 0 and "NOT A RESULT" in ok.stdout and "centre_proxy" in ok.stdout
    assert _cli(*base, "--out-dir", tmp_path / "x", "--split", "test").returncode == 2
    assert _cli(*base, "--out-dir", tmp_path / "y", "--mask-dir", tmp_path / "nope").returncode == 2
