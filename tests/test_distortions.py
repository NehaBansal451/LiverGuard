"""Each distortion family is tested independently against an analytic property, plus contract tests for all conditions."""
import numpy as np
import pytest
from PIL import Image

from liverguard.robustness.config import load_robustness_config
from liverguard.robustness.distortions import DISTORTION_FUNCS, apply_distortion, rng_for
from liverguard.robustness.fidelity import at_risk, fidelity_metrics

CFG = load_robustness_config()
COND = {c.name: c for c in CFG.conditions}
SEED, KEY = 0, "img-001"


def textured(size: int = 64, lo: int = 0, hi: int = 255, seed: int = 3) -> Image.Image:
    """Greyscale test image with structure (gradient + blobs + fine texture), values within [lo, hi]."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size] / size
    a = 0.4 * xx + 0.3 * yy + 0.5 * np.exp(-(((xx - .5) ** 2 + (yy - .4) ** 2) / 0.02)) + 0.08 * rng.random((size, size))
    a = (a - a.min()) / (a.max() - a.min())
    g = np.rint(lo + a * (hi - lo)).astype(np.uint8)
    return Image.fromarray(np.stack([g] * 3, -1))


def arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float64)


def run(name: str, level: int, img: Image.Image, key: str = KEY) -> Image.Image:
    c = COND[name]
    return apply_distortion(img, c.func, c.params[level - 1], seed=SEED, image_id=key, condition=name)


def mse(a: Image.Image, b: Image.Image) -> float:
    return float(np.mean((arr(a) - arr(b)) ** 2))


# ------------------------------------------------------------------ contract: every condition x level
ALL = [(c.name, lv) for c in CFG.conditions for lv in range(1, len(c.params) + 1)]


@pytest.mark.parametrize("name,level", ALL)
def test_contract_size_mode_range_greyscale_original_untouched_deterministic(name, level):
    img = textured()
    before = img.tobytes()
    out = run(name, level, img)
    assert img.tobytes() == before                                   # original preserved
    assert out.size == img.size and out.mode == "RGB"
    a = arr(out)
    assert a.min() >= 0 and a.max() <= 255
    assert np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2])   # still greyscale
    assert out.tobytes() == run(name, level, img).tobytes()           # same key -> same output


@pytest.mark.parametrize("name", [c.name for c in CFG.conditions])
def test_error_grows_with_severity(name):
    img = textured(96)
    errs = [mse(img, run(name, lv, img)) for lv in range(1, len(COND[name].params) + 1)]
    assert all(b >= a - 1e-9 for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] > errs[0] and errs[0] > 0                         # every level actually changes the image


def test_every_distortion_function_is_used_and_config_matches_spec():
    assert {c.func for c in CFG.conditions} == set(DISTORTION_FUNCS)
    assert {c.family for c in CFG.conditions} == {"brightness", "contrast", "gaussian_noise", "speckle_noise",
                                                  "blur", "rotation", "crop", "gamma", "jpeg"}
    assert all(len(c.params) == 5 for c in CFG.conditions)             # multiple severity levels for each
    with pytest.raises(ValueError):
        apply_distortion(textured(), "sharpen", 1.0, seed=0, image_id="x", condition="x")


# ------------------------------------------------------------------ brightness
def test_brightness_scales_intensity_exactly_when_unclipped():
    img = textured(lo=40, hi=110)
    for name in ("brightness_dark", "brightness_bright"):
        for lv in range(1, 6):
            f = COND[name].params[lv - 1]
            expected = np.clip(np.rint(arr(img) * f), 0, 255)
            assert np.array_equal(arr(run(name, lv, img)), expected), (name, lv)
    assert arr(run("brightness_dark", 5, img)).mean() < arr(img).mean() < arr(run("brightness_bright", 5, img)).mean()


def test_brightness_clips_at_white():
    out = arr(run("brightness_bright", 5, textured(lo=150, hi=250)))
    assert out.max() == 255 and (out == 255).mean() > 0.3


# ------------------------------------------------------------------ contrast
def test_contrast_preserves_mean_and_scales_spread():
    img = textured(lo=60, hi=100)
    m0, s0 = arr(img).mean(), arr(img).std()
    for name in ("contrast_low", "contrast_high"):
        for lv in range(1, 6):
            out = arr(run(name, lv, img))
            f = COND[name].params[lv - 1]
            assert abs(out.mean() - m0) < 0.6
            assert out.std() == pytest.approx(f * s0, rel=0.03, abs=0.5)


# ------------------------------------------------------------------ gaussian noise
def test_gaussian_noise_has_requested_std_zero_mean_and_shared_channels():
    flat = Image.new("RGB", (160, 160), (128, 128, 128))
    for lv in range(1, 6):
        d = arr(run("gaussian_noise", lv, flat)) - 128
        assert d[..., 0].std() == pytest.approx(COND["gaussian_noise"].params[lv - 1], rel=0.07)
        assert abs(d.mean()) < 0.7
        assert np.array_equal(d[..., 0], d[..., 2])


def test_gaussian_noise_is_seeded_per_image_and_nested_across_levels():
    flat = Image.new("RGB", (64, 64), (128, 128, 128))
    a, b = run("gaussian_noise", 3, flat, "A"), run("gaussian_noise", 3, flat, "B")
    assert a.tobytes() != b.tobytes()                                   # different images -> different noise
    assert apply_distortion(flat, "gaussian_noise", 12, seed=1, image_id="A", condition="gaussian_noise").tobytes() != a.tobytes()
    n2, n4 = arr(run("gaussian_noise", 2, flat)) - 128, arr(run("gaussian_noise", 4, flat)) - 128
    assert np.corrcoef(n2.ravel(), n4.ravel())[0, 1] > 0.99             # same noise field, scaled (common random numbers)


# ------------------------------------------------------------------ speckle noise
def test_speckle_is_multiplicative():
    flat = Image.new("RGB", (160, 160), (100, 100, 100))
    for lv in range(1, 5):                                              # level 5 loses std to clipping at 0
        d = arr(run("speckle_noise", lv, flat))
        assert d[..., 0].std() == pytest.approx(100 * COND["speckle_noise"].params[lv - 1], rel=0.08)
    assert np.all(arr(run("speckle_noise", 5, Image.new("RGB", (32, 32), (0, 0, 0)))) == 0)   # black stays black
    two = np.zeros((128, 128, 3), np.uint8); two[:, :64] = 50; two[:, 64:] = 150
    d = arr(run("speckle_noise", 3, Image.fromarray(two)))
    assert d[:, 64:].std() > 2.0 * d[:, :64].std()                      # noise std scales with intensity (150 vs 50)


# ------------------------------------------------------------------ blur
def test_blur_removes_high_frequency_but_keeps_mean():
    img = textured(96)
    lap = lambda im: np.var(np.diff(arr(im)[..., 0], 2, axis=1))    # noqa: E731 - 2nd-difference energy
    energy = [lap(run("gaussian_blur", lv, img)) for lv in range(1, 6)]
    assert lap(img) > energy[0] and all(b < a for a, b in zip(energy, energy[1:]))
    assert abs(arr(run("gaussian_blur", 5, img)).mean() - arr(img).mean()) < 2.0
    flat = Image.new("RGB", (40, 40), (90, 90, 90))
    assert np.array_equal(arr(run("gaussian_blur", 5, flat)), arr(flat))   # flat image unchanged


# ------------------------------------------------------------------ rotation
def test_rotation_equals_pil_rotate_by_the_configured_angle_with_black_corners():
    img = textured(64, lo=60, hi=200)
    for lv in range(1, 6):
        deg = COND["rotation"].params[lv - 1]
        out = run("rotation", lv, img)
        cands = [img.rotate(s * deg, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0)) for s in (1, -1)]
        assert any(out.tobytes() == c.tobytes() for c in cands), lv
    assert arr(run("rotation", 5, img))[0, 0].max() == 0              # corner is fill
    keys = {run("rotation", 3, img, f"id{i}").tobytes() for i in range(12)}
    assert len(keys) == 2                                              # both directions occur across images, nothing else
    same_sign = {run("rotation", lv, img, "fixed").tobytes() == img.rotate(s * COND["rotation"].params[lv - 1],
                resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0)).tobytes() for lv in (2, 4) for s in (1,)}
    assert same_sign in ({True}, {False}, {True, False})               # sign is fixed per image across levels (checked next)
    signs = []
    for lv in (1, 3, 5):
        deg = COND["rotation"].params[lv - 1]
        pos = img.rotate(deg, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0)).tobytes()
        signs.append(run("rotation", lv, img, "fixed").tobytes() == pos)
    assert len(set(signs)) == 1


# ------------------------------------------------------------------ crop
def test_crop_equals_manual_center_crop_and_resize():
    img = textured(80)
    for lv in range(1, 6):
        keep = COND["center_crop"].params[lv - 1]
        n = round(80 * keep); o = (80 - n) // 2
        manual = img.crop((o, o, o + n, o + n)).resize((80, 80), Image.Resampling.BICUBIC)
        assert run("center_crop", lv, img).tobytes() == manual.tobytes()
    assert apply_distortion(img, "crop", 1.0, seed=0, image_id="x", condition="c").tobytes() == img.tobytes()


def test_crop_magnifies_the_centre():
    a = np.zeros((64, 64, 3), np.uint8); a[28:36, 28:36] = 255          # small central square
    out = arr(run("center_crop", 5, Image.fromarray(a)))
    assert (out[..., 0] > 128).sum() > (a[..., 0] > 128).sum() * 3.5    # area ~ 1/0.5^2 = 4x


# ------------------------------------------------------------------ gamma
def test_gamma_follows_formula_and_fixes_endpoints():
    img = textured()
    for name in ("gamma_low", "gamma_high"):
        for lv in range(1, 6):
            g = COND[name].params[lv - 1]
            expected = np.clip(np.rint(255 * (arr(img) / 255) ** g), 0, 255)
            assert np.array_equal(arr(run(name, lv, img)), expected)
    ramp = Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (4, 1))).convert("RGB")
    for name in ("gamma_low", "gamma_high"):
        r = arr(run(name, 4, ramp))[0, :, 0]
        assert r[0] == 0 and r[255] == 255 and np.all(np.diff(r) >= 0)   # monotone, endpoints fixed
    assert arr(run("gamma_low", 4, img)).mean() > arr(img).mean() > arr(run("gamma_high", 4, img)).mean()
    assert apply_distortion(img, "gamma", 1.0, seed=0, image_id="x", condition="g").tobytes() == img.tobytes()


# ------------------------------------------------------------------ jpeg
def test_jpeg_quality_orders_fidelity_and_file_size():
    import io
    img = textured(128)
    psnr = [fidelity_metrics(img, run("jpeg", lv, img))["psnr_db"] for lv in range(1, 6)]   # quality 90 -> 10
    assert all(a > b for a, b in zip(psnr, psnr[1:]))
    sizes = []
    for q in (90, 50, 10):
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=q); sizes.append(buf.tell())
    assert sizes[0] > sizes[1] > sizes[2]
    assert fidelity_metrics(img, apply_distortion(img, "jpeg", 100, seed=0, image_id="x", condition="j"))["psnr_db"] > 35


# ------------------------------------------------------------------ fidelity / label-preservation proxy
def test_fidelity_flags_destroyed_information_but_not_mild_changes():
    img = textured(96)
    same = fidelity_metrics(img, img)
    assert same["psnr_db"] == 99.0 and same["correlation"] == 1.0 and same["std_ratio"] == pytest.approx(1.0)
    assert not at_risk(same, CFG.fidelity)
    flat = Image.new("RGB", (96, 96), (10, 10, 10))                    # contrast collapse
    assert at_risk(fidelity_metrics(img, flat), CFG.fidelity)
    assert not at_risk(fidelity_metrics(img, run("brightness_dark", 1, img)), CFG.fidelity)
    assert at_risk(fidelity_metrics(textured(96, 150, 250), run("brightness_bright", 5, textured(96, 150, 250))), CFG.fidelity)


def test_rng_for_is_stable_and_key_sensitive():
    assert rng_for(0, "a", "b").random() == rng_for(0, "a", "b").random()
    assert rng_for(0, "a", "b").random() != rng_for(1, "a", "b").random()
    assert rng_for(0, "a", "b").random() != rng_for(0, "a", "c").random()


def test_geometric_distortions_are_not_flagged_by_unaligned_pixel_metrics():
    """Regression (found on the first audit run): a 5% zoom / 5 degree rotation was flagged as 'information destroyed'
    because PSNR and correlation compare unaligned pixels."""
    noisy = Image.fromarray(np.repeat(np.random.default_rng(0).integers(0, 256, (96, 96, 1), dtype=np.uint8), 3, 2))
    for name in ("center_crop", "rotation"):
        m = fidelity_metrics(noisy, run(name, 1, noisy))
        assert at_risk(m, CFG.fidelity)                                   # the naive proxy raises a false alarm ...
        assert not at_risk(m, CFG.fidelity, geometric=True)               # ... which the geometric mode avoids
    flat = Image.new("RGB", (96, 96), (10, 10, 10))
    assert at_risk(fidelity_metrics(noisy, flat), CFG.fidelity, geometric=True)   # contrast collapse is still caught
