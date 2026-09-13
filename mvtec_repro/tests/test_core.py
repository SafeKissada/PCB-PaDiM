"""pytest mvtec_repro/tests -q  — runs offline, no torch, no MVTec download."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mvtec_repro import core

# Class-mean AUROC printed in each paper (percent). If a per-class value was
# mistyped, the recomputed mean drifts away from the paper's own average.
PAPER_MEANS = {
    ("patchcore", "image_auroc"): 99.0, ("patchcore", "pixel_auroc"): 98.0,
    ("padim", "image_auroc"): 95.5, ("padim", "pixel_auroc"): 97.5,
    ("rd4ad", "image_auroc"): 98.5, ("rd4ad", "pixel_auroc"): 97.8,
    ("simplenet", "image_auroc"): 99.6, ("simplenet", "pixel_auroc"): 98.1,
}


# ───────────────────────── reference tables ────────────────────────────────
@pytest.mark.parametrize("key,metric", sorted(PAPER_MEANS))
def test_reference_mean_matches_paper_average(key, metric):
    ref = core.REFERENCES[key]
    assert abs(ref.mean(metric) * 100 - PAPER_MEANS[(key, metric)]) <= 0.1


@pytest.mark.parametrize("key", sorted(core.REFERENCES))
def test_reference_covers_all_15_categories(key):
    ref = core.REFERENCES[key]
    assert set(ref.image_auroc) == set(core.MVTEC_CATEGORIES)
    assert set(ref.pixel_auroc) == set(core.MVTEC_CATEGORIES)


def _dummy_protocol():
    return core.REFERENCES["padim"].protocol


def test_reference_rejects_category_typo():
    with pytest.raises(core.ReferenceTableError, match="transitor"):
        core.PublishedReference("x", "c", "t", None, _dummy_protocol(),
                                image_auroc={"transitor": 0.97}, pixel_auroc={})


def test_reference_rejects_percent_values():
    with pytest.raises(core.ReferenceTableError, match="fractions"):
        core.PublishedReference("x", "c", "t", None, _dummy_protocol(),
                                image_auroc={"bottle": 99.9}, pixel_auroc={})


def test_protocol_differences_are_detectable():
    paper = core.REFERENCES["patchcore"].protocol
    ours = core.ProtocolSpec(**{**paper.__dict__, "preprocess": core.PreprocessSpec(224, None)})
    assert ours != paper and paper.preprocess.output_size == ours.preprocess.output_size == 224


def test_center_crop_offset_matches_torchvision_rounding():
    assert core.PreprocessSpec(256, 224).crop_offset() == 16
    assert core.PreprocessSpec(256, None).crop_offset() == 0
    with pytest.raises(ValueError):
        core.PreprocessSpec(224, 256)


# ───────────────────────── dataset scan ────────────────────────────────────
def _png(path: Path, size=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size, size, 3), np.uint8)).save(path)


@pytest.fixture
def fake_bottle(tmp_path):
    root = tmp_path / "mvtec"
    for i in range(3):
        _png(root / "bottle/train/good" / f"{i:03d}.png")
    _png(root / "bottle/test/good/000.png")
    for i in range(2):
        _png(root / "bottle/test/broken_large" / f"{i:03d}.png")
        _png(root / "bottle/ground_truth/broken_large" / f"{i:03d}_mask.png")
    return root


def test_scan_rejects_non_official_counts(fake_bottle):
    with pytest.raises(core.DatasetLayoutError, match="official is"):
        core.scan_category(fake_bottle, "bottle")


def test_scan_reads_official_layout(fake_bottle):
    train, test = core.scan_category(fake_bottle, "bottle", strict_counts=False)
    assert len(train) == 3 and all(s.label == 0 for s in train)
    assert [s.label for s in test] == [1, 1, 0]
    assert all(s.mask_path is not None and s.mask_path.is_file() for s in test if s.label == 1)


def test_scan_fails_on_missing_mask(fake_bottle):
    (fake_bottle / "bottle/ground_truth/broken_large/001_mask.png").unlink()
    with pytest.raises(core.DatasetLayoutError, match="mask"):
        core.scan_category(fake_bottle, "bottle", strict_counts=False)


def test_scan_rejects_unknown_category(fake_bottle):
    with pytest.raises(core.DatasetLayoutError, match="transitor"):
        core.scan_category(fake_bottle, "transitor")


def test_manifest_is_deterministic_and_content_sensitive(fake_bottle):
    tr, te = core.scan_category(fake_bottle, "bottle", strict_counts=False)
    h1 = core.manifest_sha256(tr + te, fake_bottle)
    assert h1 == core.manifest_sha256(list(reversed(tr + te)), fake_bottle)
    _png(fake_bottle / "bottle/train/good/000.png", size=16)   # file changed
    assert h1 != core.manifest_sha256(tr + te, fake_bottle)


# ───────────────────────── metrics & guards ────────────────────────────────
def test_image_metrics_perfect_and_invalid():
    m = core.image_level_metrics([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1])
    assert m["image_auroc"] == 1.0 and m["image_ap"] == 1.0
    with pytest.raises(core.InvalidScoreError, match="NaN"):
        core.image_level_metrics([0.1, np.nan], [0, 1])
    with pytest.raises(core.InvalidScoreError, match="both classes"):
        core.image_level_metrics([0.1, 0.2], [1, 1])


def _toy_pixel_problem(seed=0, n_good=6, n_bad=6, s=16):
    """Normal images: low noise. Anomalous images: same noise + a hot square."""
    rng = np.random.default_rng(seed)
    maps = rng.normal(1.0, 0.1, (n_good + n_bad, s, s))
    masks = np.zeros_like(maps, dtype=np.uint8)
    masks[n_good:, 4:8, 4:8] = 1
    maps[n_good:][:, 4:8, 4:8] += 2.0
    return maps, masks


def test_pixel_auroc_and_geometry_guard():
    maps, masks = _toy_pixel_problem()
    assert core.pixel_level_auroc(maps, masks) > 0.99
    with pytest.raises(core.InvalidScoreError, match="geometry"):
        core.pixel_level_auroc(maps[:, :8], masks)


def test_per_image_minmax_is_detected_and_actually_hurts_pauroc():
    maps, masks = _toy_pixel_problem()
    core.assert_raw_pixel_maps(maps)                       # raw -> OK
    mn = maps.min(axis=(1, 2), keepdims=True)
    mx = maps.max(axis=(1, 2), keepdims=True)
    normed = (maps - mn) / (mx - mn)                       # what padim.py returns today
    with pytest.raises(core.PerImageNormalizedMapError):
        core.assert_raw_pixel_maps(normed)
    # Normal images get stretched to reach 1.0 -> false positives -> P-AUROC is
    # biased downward (magnitude is data-dependent; the point is it is no longer
    # the paper's metric).
    assert core.pixel_level_auroc(normed, masks) < core.pixel_level_auroc(maps, masks) - 0.005


# ───────────────────────── comparison ──────────────────────────────────────
def test_compare_statuses():
    ref = {"bottle": 1.0, "wood": 0.992}
    per_seed = {"bottle": [0.999, 1.0, 0.998], "wood": [0.95, 0.96, 0.955],
                "transistor": [0.99, 0.98, 0.985], "grid": [0.97]}
    v = {x.category: x for x in core.compare_to_reference(
        per_seed, {**ref, "grid": 0.98}, metric="image_auroc", tolerance=0.01)}
    assert v["bottle"].status is core.VerdictStatus.PASS
    assert v["wood"].status is core.VerdictStatus.FAIL and v["wood"].delta < 0
    assert v["transistor"].status is core.VerdictStatus.NO_REFERENCE
    assert v["grid"].status is core.VerdictStatus.INSUFFICIENT_SEEDS
    assert v["wood"].ci95_low < v["wood"].mean < v["wood"].ci95_high


def test_single_seed_has_no_ci_and_serializes_to_null():
    (v,) = core.compare_to_reference({"bottle": [1.0]}, {"bottle": 1.0},
                                     metric="image_auroc", tolerance=0.01, min_seeds=1)
    d = v.to_dict()
    assert d["ci95_low"] is None and d["status"] == "PASS"


# ───────────────────────── provenance ──────────────────────────────────────
def test_write_run_record_is_valid_json_without_nan(tmp_path):
    p = core.write_run_record(tmp_path / "r/run.json",
                              {"x": float("nan"), "y": np.float32(0.5),
                               "s": core.VerdictStatus.PASS, "p": Path("a/b")})
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data == {"x": None, "y": 0.5, "s": "PASS", "p": "a/b"}
    assert not (tmp_path / "r/run.json.tmp").exists()


def test_environment_info_never_raises_even_with_broken_torch():
    # Exercises whatever torch install state this sandbox actually has
    # (including a broken/partial one) — must never raise.
    info = core.environment_info()
    assert info["python"] and info["numpy"] and info["sklearn"]
    assert "torch" in info and "gpu" in info


def test_fingerprint_and_git_provenance_never_raise(tmp_path):
    fp = core.package_fingerprint()
    assert len(fp) == 64 and fp == core.package_fingerprint()
    prov = core.git_provenance(tmp_path)            # not a git repo
    assert prov["commit"] is None and "error" in prov


# ───────────────────────── adapter wiring sanity ───────────────────────────
def test_simplenet_adaptor_bias_polarity_matches_paper():
    """Regression test for a real bug caught while writing
    adapter_simplenet.py: `ADAPTOR_BIAS=not spec.hp("adaptor_bias", False)`
    silently inverts the intended value. The paper's adaptor has NO bias,
    so REFERENCES["simplenet"].protocol stores adaptor_bias=False, and the
    adapter must assign that value directly (no `not`)."""
    spec = core.REFERENCES["simplenet"].protocol
    assert spec.hp("adaptor_bias", None) is False
    correct_cfg_value = spec.hp("adaptor_bias", False)   # what the adapter should assign
    assert correct_cfg_value is False
    buggy_cfg_value = not spec.hp("adaptor_bias", False)  # the bug that was caught
    assert buggy_cfg_value is True and buggy_cfg_value != correct_cfg_value


# ───────────────────────── ICT timestamps ──────────────────────────────────
def test_ict_is_utc_plus_7():
    from datetime import datetime, timezone
    utc_dt = datetime(2026, 9, 11, 9, 0, 0, tzinfo=timezone.utc)
    s = core.ict_isoformat(utc_dt)
    assert s == "2026-09-11 16:00:00+0700"


def test_now_ict_has_thailand_offset():
    dt = core.now_ict()
    assert dt.utcoffset().total_seconds() == 7 * 3600


# ───────────────────────── build_run_record (4-section schema) ────────────
def _toy_categories():
    return [
        {"category": "bottle", "n_train": 209, "n_test": 83,
         "per_seed": [{"seed": 0, "image_auroc": 0.998, "image_ap": 0.997, "pixel_auroc": 0.983},
                      {"seed": 1, "image_auroc": 1.000, "image_ap": 1.000, "pixel_auroc": 0.985},
                      {"seed": 2, "image_auroc": 0.999, "image_ap": 0.998, "pixel_auroc": 0.984}]},
        {"category": "screw", "n_train": 320, "n_test": 160,
         "per_seed": [{"seed": 0, "image_auroc": 0.941, "image_ap": 0.912, "pixel_auroc": 0.987},
                      {"seed": 1, "image_auroc": 0.958, "image_ap": 0.930, "pixel_auroc": 0.989},
                      {"seed": 2, "image_auroc": 0.935, "image_ap": 0.905, "pixel_auroc": 0.986}]},
    ]


def test_build_run_record_key_order_matches_user_spec():
    from datetime import datetime, timezone
    t0 = datetime(2026, 9, 11, 9, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 11, 9, 47, 12, tzinfo=timezone.utc)
    record = core.build_run_record(core.REFERENCES["patchcore"], core.REFERENCES["patchcore"].protocol,
                                   _toy_categories(), t0, t1)
    # Section order: run_info -> protocol -> results -> summary (+ pass_rate)
    assert list(record.keys())[:4] == ["run_info", "protocol", "results", "summary"]
    # Section 1 contains ONLY started/finished (in ICT) + environment.
    assert set(record["run_info"]) == {"started_ict", "finished_ict", "environment"}
    assert record["run_info"]["started_ict"] == "2026-09-11 16:00:00+0700"
    # Section 3: paper value appears before per_seed within each category dict.
    bottle = record["results"]["categories"][0]
    assert list(bottle.keys()).index("paper_image_auroc") < list(bottle.keys()).index("per_seed")
    assert bottle["paper_image_auroc"] == core.REFERENCES["patchcore"].image_auroc["bottle"]


def test_build_run_record_summary_matches_mean_and_diff():
    record = core.build_run_record(core.REFERENCES["patchcore"], core.REFERENCES["patchcore"].protocol,
                                   _toy_categories(), core.now_ict(), core.now_ict())
    by_cat_metric = {(s["category"], s["metric"]): s for s in record["summary"]}
    screw = by_cat_metric[("screw", "image_auroc")]
    assert screw["mean"] == pytest.approx((0.941 + 0.958 + 0.935) / 3, abs=1e-4)
    assert screw["paper"] == core.REFERENCES["patchcore"].image_auroc["screw"]
    assert screw["diff_vs_paper"] == pytest.approx(screw["mean"] - screw["paper"], abs=1e-4)
    assert screw["status"] == "FAIL"  # diff ~-0.019, outside default 0.01 tolerance
    assert by_cat_metric[("bottle", "image_auroc")]["status"] == "PASS"


def test_build_run_record_pass_rate_is_summarized():
    record = core.build_run_record(core.REFERENCES["patchcore"], core.REFERENCES["patchcore"].protocol,
                                   _toy_categories(), core.now_ict(), core.now_ict())
    pr = record["pass_rate"]
    assert pr["checkable"] == len(record["summary"])
    assert pr["passed"] == sum(1 for s in record["summary"] if s["status"] == "PASS")
    assert pr["percent"] == pytest.approx(100 * pr["passed"] / pr["checkable"], abs=0.1)


def test_build_run_record_is_json_roundtrippable(tmp_path):
    record = core.build_run_record(core.REFERENCES["patchcore"], core.REFERENCES["patchcore"].protocol,
                                   _toy_categories(), core.now_ict(), core.now_ict())
    p = core.write_run_record(tmp_path / "run.json", record)
    reloaded = json.loads(p.read_text(encoding="utf-8"))
    # Key order must survive the round trip (sort_keys=False).
    assert list(reloaded.keys())[:4] == ["run_info", "protocol", "results", "summary"]
