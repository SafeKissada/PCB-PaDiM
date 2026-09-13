"""
mvtec_repro.core — torch-free core of the unified MVTec AD reproduction harness
shared by PCB-Anomaly-{PatchCore,PaDiM,RD4AD,SimpleNet}.

Copy the whole ``mvtec_repro/`` package VERBATIM into every baseline repo (same
policy as src/data/dataset.py / src/evaluate.py). Every run record embeds
``package_fingerprint()`` so a reader can verify that all four baselines were
validated by byte-identical harness code.

Design rules
------------
* Pure functions only (no torch, no hidden global state) -> fully unit-testable.
* Fail hard, never degrade silently: wrong category name, missing mask, wrong
  split size, NaN score, per-image-normalised heatmap -> typed exception.
* Reference numbers carry their citation, table id and source quality, and are
  validated at import time (catches typos such as "transitor").
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger("mvtec_repro")

# Thailand time (Indochina Time, UTC+7, no daylight saving) — the user
# runs on Colab (which defaults to UTC), but wants timestamps readable in
# local Thai time rather than UTC.
ICT = timezone(timedelta(hours=7))


def now_ict() -> datetime:
    """Current time in Thailand (ICT, UTC+7)."""
    return datetime.now(ICT)


def ict_isoformat(dt: Optional[datetime] = None) -> str:
    """ISO-8601 string in ICT, e.g. '2026-09-11 16:47:12+07:00'."""
    dt = (dt or now_ict()).astimezone(ICT)
    return dt.strftime("%Y-%m-%d %H:%M:%S%z")

# (n_train_good, n_test_total) per category — official MVTec AD release.
# Totals 3629 / 1725, matching the counts quoted by the RD4AD/PatchCore papers.
OFFICIAL_SPLIT_SIZES: Mapping[str, Tuple[int, int]] = MappingProxyType({
    "bottle": (209, 83), "cable": (224, 150), "capsule": (219, 132),
    "carpet": (280, 117), "grid": (264, 78), "hazelnut": (391, 110),
    "leather": (245, 124), "metal_nut": (220, 115), "pill": (267, 167),
    "screw": (320, 160), "tile": (230, 117), "toothbrush": (60, 42),
    "transistor": (213, 100), "wood": (247, 79), "zipper": (240, 151),
})
MVTEC_CATEGORIES: Tuple[str, ...] = tuple(sorted(OFFICIAL_SPLIT_SIZES))
VALID_EXT = frozenset({".png", ".jpg", ".jpeg", ".bmp"})


# ───────────────────────────── exceptions ──────────────────────────────────
class ReproError(Exception):
    """Base class for every error raised by the reproduction harness."""


class DatasetLayoutError(ReproError):
    """MVTec folder layout/size does not match the official release."""


class ReferenceTableError(ReproError):
    """A published-reference table is malformed (bad key, value out of range)."""


class InvalidScoreError(ReproError):
    """Model output unusable for metrics (NaN/inf, shape mismatch, one class)."""


class PerImageNormalizedMapError(InvalidScoreError):
    """Pixel maps were min-max normalised per image -> P-AUROC is meaningless."""


# ───────────────────────────── protocol spec ───────────────────────────────
@dataclass(frozen=True)
class PreprocessSpec:
    """Square resize, then optional center crop. Masks use the same geometry
    with nearest-neighbour interpolation."""
    resize: int
    center_crop: Optional[int] = None

    def __post_init__(self) -> None:
        if self.resize <= 0:
            raise ValueError(f"resize must be > 0, got {self.resize}")
        if self.center_crop is not None and not 0 < self.center_crop <= self.resize:
            raise ValueError(f"center_crop must be in (0, resize], got {self.center_crop}")

    @property
    def output_size(self) -> int:
        return self.center_crop if self.center_crop is not None else self.resize

    def crop_offset(self) -> int:
        """Top/left offset of the crop — same rounding as torchvision.center_crop."""
        if self.center_crop is None:
            return 0
        return int(round((self.resize - self.center_crop) / 2.0))


@dataclass(frozen=True)
class ProtocolSpec:
    """Everything that must be identical between 'paper' and 'our run' for a
    numeric comparison to mean anything."""
    method: str
    preprocess: PreprocessSpec
    backbone: str
    backbone_weights: str                 # torchvision enum name, e.g. IMAGENET1K_V1
    feature_layers: Tuple[str, ...]
    heatmap_sigma: float = 4.0
    epochs: Optional[int] = None          # None = training-free method
    batch_size: int = 32
    hyperparams: Tuple[Tuple[str, Any], ...] = ()

    def hp(self, key: str, default: Any = None) -> Any:
        return dict(self.hyperparams).get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hyperparams"] = dict(self.hyperparams)
        return d


class SourceQuality(str, Enum):
    PRIMARY = "primary"       # transcribed from the method's own paper
    SECONDARY = "secondary"   # transcribed from a later paper's comparison table


@dataclass(frozen=True)
class PublishedReference:
    """Per-category published AUROC in [0, 1] with full citation."""
    method: str
    citation: str
    image_table: str
    pixel_table: Optional[str]
    protocol: ProtocolSpec
    image_auroc: Mapping[str, float]
    pixel_auroc: Mapping[str, float]
    image_quality: SourceQuality = SourceQuality.PRIMARY
    pixel_quality: SourceQuality = SourceQuality.PRIMARY
    notes: str = ""

    def __post_init__(self) -> None:
        for name, table in (("image_auroc", self.image_auroc), ("pixel_auroc", self.pixel_auroc)):
            unknown = set(table) - set(MVTEC_CATEGORIES)
            if unknown:
                raise ReferenceTableError(
                    f"{self.method}.{name}: unknown MVTec categories {sorted(unknown)} "
                    f"(typo? valid: {MVTEC_CATEGORIES})")
            bad = {k: v for k, v in table.items() if not (0.0 <= float(v) <= 1.0)}
            if bad:
                raise ReferenceTableError(
                    f"{self.method}.{name}: values must be fractions in [0,1] "
                    f"(did you paste percentages?): {bad}")

    def mean(self, metric: str, categories: Optional[Sequence[str]] = None) -> float:
        table = self.image_auroc if metric == "image_auroc" else self.pixel_auroc
        cats = list(categories) if categories is not None else list(table)
        missing = [c for c in cats if c not in table]
        if missing:
            raise ReferenceTableError(f"{self.method}.{metric}: no reference for {missing}")
        return float(np.mean([table[c] for c in cats]))


def _pct(d: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType({k: round(v / 100.0, 4) for k, v in d.items()})


REFERENCES: Mapping[str, PublishedReference] = MappingProxyType({
    "patchcore": PublishedReference(
        method="PatchCore-1%",
        citation="Roth et al., Towards Total Recall in Industrial Anomaly Detection, "
                 "CVPR 2022 (arXiv:2106.08265v2)",
        image_table="Suppl. Table S1 (PatchCore-1)", pixel_table="Suppl. Table S2 (PatchCore-1)",
        protocol=ProtocolSpec(
            method="patchcore", preprocess=PreprocessSpec(256, 224),
            backbone="wide_resnet50_2", backbone_weights="IMAGENET1K_V1",
            feature_layers=("layer2", "layer3"),
            hyperparams=(("coreset_ratio", 0.01), ("patch_size", 3),
                         ("target_embed_dim", 1024), ("projection_dim", 128))),
        image_auroc=_pct({"bottle": 100, "cable": 99.3, "capsule": 98.0, "carpet": 98.0,
                          "grid": 98.6, "hazelnut": 100, "leather": 100, "metal_nut": 99.7,
                          "pill": 97.0, "screw": 96.4, "tile": 99.4, "toothbrush": 100,
                          "transistor": 99.9, "wood": 99.2, "zipper": 99.2}),
        pixel_auroc=_pct({"bottle": 98.5, "cable": 98.2, "capsule": 98.8, "carpet": 98.9,
                          "grid": 98.6, "hazelnut": 98.6, "leather": 99.3, "metal_nut": 98.4,
                          "pill": 97.1, "screw": 99.2, "tile": 96.1, "toothbrush": 98.5,
                          "transistor": 94.9, "wood": 95.1, "zipper": 98.8}),
    ),
    "padim": PublishedReference(
        method="PaDiM-WR50-Rd550",
        citation="Defard et al., PaDiM, ICPR Workshops 2021 (arXiv:2011.08785)",
        image_table="per-class values as re-tabulated in Deng & Li CVPR 2022 Table 1 "
                    "('PaDiM (WResNet50)')",
        pixel_table="PaDiM Table 2 (identical in RD4AD Table 2 and SimpleNet Table 1)",
        protocol=ProtocolSpec(
            method="padim", preprocess=PreprocessSpec(256, 224),
            backbone="wide_resnet50_2", backbone_weights="IMAGENET1K_V1",
            feature_layers=("layer1", "layer2", "layer3"),
            hyperparams=(("num_selected_channels", 550), ("cov_eps", 0.01))),
        image_auroc=_pct({"bottle": 99.9, "cable": 92.7, "capsule": 91.3, "carpet": 99.8,
                          "grid": 96.7, "hazelnut": 92.0, "leather": 100, "metal_nut": 98.7,
                          "pill": 93.3, "screw": 85.8, "tile": 98.1, "toothbrush": 96.1,
                          "transistor": 97.4, "wood": 99.2, "zipper": 90.3}),
        pixel_auroc=_pct({"bottle": 98.3, "cable": 96.7, "capsule": 98.5, "carpet": 99.1,
                          "grid": 97.3, "hazelnut": 98.2, "leather": 99.2, "metal_nut": 97.2,
                          "pill": 95.7, "screw": 98.5, "tile": 94.1, "toothbrush": 98.8,
                          "transistor": 97.5, "wood": 94.9, "zipper": 98.5}),
        image_quality=SourceQuality.SECONDARY,
        notes="Per-class image AUROC differs between secondary sources (SimpleNet Table 1 "
              "lists a different PaDiM variant); treat per-class image checks as advisory "
              "and gate on pixel AUROC + the class-mean image AUROC.",
    ),
    "rd4ad": PublishedReference(
        method="RD4AD (WRN50, one-class)",
        citation="Deng & Li, Anomaly Detection via Reverse Distillation from One-Class "
                 "Embedding, CVPR 2022",
        image_table="Table 1 (Ours, 256x256)", pixel_table="Table 2 (Ours, AUROC column)",
        protocol=ProtocolSpec(
            method="rd4ad", preprocess=PreprocessSpec(256, None),
            backbone="wide_resnet50_2", backbone_weights="IMAGENET1K_V1",
            feature_layers=("layer1", "layer2", "layer3"), epochs=200, batch_size=16,
            hyperparams=(("lr", 0.005), ("adam_beta1", 0.5), ("adam_beta2", 0.999))),
        image_auroc=_pct({"bottle": 100, "cable": 95.0, "capsule": 96.3, "carpet": 98.9,
                          "grid": 100, "hazelnut": 99.9, "leather": 100, "metal_nut": 100,
                          "pill": 96.6, "screw": 97.0, "tile": 99.3, "toothbrush": 99.5,
                          "transistor": 96.7, "wood": 99.2, "zipper": 98.5}),
        pixel_auroc=_pct({"bottle": 98.7, "cable": 97.4, "capsule": 98.7, "carpet": 98.9,
                          "grid": 99.3, "hazelnut": 98.9, "leather": 99.4, "metal_nut": 97.3,
                          "pill": 98.2, "screw": 99.6, "tile": 95.6, "toothbrush": 99.1,
                          "transistor": 92.5, "wood": 95.3, "zipper": 98.2}),
    ),
    "simplenet": PublishedReference(
        method="SimpleNet (WRN50)",
        citation="Liu et al., SimpleNet, CVPR 2023 (arXiv:2303.15140)",
        image_table="Table 1 (I-AUROC)", pixel_table="Table 1 (P-AUROC)",
        protocol=ProtocolSpec(
            method="simplenet", preprocess=PreprocessSpec(256, 224),
            backbone="wide_resnet50_2", backbone_weights="IMAGENET1K_V1",
            feature_layers=("layer2", "layer3"), epochs=160, batch_size=4,
            hyperparams=(("embed_dim", 1536), ("noise_std", 0.015), ("adaptor_bias", False),
                         ("lr_adaptor", 1e-4), ("lr_discriminator", 2e-4),
                         ("weight_decay", 1e-5), ("loss", "truncated_l1"), ("th", 0.5))),
        image_auroc=_pct({"bottle": 100, "cable": 99.9, "capsule": 97.7, "carpet": 99.7,
                          "grid": 99.7, "hazelnut": 100, "leather": 100, "metal_nut": 100,
                          "pill": 99.0, "screw": 98.2, "tile": 99.8, "toothbrush": 99.7,
                          "transistor": 100, "wood": 100, "zipper": 99.9}),
        pixel_auroc=_pct({"bottle": 98.0, "cable": 97.6, "capsule": 98.9, "carpet": 98.2,
                          "grid": 98.8, "hazelnut": 97.9, "leather": 99.2, "metal_nut": 98.8,
                          "pill": 98.6, "screw": 99.3, "tile": 97.0, "toothbrush": 98.5,
                          "transistor": 97.6, "wood": 94.5, "zipper": 98.9}),
        notes="Paper selects checkpoints on test AUROC (official code); report both "
              "'official' and label-free protocols.",
    ),
})


# ───────────────────────────── dataset scan ────────────────────────────────
@dataclass(frozen=True)
class Sample:
    path: Path
    label: int                      # 0 = normal, 1 = anomaly
    defect_type: str
    mask_path: Optional[Path]       # None for good images


def _images_in(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT)


def scan_category(mvtec_root: Path, category: str, *, strict_counts: bool = True,
                  require_masks: bool = True) -> Tuple[List[Sample], List[Sample]]:
    """Official protocol: train = train/good, test = test/* (+ ground_truth masks)."""
    if category not in OFFICIAL_SPLIT_SIZES:
        raise DatasetLayoutError(f"Unknown category {category!r}; valid: {MVTEC_CATEGORIES}")
    root = Path(mvtec_root) / category
    train_dir, test_dir, gt_dir = root / "train" / "good", root / "test", root / "ground_truth"
    for d in (train_dir, test_dir):
        if not d.is_dir():
            raise DatasetLayoutError(f"Missing folder {d}")

    train = [Sample(p, 0, "good", None) for p in _images_in(train_dir)]
    test: List[Sample] = []
    for sub in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        is_good = sub.name == "good"
        for img in _images_in(sub):
            mask = None if is_good else gt_dir / sub.name / f"{img.stem}_mask.png"
            if mask is not None and require_masks and not mask.is_file():
                raise DatasetLayoutError(f"Missing ground-truth mask {mask} for {img}")
            test.append(Sample(img, 0 if is_good else 1, sub.name, mask))

    if strict_counts:
        expected = OFFICIAL_SPLIT_SIZES[category]
        if (len(train), len(test)) != expected:
            raise DatasetLayoutError(
                f"{category}: found (train={len(train)}, test={len(test)}), official is "
                f"{expected}. Incomplete extraction or modified dataset.")
    if not any(s.label == 1 for s in test) or not any(s.label == 0 for s in test):
        raise DatasetLayoutError(f"{category}: test split must contain both classes")
    return train, test


def manifest_sha256(samples: Iterable[Sample], root: Path) -> str:
    """Hash of (relative path, byte size) — proves which files a run used."""
    h = hashlib.sha256()
    for s in sorted(samples, key=lambda x: str(x.path)):
        rel = Path(s.path).resolve().relative_to(Path(root).resolve()).as_posix()
        h.update(f"{rel}\t{Path(s.path).stat().st_size}\t{s.label}\n".encode())
    return h.hexdigest()


# ───────────────────────────── metrics ─────────────────────────────────────
def _finite(arr: Any, name: str, ndim: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != ndim:
        raise InvalidScoreError(f"{name}: expected {ndim}-D array, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise InvalidScoreError(f"{name}: contains NaN/inf ({int((~np.isfinite(a)).sum())} values)")
    return a


def _binary(arr: Any, name: str) -> np.ndarray:
    a = np.asarray(arr)
    uniq = np.unique(a)
    if not set(uniq.tolist()) <= {0, 1}:
        raise InvalidScoreError(f"{name}: must be binary {{0,1}}, got values {uniq[:10]}")
    if uniq.size < 2:
        raise InvalidScoreError(f"{name}: needs both classes, got only {uniq.tolist()}")
    return a.astype(np.int64)


def image_level_metrics(image_scores: Any, y_true: Any) -> Dict[str, float]:
    """image_scores [N], y_true [N] in {0,1} -> {'image_auroc', 'image_ap'}."""
    s = _finite(image_scores, "image_scores", 1)
    y = _binary(y_true, "y_true")
    if s.shape != y.shape:
        raise InvalidScoreError(f"image_scores {s.shape} vs y_true {y.shape}")
    return {"image_auroc": float(roc_auc_score(y, s)),
            "image_ap": float(average_precision_score(y, s))}


def assert_raw_pixel_maps(pixel_maps: Any, *, atol: float = 1e-6,
                          max_fraction: float = 0.9) -> None:
    """Raise if (almost) every map spans exactly [0, 1] -> per-image min-max
    normalisation, which destroys cross-image comparability required by P-AUROC."""
    m = _finite(pixel_maps, "pixel_maps", 3)
    if m.shape[0] < 2:
        return
    flat = m.reshape(m.shape[0], -1)
    spans_unit = (flat.min(axis=1) <= atol) & (flat.max(axis=1) >= 1.0 - atol)
    if spans_unit.mean() > max_fraction:
        raise PerImageNormalizedMapError(
            f"{spans_unit.mean():.0%} of pixel maps span exactly [0,1] — maps look "
            f"per-image normalised. Return raw scores for P-AUROC.")


def pixel_level_auroc(pixel_maps: Any, masks: Any) -> float:
    """pixel_maps [N,H,W] raw scores, masks [N,H,W] in {0,1} -> pooled P-AUROC."""
    m = _finite(pixel_maps, "pixel_maps", 3)
    g = np.asarray(masks)
    if g.shape != m.shape:
        raise InvalidScoreError(f"pixel_maps {m.shape} vs masks {g.shape} — geometry mismatch "
                                f"(resize/crop must be identical for images and masks)")
    y = _binary(g.reshape(-1), "masks")
    return float(roc_auc_score(y, m.reshape(-1)))


# ───────────────────────────── comparison ──────────────────────────────────
class VerdictStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NO_REFERENCE = "NO_REFERENCE"
    INSUFFICIENT_SEEDS = "INSUFFICIENT_SEEDS"


@dataclass(frozen=True)
class Verdict:
    category: str
    metric: str
    n_seeds: int
    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    reference: Optional[float]
    delta: Optional[float]
    tolerance: float
    reference_quality: str
    status: VerdictStatus

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


def summarize_seeds(values: Sequence[float]) -> Tuple[float, float, float, float]:
    """-> (mean, sample std, ci95_low, ci95_high) with a Student-t interval."""
    a = _finite(values, "seed values", 1)
    if a.size == 0:
        raise InvalidScoreError("no seed values")
    mean = float(a.mean())
    if a.size == 1:
        return mean, 0.0, float("nan"), float("nan")
    std = float(a.std(ddof=1))
    half = float(stats.t.ppf(0.975, a.size - 1) * std / math.sqrt(a.size))
    return mean, std, mean - half, mean + half


def compare_to_reference(per_seed: Mapping[str, Sequence[float]], reference: Mapping[str, float],
                         *, metric: str, tolerance: float, min_seeds: int = 3,
                         reference_quality: str = SourceQuality.PRIMARY.value) -> List[Verdict]:
    """PASS iff n_seeds >= min_seeds and |mean - reference| <= tolerance."""
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    out: List[Verdict] = []
    for cat in sorted(per_seed):
        mean, std, lo, hi = summarize_seeds(per_seed[cat])
        n = len(per_seed[cat])
        ref = reference.get(cat)
        delta = None if ref is None else mean - float(ref)
        if ref is None:
            status = VerdictStatus.NO_REFERENCE
        elif n < min_seeds:
            status = VerdictStatus.INSUFFICIENT_SEEDS
        else:
            status = VerdictStatus.PASS if abs(delta) <= tolerance else VerdictStatus.FAIL
        out.append(Verdict(cat, metric, n, mean, std, lo, hi,
                           None if ref is None else float(ref), delta, tolerance,
                           reference_quality, status))
    return out


# ───────────────────────────── provenance ──────────────────────────────────
def git_provenance(repo_root: Path) -> Dict[str, Any]:
    """Never raises — a missing git binary must not kill a multi-hour run."""
    def _git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo_root), *args],
                                       text=True, stderr=subprocess.STDOUT).strip()
    try:
        dirty = _git("status", "--porcelain")
        return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(dirty),
                "dirty_files": dirty.splitlines()}
    except Exception as exc:  # noqa: BLE001 — provenance is best-effort by design
        return {"commit": None, "dirty": None, "error": f"{type(exc).__name__}: {exc}"}


def package_fingerprint() -> str:
    """SHA-256 over this package's .py sources (excluding tests)."""
    h = hashlib.sha256()
    for p in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def environment_info() -> Dict[str, Optional[str]]:
    import sklearn
    info: Dict[str, Optional[str]] = {"python": platform.python_version(),
                                      "platform": platform.platform(),
                                      "numpy": np.__version__, "sklearn": sklearn.__version__}
    try:
        import torch
        import torchvision
        info.update(torch=torch.__version__, torchvision=torchvision.__version__,
                    cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as exc:  # noqa: BLE001 — a broken/partial torch install
        # (e.g. missing shared libs -> OSError, not just ImportError) must
        # never crash environment reporting, which is best-effort by design.
        info.update(torch=None, torchvision=None, cuda=None, gpu=None,
                    torch_import_error=f"{type(exc).__name__}: {exc}")
    return info


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return _json_safe(obj.item())
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def write_run_record(path: Path, record: Mapping[str, Any], *, sort_keys: bool = False) -> Path:
    """Atomic JSON write (tmp + replace) so a crash never leaves half a file.

    sort_keys defaults to False (changed from the original True): the
    caller now builds the record with a deliberate key order (e.g. paper
    reference values placed before the raw per-seed results within each
    category, so a human reading the file top-to-bottom sees "paper says
    X" before "here's what we measured") — alphabetical sorting would
    destroy that intentional ordering. Python dicts preserve insertion
    order, and json.dumps does too when sort_keys=False.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_safe(record), indent=2, ensure_ascii=False,
                              sort_keys=sort_keys), encoding="utf-8")
    tmp.replace(path)
    logger.info("run record written: %s", path)
    return path


def build_run_record(reference: "PublishedReference", protocol: "ProtocolSpec",
                     categories: List[Dict[str, Any]], started_at: datetime,
                     finished_at: datetime, *, image_tol: float = 0.01,
                     pixel_tol: float = 0.01, min_seeds: int = 3,
                     compute_pixel: bool = True,
                     declared_deviations: Sequence[str] = ()) -> Dict[str, Any]:
    """Assemble the final JSON record in the exact 4-section shape the user
    asked for, in this key order (json.dumps preserves it since
    write_run_record no longer sort_keys):

    1) run_info    — started/finished time in Thailand (ICT) + environment only
                      (git commit / dirty-state / harness fingerprint deliberately
                      dropped from the top-level view per the user's spec — they
                      still live inside each category's raw per-seed data if
                      ever needed, but are not part of what gets read first)
    2) protocol    — the config actually used for this run
    3) results     — paper's own reference value FIRST, then the raw per-seed
                      numbers, for every category
    4) summary     — mean across seeds vs. paper, with the gap and a verdict

    `categories` is a list of dicts, each already containing
    {"category", "n_train", "n_test", "per_seed": [...]} as produced by
    torch_runner.run_category() — this function does not run anything
    itself, it only reshapes + summarizes.
    """
    def _collect(key: str) -> Dict[str, List[float]]:
        return {c["category"]: [r[key] for r in c["per_seed"] if key in r] for c in categories}

    image_per_seed = _collect("image_auroc")
    pixel_per_seed = _collect("pixel_auroc") if compute_pixel else {}

    image_verdicts = {v.category: v for v in compare_to_reference(
        image_per_seed, reference.image_auroc, metric="image_auroc",
        tolerance=image_tol, min_seeds=min_seeds,
        reference_quality=reference.image_quality.value)}
    pixel_verdicts = ({v.category: v for v in compare_to_reference(
        pixel_per_seed, reference.pixel_auroc, metric="pixel_auroc",
        tolerance=pixel_tol, min_seeds=min_seeds,
        reference_quality=reference.pixel_quality.value)} if compute_pixel else {})

    # Section 3: paper reference value written first, per_seed raw results after.
    results_categories = []
    for c in categories:
        cat = c["category"]
        entry = {
            "category": cat,
            "paper_image_auroc": reference.image_auroc.get(cat),
            "paper_pixel_auroc": reference.pixel_auroc.get(cat) if compute_pixel else None,
            "n_train": c["n_train"], "n_test": c["n_test"],
            "per_seed": c["per_seed"],
        }
        results_categories.append(entry)

    # Section 4: mean vs paper, with the gap, per category per metric.
    summary = []
    for cat in sorted(image_per_seed):
        v = image_verdicts[cat]
        summary.append({
            "category": cat, "metric": "image_auroc", "n_seeds": v.n_seeds,
            "mean": round(v.mean, 4), "std": round(v.std, 4),
            "paper": v.reference, "diff_vs_paper": (round(v.delta, 4) if v.delta is not None else None),
            "status": v.status.value,
        })
    for cat in sorted(pixel_per_seed):
        v = pixel_verdicts[cat]
        summary.append({
            "category": cat, "metric": "pixel_auroc", "n_seeds": v.n_seeds,
            "mean": round(v.mean, 4), "std": round(v.std, 4),
            "paper": v.reference, "diff_vs_paper": (round(v.delta, 4) if v.delta is not None else None),
            "status": v.status.value,
        })

    n_pass = sum(1 for s in summary if s["status"] == "PASS")
    n_checkable = sum(1 for s in summary if s["status"] in ("PASS", "FAIL"))

    return {
        "run_info": {
            "started_ict": ict_isoformat(started_at),
            "finished_ict": ict_isoformat(finished_at),
            "environment": environment_info(),
        },
        "protocol": protocol.to_dict(),
        "results": {
            "paper_citation": reference.citation,
            "paper_image_table": reference.image_table,
            "paper_pixel_table": reference.pixel_table,
            "paper_notes": reference.notes,
            "declared_deviations": list(declared_deviations),
            "categories": results_categories,
        },
        "summary": summary,
        "pass_rate": {
            "passed": n_pass, "checkable": n_checkable,
            "percent": round(100 * n_pass / n_checkable, 1) if n_checkable else None,
        },
    }
