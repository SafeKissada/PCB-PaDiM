"""
mvtec_repro.torch_runner — torch-dependent half of the harness.

The SAME runner drives all four baselines; model code is injected through a
``ModelFactory`` (dependency injection), so the only per-repo code is a ~20-line
adapter that maps a ``ProtocolSpec`` onto that repo's model constructor.

Contract with the existing models (src/models/*.py):
  * ``model.fit(normal_loader)`` / ``model.score(loader) -> ScoreResult``
  * loader batches are the existing 6-tuple
    (norm_img[B,3,S,S], disp_img[B,3,S,S], disp_img, paths, labels, (w, h))
  * ``ScoreResult.pixel_maps`` must be RAW scores [N,S,S] at S = output_size
"""
from __future__ import annotations

import dataclasses
import gc
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, v2

from . import core

logger = logging.getLogger("mvtec_repro")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ScoreResultLike(Protocol):
    image_scores: np.ndarray   # [N]
    y_true: np.ndarray         # [N]
    paths: List[str]           # [N]
    pixel_maps: Optional[np.ndarray]  # [N, S, S]


class AnomalyModel(Protocol):
    def fit(self, normal_loader: DataLoader) -> None: ...
    def score(self, loader: DataLoader) -> ScoreResultLike: ...


ModelFactory = Callable[[core.ProtocolSpec, int, torch.device], AnomalyModel]


# ───────────────────────── geometry (image + mask) ─────────────────────────
def build_transforms(pre: core.PreprocessSpec) -> Tuple[v2.Compose, v2.Compose]:
    """-> (display_tf [3,S,S] in 0..1, model_tf [3,S,S] ImageNet-normalised)."""
    geo: List[Any] = [v2.Resize((pre.resize, pre.resize),
                                interpolation=InterpolationMode.BILINEAR, antialias=True)]
    if pre.center_crop is not None:
        geo.append(v2.CenterCrop(pre.center_crop))
    base = [*geo, v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    return v2.Compose(base), v2.Compose([*base, v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def load_mask(mask_path: Optional[Path], pre: core.PreprocessSpec) -> np.ndarray:
    """Binary mask [S,S] uint8 with EXACTLY the image geometry (nearest interp)."""
    s = pre.output_size
    if mask_path is None:
        return np.zeros((s, s), dtype=np.uint8)
    with Image.open(mask_path) as m:
        arr = np.asarray(m.convert("L").resize((pre.resize, pre.resize), Image.NEAREST))
    off = pre.crop_offset()
    arr = arr[off:off + s, off:off + s]
    if arr.shape != (s, s):
        raise core.DatasetLayoutError(f"mask {mask_path} -> {arr.shape}, expected {(s, s)}")
    return (arr > 127).astype(np.uint8)


class MVTecDataset(Dataset):
    """Strict dataset: an unreadable file raises (no random-substitute fallback —
    a silently swapped test image would corrupt the benchmark)."""

    def __init__(self, samples: Sequence[core.Sample], pre: core.PreprocessSpec) -> None:
        self.samples = list(samples)
        self.display_tf, self.model_tf = build_transforms(pre)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        try:
            with Image.open(s.path) as img:
                img = img.convert("RGB")
                w, h = img.size
                disp = self.display_tf(img)
                norm = self.model_tf(img)
        except (OSError, ValueError) as exc:
            raise core.DatasetLayoutError(f"cannot read {s.path}: {exc}") from exc
        label = "anomaly" if s.label == 1 else "normal"
        return norm, disp, disp, str(s.path), label, (w, h)


# ───────────────────────── determinism ─────────────────────────────────────
def seed_everything(seed: int, deterministic: bool = True) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.Generator().manual_seed(seed)


def _worker_init(worker_id: int) -> None:
    ws = torch.initial_seed() % 2**32
    np.random.seed(ws + worker_id)
    random.seed(ws + worker_id)


def make_loader(ds: Dataset, batch_size: int, *, shuffle: bool, generator: torch.Generator,
                num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=device.type == "cuda", drop_last=False,
                      generator=generator, worker_init_fn=_worker_init,
                      persistent_workers=False)


# ───────────────────────── runner ──────────────────────────────────────────
def _check_alignment(result: ScoreResultLike, test: Sequence[core.Sample]) -> None:
    got = [str(p) for p in result.paths]
    exp = [str(s.path) for s in test]
    if got != exp:
        raise core.InvalidScoreError("score() changed sample order — masks would be misaligned")
    if not np.array_equal(np.asarray(result.y_true), np.array([s.label for s in test])):
        raise core.InvalidScoreError("score() y_true disagrees with dataset labels")


def _release(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


class ModelIntegrationError(core.ReproError):
    """Raised by smoke_test_model() when a model/adapter is wired wrong —
    e.g. wrong Config field name, PatchCore/PaDiM missing weights_override
    plumbing, a shape mismatch between what score() returns and what the
    harness expects. Meant to fail in seconds on fake data, before anyone
    burns GPU time running the real (possibly hours-long) MVTec sweep only
    to discover the adapter was broken.
    """


def smoke_test_model(model_factory: ModelFactory, spec: core.ProtocolSpec, device: torch.device,
                     *, n_normal: int = 6, n_test: int = 4) -> None:
    """Run fit()+score() once on tiny synthetic images before touching real
    MVTec data. Catches integration bugs — wrong Config field name typo'd
    in an adapter, a model that silently expects a field the adapter never
    set (e.g. PaDiM needing RAW_PIXEL_MAPS=True or its pixel_maps come back
    per-image-normalized and useless for P-AUROC), a ScoreResult whose
    pixel_maps shape doesn't match IMAGE_SIZE — in seconds, on tiny fake
    data, rather than after a multi-hour real run.

    Builds core.Sample objects directly (does NOT go through
    core.scan_category(), which intentionally rejects any category name
    outside the official 15 MVTec categories — exactly right for a real
    run, but incompatible with a throwaway fake category used only here).

    This does NOT verify the model's architecture matches its paper (see
    each repo's own reproducibility_proof/ for that, where it exists) — it
    only verifies the *adapter wiring* is self-consistent enough to run at
    all and to produce a ScoreResult the rest of the harness can consume.
    """
    import shutil
    import tempfile

    tmp_root = Path(tempfile.mkdtemp(prefix="mvtec_repro_smoke_"))
    try:
        s = spec.preprocess.resize  # generate at pre-crop resolution; crop happens in the transform
        rng = np.random.default_rng(0)

        def _make_png(path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (s, s, 3), dtype=np.uint8)).save(path)

        train: List[core.Sample] = []
        for i in range(n_normal):
            p = tmp_root / "train" / f"{i:03d}.png"
            _make_png(p)
            train.append(core.Sample(p, 0, "good", None))

        test: List[core.Sample] = []
        n_good_test = max(1, n_test // 2)
        for i in range(n_good_test):
            p = tmp_root / "test_good" / f"{i:03d}.png"
            _make_png(p)
            test.append(core.Sample(p, 0, "good", None))
        for i in range(n_test - n_good_test):
            p = tmp_root / "test_bad" / f"{i:03d}.png"
            _make_png(p)
            mask_p = tmp_root / "mask" / f"{i:03d}_mask.png"
            mask_p.parent.mkdir(parents=True, exist_ok=True)
            mask = np.zeros((s, s), dtype=np.uint8)
            mask[s // 4: s // 2, s // 4: s // 2] = 255
            Image.fromarray(mask).save(mask_p)
            test.append(core.Sample(p, 1, "fake_defect", mask_p))

        masks = np.stack([load_mask(sm.mask_path, spec.preprocess) for sm in test])

        gen = seed_everything(0)
        train_loader = make_loader(MVTecDataset(train, spec.preprocess), min(4, n_normal),
                                   shuffle=True, generator=gen, num_workers=0, device=device)
        test_loader = make_loader(MVTecDataset(test, spec.preprocess), min(4, n_test),
                                  shuffle=False, generator=gen, num_workers=0, device=device)

        try:
            model = model_factory(spec, 0, device)
            model.fit(train_loader)
            result = model.score(test_loader)
        except Exception as exc:  # noqa: BLE001 — re-raise as a labeled integration error
            raise ModelIntegrationError(
                f"smoke test failed during fit()/score() — the adapter or Config passed to "
                f"the model factory is likely wrong: {type(exc).__name__}: {exc}") from exc

        _check_alignment(result, test)
        if result.pixel_maps is None:
            raise ModelIntegrationError("score() returned pixel_maps=None — P-AUROC cannot be computed")
        if result.pixel_maps.shape != masks.shape:
            raise ModelIntegrationError(
                f"pixel_maps shape {result.pixel_maps.shape} != mask shape {masks.shape} — "
                f"the model's output geometry doesn't match spec.preprocess.output_size "
                f"({spec.preprocess.output_size}); check IMAGE_SIZE in the adapter's Config")
        try:
            core.assert_raw_pixel_maps(result.pixel_maps)
        except core.PerImageNormalizedMapError as exc:
            raise ModelIntegrationError(
                f"{exc} — this adapter forgot to request raw pixel maps from the model "
                f"(e.g. PaDiM needs cfg.RAW_PIXEL_MAPS=True)") from exc
        logger.info("smoke test passed for %s", spec.method)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        _release(device)


def run_category(model_factory: ModelFactory, reference: core.PublishedReference,
                 mvtec_root: Path, category: str, seeds: Sequence[int], device: torch.device,
                 *, protocol: Optional[core.ProtocolSpec] = None, num_workers: int = 2,
                 compute_pixel: bool = True,
                 on_seed_start: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """Fit/score one category for every seed under one protocol.

    ``on_seed_start(seed_index_1based, n_seeds)``, if given, is called right
    before each seed starts — this is how run_benchmark() reports a single,
    uniform "N/M runs done (X%)" progress banner across all four methods,
    regardless of what each model's own fit() does internally (PatchCore/
    PaDiM have no training loop and log feature-extraction progress instead;
    RD4AD/SimpleNet log per-epoch progress) — the run-level percentage is a
    property of the harness's loop, not of the model, so it can be uniform
    even when the per-model internals aren't.
    """
    spec = protocol or reference.protocol
    train, test = core.scan_category(Path(mvtec_root), category, require_masks=compute_pixel)
    masks = (np.stack([load_mask(s.mask_path, spec.preprocess) for s in test])
             if compute_pixel else None)                                  # [N,S,S]
    per_seed: List[Dict[str, Any]] = []

    for i, seed in enumerate(seeds, 1):
        if on_seed_start is not None:
            on_seed_start(i, len(seeds))
        gen = seed_everything(seed)
        train_loader = make_loader(MVTecDataset(train, spec.preprocess), spec.batch_size,
                                   shuffle=True, generator=gen, num_workers=num_workers,
                                   device=device)
        test_loader = make_loader(MVTecDataset(test, spec.preprocess), spec.batch_size,
                                  shuffle=False, generator=gen, num_workers=num_workers,
                                  device=device)
        model = model_factory(spec, seed, device)
        try:
            t0 = time.perf_counter()
            model.fit(train_loader)
            t1 = time.perf_counter()
            result = model.score(test_loader)
            t2 = time.perf_counter()
        except torch.cuda.OutOfMemoryError as exc:
            _release(device)
            raise core.ReproError(
                f"CUDA OOM: {category} seed={seed} batch_size={spec.batch_size}. Lower "
                f"batch_size or the model's chunk size; do NOT change it silently — it is "
                f"part of the protocol record.") from exc

        _check_alignment(result, test)
        row: Dict[str, Any] = {"seed": seed, "fit_sec": round(t1 - t0, 2),
                               "score_sec": round(t2 - t1, 2),
                               **core.image_level_metrics(result.image_scores, result.y_true)}
        if compute_pixel:
            if result.pixel_maps is None:
                raise core.InvalidScoreError("model returned no pixel_maps")
            core.assert_raw_pixel_maps(result.pixel_maps)
            row["pixel_auroc"] = core.pixel_level_auroc(result.pixel_maps, masks)
        logger.info("[%s] seed=%d I-AUROC=%.4f P-AUROC=%s", category, seed,
                    row["image_auroc"], f"{row.get('pixel_auroc', float('nan')):.4f}")
        per_seed.append(row)
        del model, result
        _release(device)

    return {"category": category, "n_train": len(train), "n_test": len(test),
            "dataset_manifest_sha256": core.manifest_sha256(train + test, Path(mvtec_root)),
            "per_seed": per_seed}


def run_benchmark(model_factory: ModelFactory, reference: core.PublishedReference,
                  mvtec_root: Path, categories: Sequence[str], seeds: Sequence[int],
                  out_path: Path, repo_root: Path, *, protocol: Optional[core.ProtocolSpec] = None,
                  image_tol: float = 0.01, pixel_tol: float = 0.01, min_seeds: int = 3,
                  device: Optional[torch.device] = None, num_workers: int = 2,
                  compute_pixel: bool = True, deviations: Sequence[str] = (),
                  skip_smoke_test: bool = False) -> Dict[str, Any]:
    """Run all categories, compare to the reference, write ONE self-describing record.

    ``deviations``: human-readable list of every knowingly-kept difference from the
    paper protocol (e.g. "backbone weights IMAGENET1K_V2"). A proof without this list
    is not reviewable.

    Progress reporting: prints one banner per (category, seed) pair showing
    "N/total runs done (X%)" — this is uniform across PatchCore/PaDiM/RD4AD/
    SimpleNet regardless of how each model reports its own internal progress
    (or whether it has any training loop at all), because it counts completed
    (category, seed) pairs at the harness level, not anything model-specific.

    Before touching any real MVTec data, runs smoke_test_model() once on tiny
    synthetic images (skip_smoke_test=True to bypass, e.g. when iterating
    quickly and the adapter is already known-good) — this fails in seconds
    if the adapter/Config wiring is wrong, instead of after a multi-hour run.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = protocol or reference.protocol
    if not skip_smoke_test:
        # For training-loop methods (RD4AD/SimpleNet), run only a handful of
        # epochs during the smoke test — the whole point is a fast fail,
        # not a faithful (if tiny) training run. ProtocolSpec is frozen, so
        # this creates a separate copy; the real sweep below still uses the
        # original `spec` with its full epoch count.
        smoke_spec = spec if spec.epochs is None else dataclasses.replace(spec, epochs=min(3, spec.epochs))
        logger.info("running smoke test on synthetic data before the real MVTec sweep...")
        smoke_test_model(model_factory, smoke_spec, device, n_normal=max(6, spec.batch_size // 2))
    started_at = core.now_ict()
    total_runs = len(categories) * len(seeds)
    done = 0

    def _make_progress_cb(category: str) -> Callable[[int, int], None]:
        def _cb(seed_idx: int, n_seeds: int) -> None:
            nonlocal done
            done += 1
            pct = 100 * done / total_runs
            banner = (f"[{done}/{total_runs} · {pct:.0f}%] {reference.method} "
                     f"— {category} — seed {seed_idx}/{n_seeds}")
            print("=" * len(banner))
            print(banner)
            print("=" * len(banner))
        return _cb

    cats = [run_category(model_factory, reference, mvtec_root, c, seeds, device,
                         protocol=spec, num_workers=num_workers, compute_pixel=compute_pixel,
                         on_seed_start=_make_progress_cb(c))
            for c in categories]
    finished_at = core.now_ict()

    record = core.build_run_record(
        reference, spec, cats, started_at, finished_at,
        image_tol=image_tol, pixel_tol=pixel_tol, min_seeds=min_seeds,
        compute_pixel=compute_pixel, declared_deviations=deviations)

    core.write_run_record(out_path, record)
    _print_summary(record, reference)
    return record


def _print_summary(record: Dict[str, Any], reference: core.PublishedReference) -> None:
    """Human-readable summary table printed at the end of a run — the
    Colab-cell-friendly counterpart to the JSON file."""
    width = 88
    print("\n" + "=" * width)
    print(f"{reference.method}  vs  {reference.citation}")
    proto = record["protocol"]
    print(f"resize={proto['preprocess']['resize']} "
         f"center_crop={proto['preprocess']['center_crop']}  "
         f"weights={proto['backbone_weights']}")
    print("=" * width)
    header = f"{'category':<12}{'metric':<14}{'mean±std':<18}{'paper':>8}{'diff':>9}   status"
    print(header)
    print("-" * width)
    for row in record["summary"]:
        mean_std = f"{row['mean']:.4f}±{row['std']:.4f}"
        paper_s = f"{row['paper']:.3f}" if row["paper"] is not None else "N/A"
        diff_s = f"{row['diff_vs_paper']:+.4f}" if row["diff_vs_paper"] is not None else "N/A"
        mark = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "NO_REFERENCE": "— N/A",
               "INSUFFICIENT_SEEDS": "⚠️ NEED MORE SEEDS"}[row["status"]]
        print(f"{row['category']:<12}{row['metric']:<14}{mean_std:<18}{paper_s:>8}{diff_s:>9}   {mark}")
    print("-" * width)
    pr = record["pass_rate"]
    if pr["percent"] is not None:
        print(f"Pass rate: {pr['passed']}/{pr['checkable']} checks ({pr['percent']}%)")
    print("=" * width)
