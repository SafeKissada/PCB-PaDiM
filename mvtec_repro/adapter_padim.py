"""
mvtec_repro.adapter_padim — PaDiM adapter for the unified MVTec reproduction
harness.

Drop this file (plus the rest of mvtec_repro/) at the repo root of
PCB-Anomaly-PaDiM, replacing the old mvtec_reproduce/ folder.

Usage:
    python -m mvtec_repro.adapter_padim /path/to/mvtec_ad \
        --categories bottle wood transistor capsule screw cable \
        --seeds 0 1 2

⚠️ Known caveat carried over from the earlier analysis of this repo's
per-class image-AUROC reference table: two independent secondary sources
(RD4AD CVPR2022 Table 1 vs SimpleNet CVPR2023 Table 1) disagree on PaDiM's
per-category image-AUROC in several categories (e.g. bottle 99.9 vs 99.1,
screw 85.8 vs 97.5) while agreeing closely on the class-mean. core.py uses
the RD4AD-sourced numbers and marks them SourceQuality.SECONDARY — treat
any single-category image-AUROC verdict here as advisory, and lean on the
pixel-AUROC verdicts (which agree across sources) and the class-mean
image-AUROC for the real correctness call. If you need citation-grade
per-class numbers, transcribe them from the original PaDiM paper's own
appendix table instead of trusting either secondary source blindly.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import torch

from mvtec_repro import core
from mvtec_repro.torch_runner import run_benchmark

METHOD_KEY = "padim"


def make_padim(spec: core.ProtocolSpec, seed: int, device: torch.device):
    """ProtocolSpec -> PaDiM instance.

    Every field below is read directly by src/models/padim.py (checked
    line by line against PaDiM.__init__ / .fit() / .score()).

    RAW_PIXEL_MAPS=True is NOT optional here: src/models/padim.py's
    score() normalizes pixel_maps to per-image [0,1] by default (a
    deliberate fix for a *visualization* bug in the production PCB
    pipeline — see that file's own comments), which makes P-AUROC
    meaningless across images. This adapter always requests the raw,
    un-normalized map; the harness's smoke test would catch it and raise
    ModelIntegrationError if this were forgotten (core.PerImageNormalizedMapError).
    """
    from src.models.padim import PaDiM  # the production class under test

    s = spec.preprocess.output_size
    cfg = SimpleNamespace(
        DEVICE=device,
        SEED=seed,
        PRETRAINED=True,
        BACKBONE=spec.backbone,
        BACKBONE_WEIGHTS=spec.backbone_weights,
        FEATURE_LAYERS=spec.feature_layers,       # paper: layer1+layer2+layer3
        IMAGE_SIZE=(s, s),
        NUM_SELECTED_CHANNELS=spec.hp("num_selected_channels", 550),  # WR50-Rd550
        COV_REG_EPSILON=0.01,                     # paper's regularization epsilon
        HEATMAP_SIGMA=spec.heatmap_sigma,
        # See docstring above — required for P-AUROC to be meaningful.
        RAW_PIXEL_MAPS=True,
    )
    return PaDiM(cfg)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mvtec_root", type=Path)
    ap.add_argument("--categories", nargs="+", default=list(core.MVTEC_CATEGORIES),
                    help=f"default: all 15. Valid: {', '.join(core.MVTEC_CATEGORIES)}")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="PaDiM has no training loop, but the random "
                        "channel-selection step (NUM_SELECTED_CHANNELS out "
                        "of the full concatenated feature) is seed-dependent.")
    ap.add_argument("--out", type=Path, default=Path("mvtec_repro_results/padim.json"))
    ap.add_argument("--no-pixel-auroc", action="store_true")
    ap.add_argument("--skip-smoke-test", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    reference = core.REFERENCES[METHOD_KEY]
    run_benchmark(
        make_padim, reference, args.mvtec_root, args.categories, args.seeds,
        args.out, repo_root=Path(__file__).resolve().parents[1],
        compute_pixel=not args.no_pixel_auroc, num_workers=args.num_workers,
        skip_smoke_test=args.skip_smoke_test,
        deviations=[
            "per-category image-AUROC reference values are SECONDARY-quality "
            "(see this file's module docstring) — two third-party sources "
            "disagree per-category while agreeing on the class mean; treat "
            "single-category image_auroc verdicts as advisory only",
        ],
    )


if __name__ == "__main__":
    main()
