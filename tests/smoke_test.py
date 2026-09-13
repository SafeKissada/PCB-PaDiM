"""Smoke test — เช็คว่า PaDiM pipeline รันจบไม่มี error ก่อนรันจริง"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import Config
from scripts.run_padim import run


def make_dummy_dataset(root: Path, n_good=20, n_defect=8, size=(64, 64)):
    good_dir = root / "good"
    defect_dir = root / "defect"
    good_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(n_good):
        arr = rng.randint(100, 140, (*size, 3), dtype=np.uint8)
        Image.fromarray(arr).save(good_dir / f"good_{i:03d}.png")
    for i in range(n_defect):
        arr = rng.randint(0, 255, (*size, 3), dtype=np.uint8)
        Image.fromarray(arr).save(defect_dir / f"defect_{i:03d}.png")


def main():
    tmp_root = Path("/tmp/padim_smoke_test")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    data_root = tmp_root / "data"
    make_dummy_dataset(data_root)

    cfg = Config(
        DATA_ROOT=str(data_root),
        SPLIT_CACHE_PATH=str(tmp_root / "splits" / "split_assignment.csv"),
        SAVE_PATH=str(tmp_root / "save/logs"),
        OUTPUT_PATH=str(tmp_root / "save/results"),
        IMAGE_SIZE=(64, 64),
        BATCH_SIZE=4,
        NUM_WORKERS=0,
        BACKBONE="resnet18",
        PRETRAINED=False,  # sandbox นี้โหลด pretrained weight ไม่ได้ — รันจริงต้อง True
        FEATURE_LAYERS=("layer1", "layer2"),
        NUM_SELECTED_CHANNELS=32,
        COV_REG_EPSILON=0.01,
        EXPERIMENT="smoke_test",
    )

    val_result, test_result = run(cfg)

    assert val_result.image_scores.shape[0] == len(val_result.labels)
    assert test_result.image_scores.shape[0] == len(test_result.labels)
    assert val_result.pixel_maps.shape[1:] == cfg.IMAGE_SIZE
    assert not np.isnan(val_result.image_scores).any()
    assert not np.isnan(test_result.image_scores).any()

    # Regression guard สำหรับบั๊กที่เจอจริงจากภาพ gallery: pixel_maps ต้อง
    # normalize เป็น [0,1] ต่อภาพเสมอ ไม่ใช่ Mahalanobis distance ดิบ —
    # ไม่งั้น src/visual.py::overlay_heatmap()'s heat.clip(0,1) จะ clip
    # เกือบทุก pixel เป็นสีเดียวกันหมด (ดู src/models/padim.py::score())
    #
    # Regression guard for the real bug found from a gallery screenshot:
    # pixel_maps must always be per-image [0,1] normalized, not the raw
    # Mahalanobis distance — otherwise src/visual.py::overlay_heatmap()'s
    # heat.clip(0,1) clips nearly every pixel to the same color (see
    # src/models/padim.py::score()).
    assert val_result.pixel_maps.min() >= 0.0
    assert val_result.pixel_maps.max() <= 1.0
    assert test_result.pixel_maps.min() >= 0.0
    assert test_result.pixel_maps.max() <= 1.0
    # image_scores ต้องไม่ถูก normalize (ต้องคง scale ดิบของ Mahalanobis
    # distance ไว้สำหรับ AUROC/threshold) — ต้องมีอย่างน้อย 1 ภาพที่ max
    # score เกิน 1.0 ถ้าไม่ใช่กรณีบังเอิญที่ทุกภาพ distance ต่ำมาก
    # (dummy dataset นี้แยกคลาสชัดเจน ควรมี anomaly score สูงกว่า 1 แน่นอน)
    #
    # image_scores must stay un-normalized (raw Mahalanobis-distance
    # scale, needed for AUROC/threshold selection) — should include at
    # least one score above 1.0 for this dummy dataset (clearly
    # separable classes), unless every image coincidentally had very low
    # distance.
    assert test_result.image_scores.max() > 1.0, (
        "image_scores ดูเหมือนถูก normalize ไปด้วย ทั้งที่ควรเป็นค่าดิบ — "
        "เช็ค src/models/padim.py::score() ว่า image_scores มาจาก pmap_np "
        "(ก่อน normalize) ไม่ใช่ pmap_vis (หลัง normalize)")

    print("\n✅ SMOKE TEST PASSED (PaDiM) — pipeline รันจบไม่มี error")
    shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
