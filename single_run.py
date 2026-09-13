"""
One-shot entry point สำหรับ PaDiM — แก้ OVERRIDES แล้วรัน:
    python single_run.py

รันครบ 3 step ในคำสั่งเดียว เหมือน single_run.py ของ repo หลัก:
  [1/3] run_padim  — fit + score + save .npz/.json/.csv
  [2/3] visualize      — สร้างภาพทุกใบจาก .npz ที่เพิ่งเซฟ
  [3/3] cost_aware     — threshold sweep (ปิดได้ผ่าน toggle ด้านล่าง)

OVERRIDES เป็นเพียงที่เดียวที่ต้องแก้ — multi_run.py import
OVERRIDES จากไฟล์นี้โดยตรง ไม่ copy ซ้ำ กัน 2 ไฟล์ไม่ sync กัน

Runs all 3 steps in one command, same as the main repo's single_run.py:
  [1/3] run_padim  — fit + score + save .npz/.json/.csv
  [2/3] visualize      — generate all images from the just-saved .npz
  [3/3] cost_aware     — threshold sweep (toggle off below if not needed)

OVERRIDES is the only file to edit — multi_run.py imports OVERRIDES
directly from here, no copy, no drift risk.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import Config
import scripts.run_padim         as run_padim
import scripts.visualize_padim   as visualize_padim
import scripts.run_cost_aware_padim as run_cost_aware_padim

# ── เปิด/ปิด step [3/3] ─────────────────────────────────────────────
# ปิดได้ถ้าแค่ต้องการ fit + score + visualize โดยไม่รัน cost-aware sweep
# ไม่กระทบ step [1/2] เลยไม่ว่าจะตั้งเป็นอะไร
#
# Turn off if you only want fit + score + visualize without the sweep.
# Never affects steps [1/2] regardless of this value.
RUN_COST_AWARE_ANALYSIS = True

OVERRIDES = dict(
    # ── Data & paths — แก้ก่อนรันจริง ──────────────────────────────
    DATA_ROOT="dataset root path (contains good/ and defect/ subfolders)",
    GOOD_DIRNAME="good",
    DEFECT_DIRNAME="defect",

    # ชี้ไปที่ split_assignment.csv เดียวกับ repo หลัก เพื่อให้
    # train/val/test membership ตรงกันเป๊ะตอนเทียบ AE กับ PaDiM
    # ปล่อยเป็น path เดียวไม่มี "SEED {n}" — multi_run.py (default
    # VARY_SPLIT_PER_SEED=False) ใช้ path นี้ตัวเดียวกันทุก seed โดย
    # ไม่ต้องแก้อะไรเพิ่ม ถ้าต้องการวัด split variance ด้วย (คนละคำถาม
    # วิจัย) ให้เปิด VARY_SPLIT_PER_SEED=True ใน multi_run.py แล้ว
    # ค่อยฝัง "SEED 42" ไว้ใน path นี้
    #
    # Point at the same split_assignment.csv as the main repo so
    # train/val/test membership matches exactly when comparing AE vs
    # PaDiM. Leave it as one fixed path (no "SEED {n}") —
    # multi_run.py (default VARY_SPLIT_PER_SEED=False) reuses this
    # exact path for every seed with no further changes needed. Only
    # embed "SEED {n}" here if you deliberately set
    # VARY_SPLIT_PER_SEED=True to study split variance itself.
    SPLIT_CACHE_PATH="splits/split_assignment.csv",
    SAVE_PATH="save log/SEED 42",
    OUTPUT_PATH="save image/table/SEED 42",
    SEED=42,

    # ── Model config — ปรับได้ตามต้องการ ────────────────────────────
    EXPERIMENT="PaDiM_group1_wide_resnet50_2",
    BACKBONE="wide_resnet50_2",
    NUM_SELECTED_CHANNELS=100,
    COV_REG_EPSILON=0.01,
    THRESHOLD_PERCENTILE=95.0,
)

if __name__ == "__main__":
    _n_steps = 3 if RUN_COST_AWARE_ANALYSIS else 2
    cfg = Config(**OVERRIDES)

    print(f"\n--- [1/{_n_steps}] PaDiM: fit + score + save ---")
    run_padim.run(cfg)

    print(f"\n--- [2/{_n_steps}] Visualize: สร้างภาพทั้งหมด ---")
    visualize_padim.visualize(cfg)

    if RUN_COST_AWARE_ANALYSIS:
        # อ่านจาก .npz ที่ step 1 เพิ่งเซฟ — ไม่รัน inference ใหม่
        # Reads .npz saved by step 1 — no new inference
        print(f"\n--- [3/{_n_steps}] Cost-Aware Threshold Sweep ---")
        run_cost_aware_padim.main()

    print("\n✅ เสร็จสิ้นกระบวนการทั้งหมดเรียบร้อย!")