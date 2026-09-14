"""Sensitivity sweep: NUM_SELECTED_CHANNELS × COV_REG_EPSILON

README.md หัวข้อ 2 เตือนไว้ว่า group ที่มีภาพ normal น้อย (group 5: 159,
group 3: 155) เสี่ยง covariance matrix เกือบ singular ถ้า
NUM_SELECTED_CHANNELS ใกล้เคียงหรือมากกว่าจำนวนภาพ train — แต่ยังไม่เคยมี
สคริปต์ตรวจสอบเชิงตัวเลขจริง (ดู 00_INDEX.md ข้อ 5) สคริปต์นี้เติมช่องว่าง
นั้น: fit+score PaDiM ซ้ำหลายค่า (NUM_SELECTED_CHANNELS, COV_REG_EPSILON)
บน group ที่ตั้งไว้ใน RUN.OVERRIDES แล้วดูว่า val AUROC/escape_rate
เปลี่ยนแปลงมากแค่ไหน — ถ้าเปลี่ยนแปลงมาก (ill-conditioned) ต้องเลือกค่า
default ให้ระมัดระวังกว่าเดิมสำหรับ group นั้นโดยเฉพาะ ไม่ใช้ default
เดียวกันทุก group

รัน "เร็ว" กว่า multi_run.py มาก เพราะไม่ visualize/cost-aware sweep
ต่อรอบ — fit()+score() อย่างเดียว แล้วอ่าน val AUROC เท่านั้น

README.md §2 warns that groups with few normal images (group 5: 159,
group 3: 155) risk a near-singular covariance matrix when
NUM_SELECTED_CHANNELS is close to or larger than the number of train
images — but there was never an actual numeric sweep to check this (see
00_INDEX.md item 5). This script fills that gap: re-fits+scores PaDiM
across several (NUM_SELECTED_CHANNELS, COV_REG_EPSILON) combinations on
whatever group is configured in RUN.OVERRIDES, and reports how much val
AUROC/escape_rate move — large movement means ill-conditioning, and the
default values should be chosen more conservatively for that group
specifically rather than reused as-is across every group.

Much faster than multi_run.py since it skips visualize/cost-aware
per combination — only fit()+score(), reading val AUROC.

Usage:
    python scripts/run_channel_sensitivity.py
    (configure DATA_ROOT/SPLIT_CACHE_PATH in single_run.py, same as run_padim.py)
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.config import Config, set_seed
from src.data.dataset import build_datasets_and_loaders
from src.evaluate import compute_metrics, select_percentile_threshold
from src.models.padim import PaDiM

logging.basicConfig(level=logging.WARNING,  # เงียบ log ของ PaDiM.fit() ระหว่าง sweep — เห็นแค่ผลสรุปตอนจบ
                    format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("run_channel_sensitivity")

# ── ค่าที่จะ sweep — แก้ได้ตามต้องการ ──────────────────────────────
NUM_SELECTED_CHANNELS_VALUES = [25, 50, 100, 150]
COV_REG_EPSILON_VALUES = [0.01, 0.05, 0.1]


def main():
    from single_run import OVERRIDES

    # โหลดข้อมูล/split ครั้งเดียว reuse ข้าม combination ทั้งหมด — ไม่ต้อง
    # scan disk ใหม่ทุกรอบ (channel selection/covariance เท่านั้นที่ต่าง
    # กันต่อ combination ไม่ใช่ data loading)
    #
    # Load data/split once, reuse across every combination — no need to
    # re-scan disk each time (only channel selection/covariance differ
    # per combination, not data loading).
    base_cfg = Config(**{**OVERRIDES, "SAVE_PATH": OVERRIDES["SAVE_PATH"],
                         "OUTPUT_PATH": OVERRIDES["OUTPUT_PATH"]})
    set_seed(base_cfg.SEED)
    data = build_datasets_and_loaders(base_cfg)
    n_train = len(data["df_train"])
    logger.warning(f"Group ปัจจุบัน (จาก RUN.OVERRIDES): n_train (normal) = {n_train} ภาพ")

    rows = []
    for n_sel in NUM_SELECTED_CHANNELS_VALUES:
        for eps in COV_REG_EPSILON_VALUES:
            cfg = Config(**{**OVERRIDES,
                            "NUM_SELECTED_CHANNELS": n_sel,
                            "COV_REG_EPSILON": eps})
            print(f"  fit+score: NUM_SELECTED_CHANNELS={n_sel:>4}  "
                  f"COV_REG_EPSILON={eps:<5}  "
                  f"(n_train={n_train}, ratio={n_train/n_sel:.2f}x)")

            model = PaDiM(cfg)
            model.fit(data["normal_loader"])
            val_result = model.score(data["val_loader"])

            threshold = select_percentile_threshold(
                val_result.image_scores, val_result.y_true, cfg)
            metrics = compute_metrics(
                val_result.image_scores, val_result.y_true, threshold)

            rows.append(dict(
                num_selected_channels=n_sel,
                cov_reg_epsilon=eps,
                n_train=n_train,
                channels_to_train_ratio=n_sel / n_train,
                val_auc=metrics["auc"],
                val_escape_rate=metrics["escape_rate"],
                val_auto_clear_rate=metrics["auto_clear_rate"],
            ))

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    # สัญญาณ ill-conditioning แบบง่าย: val_auc แปรผันมากข้าม
    # COV_REG_EPSILON ที่ NUM_SELECTED_CHANNELS เดียวกัน (ทฤษฎีคือ auc ควร
    # เปลี่ยนน้อยถ้า covariance estimation เสถียร) — ไม่ใช่บทสรุปทางสถิติ
    # ที่เข้มงวด แค่ตัวช่วยอ่านตารางเร็วๆ ว่าค่าไหนน่าสงสัย
    #
    # Simple ill-conditioning signal: large val_auc swings across
    # COV_REG_EPSILON at the same NUM_SELECTED_CHANNELS (theory says auc
    # should barely move if covariance estimation is stable) — not a
    # rigorous statistical conclusion, just a quick way to flag rows
    # worth a closer look.
    print("\nช่วง (max-min) ของ val_auc ต่อ NUM_SELECTED_CHANNELS "
          "(ยิ่งกว้าง ยิ่งน่าสงสัยว่า ill-conditioned):")
    spread = (df.groupby("num_selected_channels")["val_auc"]
              .agg(lambda s: s.max() - s.min()))
    print(spread.to_string())

    out_path = Path(base_cfg.SAVE_PATH) / "channel_sensitivity_sweep.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
