"""สร้างเอกสาร README.md อัตโนมัติหลังรันเสร็จ อธิบายว่าแต่ละไฟล์ใน
Config.SAVE_PATH (ตัวเลข/log) และ Config.OUTPUT_PATH (ภาพ) เก็บอะไร
และเอาไปทำอะไรต่อได้บ้าง

dynamic — เช็คว่าไฟล์ไหนมีอยู่จริง ณ ตอนเรียกก่อนเขียนคำอธิบาย
ไม่ list ไฟล์ที่ยังไม่ถูกสร้าง

เรียกจาก:
  scripts/run_patchcore.py      → เขียน SAVE_PATH/README.md
  scripts/visualize_patchcore.py → เขียน OUTPUT_PATH/README.md

Generates README.md automatically after a run, documenting every file
in Config.SAVE_PATH (numeric/log) and Config.OUTPUT_PATH (images).

Dynamic — checks which files actually exist at call time.

Called from:
  scripts/run_patchcore.py       → writes SAVE_PATH/README.md
  scripts/visualize_patchcore.py → writes OUTPUT_PATH/README.md
"""
import glob
import os
from datetime import datetime
from pathlib import Path
from typing import List


def _size_str(path: str) -> str:
    if not os.path.exists(path):
        return "N/A"
    size = float(os.path.getsize(path))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _glob(base: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(base, pattern)))


# ══════════════════════════════════════════════════════════════════════
# SAVE_PATH — ตัวเลข/log
# ══════════════════════════════════════════════════════════════════════

def write_save_path_readme(cfg) -> str:
    p = cfg.SAVE_PATH
    os.makedirs(p, exist_ok=True)
    s = []

    s.append(f"""# Artifacts ใน `{p}` (SAVE_PATH) — {getattr(cfg, 'METHOD_NAME', 'Baseline')}

สร้างอัตโนมัติโดย `src/output_docs.py` เมื่อ {datetime.now().isoformat(timespec='seconds')}

โฟลเดอร์นี้เก็บ **ตัวเลข/log ทั้งหมด** — ไม่มีไฟล์ภาพ
(ภาพอยู่ที่ `{cfg.OUTPUT_PATH}`)

---
""")

    # ── ชื่อไฟล์ผูกกับ cfg.SEED เสมอ (ตั้งแต่เพิ่ม seed suffix กันไฟล์
    # ของแต่ละ seed ทับกัน) — สร้าง dict key ด้วย f-string แทนการ hardcode
    # ตรงๆ เพื่อให้ exists-check ด้านล่างหาไฟล์เจอ ไม่งั้น README จะว่างเปล่า
    # เพราะชื่อไฟล์จริงไม่ตรงกับ key ที่เช็ค
    #
    # Filenames always carry cfg.SEED (since the seed suffix was added to
    # prevent per-seed files from overwriting each other) — build dict keys
    # via f-string instead of hardcoding, or the exists-check below finds
    # nothing and README.md comes out empty.
    seed = cfg.SEED
    docs = {
        f"scores_val_seed{seed}.npz": (
            "7 key: `scores` (float32,[N]), `y_true` (int64,[N]), "
            "`labels` (string,[N]), `paths` (string,[N]), "
            "`heatmaps` (float32,[N,H,W]), `orig_imgs` (float32,[N,H,W,3]), "
            "`preproc_imgs` (float32,[N,H,W,3]) — schema ตรงกับ repo หลักเป๊ะ "
            "ใช้ `np.load(..., allow_pickle=True)` โหลด | "
            "7 keys matching main repo schema exactly"
        ),
        f"scores_test_seed{seed}.npz": f"เหมือน scores_val_seed{seed}.npz แต่สำหรับ test split / Same as scores_val_seed{seed}.npz but for test split",
        f"final_results_val_seed{seed}.json": (
            "config snapshot + metrics ครบชุด + naive_baselines (val) — "
            "confusion_matrix เป็น dict มี label tt/tf/ft/ff กำกับชัดเจน "
            "(+ raw array เดิม) เทียบ AUROC/escape_rate กับ repo หลักได้โดยตรง | "
            "config snapshot + full metrics + naive_baselines (val); "
            "confusion_matrix is a labeled dict (tt/tf/ft/ff + original raw array)"
        ),
        f"final_results_test_seed{seed}.json": f"เหมือน final_results_val_seed{seed}.json แต่สำหรับ test split / Same but for test split",
        f"roc_curve_data_val_seed{seed}.csv": (
            "fpr, tpr, threshold ทุกจุดบน ROC curve (val) — "
            "เปิดใน Excel ได้เลย หรือใช้เป็น input ของ multi-seed ROC aggregation | "
            "Raw ROC curve points for Excel or multi-seed vertical averaging"
        ),
        f"roc_curve_data_test_seed{seed}.csv": f"เหมือน roc_curve_data_val_seed{seed}.csv แต่สำหรับ test split / Same but for test split",
        f"cost_aware_sweep_seed{seed}.csv": (
            "29 columns: r, threshold, val_cost + metric ครบ 13 ตัว × val/test — "
            "มีก็ต่อเมื่อเคยรัน scripts/run_cost_aware_padim.py | "
            "29-column cost-aware sweep (exists only if run_cost_aware_padim.py was run)"
        ),
        f"gallery_index_seed{seed}.csv": (
            "index ของทุกภาพใน gallery (split, path, score, pred, correct) — "
            "มีก็ต่อเมื่อเคยรัน scripts/visualize_padim.py | "
            "Gallery image index (exists only if visualize_padim.py was run)"
        ),
        "README.md": "ไฟล์นี้ — auto-generated / This file — auto-generated",
    }

    for fname, desc in docs.items():
        fpath = os.path.join(p, fname)
        if os.path.exists(fpath):
            s.append(f"## `{fname}` ({_size_str(fpath)})\n{desc}\n\n---\n\n")

    out = os.path.join(p, "README.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(s))
    return out


# ══════════════════════════════════════════════════════════════════════
# OUTPUT_PATH — ภาพ .png
# ══════════════════════════════════════════════════════════════════════

def write_output_path_readme(cfg) -> str:
    p = cfg.OUTPUT_PATH
    os.makedirs(p, exist_ok=True)
    s = []

    s.append(f"""# ภาพผลลัพธ์ใน `{p}` (OUTPUT_PATH) — {getattr(cfg, 'METHOD_NAME', 'Baseline')}

สร้างอัตโนมัติโดย `src/output_docs.py` เมื่อ {datetime.now().isoformat(timespec='seconds')}

โฟลเดอร์นี้เก็บ **ไฟล์ภาพ (.png) ทั้งหมด** — ไม่มีไฟล์ตัวเลข
(ตัวเลขอยู่ที่ `{cfg.SAVE_PATH}`)

สร้างโดย `scripts/visualize_patchcore.py`

---
""")

    # ── ไฟล์ที่ไม่มี wildcard ต้องผูกกับ cfg.SEED ตรงๆ (เช่นเดียวกับ
    # write_save_path_readme ด้านบน) — ไฟล์ที่มี wildcard (heatmaps_*.png,
    # gallery_*_*.png) ไม่ต้องแก้ เพราะ "*" ครอบคลุม "..._seed{N}.png" อยู่แล้ว
    #
    # Non-wildcard entries must carry cfg.SEED explicitly (same reason as
    # write_save_path_readme above) — wildcard entries (heatmaps_*.png,
    # gallery_*_*.png) need no change since "*" already covers "..._seed{N}.png".
    seed = cfg.SEED
    image_docs = [
        (f"eda_class_distribution_seed{seed}.png",
         "จำนวนภาพต่อ class (normal/anomaly) แยก val/test — เช็ค class imbalance | "
         "Image count per class split by val/test"),
        (f"roc_curves_seed{seed}.png",
         "ROC curve (FPR vs TPR) val+test พร้อม AUC | ROC curves for val/test with AUC"),
        (f"pr_curves_seed{seed}.png",
         "Precision-Recall curve val+test | PR curves for val/test"),
        (f"confusion_matrices_seed{seed}.png",
         "Confusion matrix val+test ที่ deployment threshold | Confusion matrices at deployment threshold"),
        (f"score_distributions_seed{seed}.png",
         "Histogram ของ kNN anomaly score แยก normal/anomaly พร้อมเส้น threshold | "
         "kNN score histogram split by normal/anomaly with threshold line"),
        ("heatmaps_*.png",
         "ตัวอย่าง 20 ภาพ + kNN distance heatmap overlay แยก val/test | "
         "20 sample images + kNN distance heatmap overlay per split"),
        ("gallery_original_*.png",
         "Grid ภาพ RGB ต้นฉบับ | Grid of original RGB images"),
        ("gallery_processed_*.png",
         "Grid ภาพ + heatmap overlay | Grid of images with heatmap overlay"),
    ]

    for pattern, desc in image_docs:
        base_pat = pattern.split(" / ")[0]
        matches = _glob(p, base_pat) if "*" in base_pat else (
            [os.path.join(p, base_pat)] if os.path.exists(os.path.join(p, base_pat)) else [])
        if not matches:
            continue
        found = "\n".join(
            f"  - `{os.path.basename(m)}` ({_size_str(m)})" for m in matches)
        s.append(f"## `{pattern}`\n**พบไฟล์จริง**:\n{found}\n\n{desc}\n\n---\n\n")

    # gallery_* dynamic (browse_gallery output)
    already = set()
    for pat, _ in image_docs:
        for m in _glob(p, pat.split(" / ")[0]):
            already.add(os.path.basename(m))
    remaining = [m for m in _glob(p, "gallery_*.png")
                 if os.path.basename(m) not in already]
    if remaining:
        found = "\n".join(
            f"  - `{os.path.basename(m)}` ({_size_str(m)})" for m in remaining)
        s.append(
            f"## `gallery_{{split}}_{{label}}_{{pred_label}}.png`\n"
            f"**พบไฟล์จริง**:\n{found}\n\n"
            "Gallery 3 คอลัมน์ (ภาพต้นฉบับ | heatmap | overlay) กรองตาม group "
            "(เช่น gallery_test_defect_normal.png = escape case) | "
            "3-column gallery filtered by group (e.g. escape cases)\n\n---\n\n")

    out = os.path.join(p, "README.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(s))
    return out