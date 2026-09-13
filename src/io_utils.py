"""I/O utilities สำหรับ PatchCore baseline — ออกแบบให้ schema ของ output
ตรงกับ repo หลัก (Anomaly-Detection-THESIS) ทุก key เพื่อให้ script เทียบ
ผลข้าม repo ใช้ชื่อไฟล์และ key เดียวกันได้โดยไม่มี silent mismatch

การแบ่ง SAVE_PATH / OUTPUT_PATH:
  SAVE_PATH   — ตัวเลข/log: .npz, .json, .csv  (โหลดกลับคำนวณต่อได้)
  OUTPUT_PATH — ภาพ: .png เท่านั้น             (ดูด้วยตาเท่านั้น)

I/O utilities for the PatchCore baseline — designed so that output schemas
match the main repo (Anomaly-Detection-THESIS) key-for-key, letting
cross-repo comparison scripts use the same filenames and key names without
any silent mismatch.

SAVE_PATH / OUTPUT_PATH split:
  SAVE_PATH   — numeric/log: .npz, .json, .csv  (reloadable for further computation)
  OUTPUT_PATH — images: .png only               (for visual inspection only)
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve


# ── SAVE_PATH artifacts (ตัวเลข/log) ─────────────────────────────────────

def save_final_results(cfg, split_name: str, metrics: dict,
                        threshold: float,
                        naive_baselines: dict = None) -> Path:
    """เซฟ final_results_{split_name}_seed{N}.json ลง SAVE_PATH — รูปแบบเดียวกับ
    repo หลัก (config snapshot + metrics + naive_baselines) เพื่อให้ script
    เทียบผลข้าม repo โหลด key เดียวกันได้โดยไม่ต้องแก้ไขอะไรเพิ่ม

    "confusion_matrix" เป็น dict มี label กำกับชัดเจน (tt/tf/ft/ff ตาม
    convention เดียวกับ src/cost_aware.py) พร้อม "raw" array เดิมไว้ด้วย
    เพื่อ backward-compat — กันตีความสับสนแบบเดียวกับที่เคยเจอมาก่อน

    naive_baselines: dict จาก compute_naive_baseline_metrics() —
    เก็บ 3 baseline (always_normal, always_anomaly, random_prior) พร้อม
    seed เพื่อให้ผล random_prior reproduce ได้ข้าม run

    final_results_{split_name}_seed{N}.json goes to SAVE_PATH — same format as the
    main repo (config snapshot + metrics + naive_baselines) so cross-repo
    comparison scripts load the same keys without modification.

    "confusion_matrix" is a labeled dict (tt/tf/ft/ff, same convention as
    src/cost_aware.py) plus the original "raw" array for backward
    compatibility — prevents the same misreading risk found before.
    """
    # ⚠️ 2026-09 SYNC: src/evaluate.py sync กับ main แล้ว (คืน tt/tf/ft/ff
    # ตรงๆ ใน metrics dict) จึงอ่านค่าจาก metrics ตรงๆ แทนการคำนวณจาก
    # cm.ravel() เองอีกรอบ (ปิดความเสี่ยง 3rd independent computation ที่
    # อาจ drift ถ้า evaluate.py แก้สูตรทีหลัง — cm ยังเก็บไว้ใน output
    # เหมือนเดิมเพื่อ backward-compat)
    #
    # SYNC NOTE: src/evaluate.py is now synced with main (returns
    # tt/tf/ft/ff directly in the metrics dict), so this reads those
    # values directly instead of re-deriving them from cm.ravel() (closes
    # the risk of a 3rd independent computation drifting apart from
    # evaluate.py's formula — cm itself is still kept in the output for
    # backward compatibility).
    cm = metrics["cm"]
    tt, tf, ft, ff = metrics["tt"], metrics["tf"], metrics["ft"], metrics["ff"]
    out = {
        "experiment"           : cfg.EXPERIMENT,
        "backbone"             : cfg.BACKBONE,
        "method"               : getattr(cfg, "METHOD_NAME", "Unknown"),
        "split"                : split_name,
        "threshold"            : threshold,
        "threshold_percentile" : cfg.THRESHOLD_PERCENTILE,
        "coreset_ratio"        : getattr(cfg, "CORESET_RATIO", None),  # None for non-PatchCore methods
        # กรอง key ที่เป็น array ขนาดใหญ่ออก เพราะ JSON ไม่รองรับ ndarray
        # และ key เหล่านี้อยู่ใน scores_{split}_seed{N}.npz แล้ว
        # Filter out large-array keys (JSON doesn't support ndarray;
        # those already live in scores_{split}_seed{N}.npz).
        "metrics" : {k: (v.tolist() if hasattr(v, 'tolist') else v)
                     for k, v in metrics.items()
                     if k not in ("cm", "fpr", "tpr", "gt", "pred", "scores")},
        # แก้ระหว่าง repo-analysis: เดิมมีแค่ "confusion_matrix": [[tn,fp],
        # [fn,tp]] เป็น array ดิบ ไม่มี label กำกับเลยว่าตัวไหนคือตัวไหน —
        # เสี่ยงตีความผิดพอๆ กับบั๊ก confusion-matrix mismatch ที่เคยเจอมา
        # ก่อนหน้านี้ (ดู investigation_confusion_matrix_mismatch_OPEN.md)
        # เพิ่ม tt/tf/ft/ff แบบมี label ชัดเจน ใช้ convention เดียวกับที่
        # src/cost_aware.py ใช้อยู่แล้ว (ยืนยันด้วย unit test ใน
        # tests/test_evaluate.py, tests/test_cost_aware.py):
        #   tt = (actual=anomaly, pred=anomaly) = True Positive
        #   tf = (actual=anomaly, pred=normal)  = False Negative (หลุด/escape)
        #   ft = (actual=normal,  pred=anomaly) = False Positive (false alarm)
        #   ff = (actual=normal,  pred=normal)  = True Negative
        # เก็บ "raw" array ไว้ด้วยเพื่อ backward-compat กับ script ที่โหลด
        # key เดิม ไม่ต้องแก้อะไรที่ฝั่งอ่านไฟล์
        #
        # Fixed while reviewing the repo: previously only had
        # "confusion_matrix": [[tn,fp],[fn,tp]] as a raw, unlabeled array —
        # just as easy to misread as the earlier confusion-matrix mismatch
        # bug (see investigation_confusion_matrix_mismatch_OPEN.md). Added
        # explicit labeled tt/tf/ft/ff using the same convention
        # src/cost_aware.py already uses (verified by unit tests in
        # tests/test_evaluate.py, tests/test_cost_aware.py):
        #   tt = (actual=anomaly, pred=anomaly) = True Positive
        #   tf = (actual=anomaly, pred=normal)  = False Negative (escape)
        #   ft = (actual=normal,  pred=anomaly) = False Positive (false alarm)
        #   ff = (actual=normal,  pred=normal)  = True Negative
        # Kept "raw" for backward compatibility with any script already
        # reading the old array key — no reader-side changes needed.
        "confusion_matrix" : {
            "convention": ("tt=(actual=anomaly,pred=anomaly)=TP | "
                          "tf=(actual=anomaly,pred=normal)=FN(escape) | "
                          "ft=(actual=normal,pred=anomaly)=FP(false_alarm) | "
                          "ff=(actual=normal,pred=normal)=TN"),
            "tt": tt, "tf": tf, "ft": ft, "ff": ff,
            "raw": cm.tolist(),
            "raw_order": "[[tn, fp], [fn, tp]]  (sklearn confusion_matrix(labels=[0,1]) layout)",
        },
    }
    if naive_baselines is not None:
        out["naive_baselines"] = naive_baselines
    out_path = Path(cfg.SAVE_PATH) / f"final_results_{split_name}_seed{cfg.SEED}.json"
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def save_scores(cfg, split_name: str,
                scores: np.ndarray, y_true: np.ndarray,
                labels: list, paths: list,
                heatmaps: np.ndarray,
                orig_imgs: np.ndarray,
                preproc_imgs: np.ndarray) -> Path:
    """เซฟ scores_{split_name}_seed{N}.npz ลง SAVE_PATH ด้วย schema เดียวกับ repo
    หลัก (7 key) — visualize.py และ script เทียบผลข้าม repo โหลด key
    เดียวกันได้ทันทีโดยไม่ต้องแก้ไขอะไร

    key ที่บันทึก:
      scores       float32 [N]        : anomaly score ต่อภาพ
      y_true       int64   [N]        : 0=normal 1=anomaly (ใช้คำนวณ metric)
      labels       string  [N]        : "good"/"defect" (ใช้ display เท่านั้น)
      paths        string  [N]        : path ต้นฉบับ
      heatmaps     float32 [N,H,W]   : kNN distance map หลัง upsample+smooth
      orig_imgs    float32 [N,H,W,3] : RGB ต้นฉบับก่อน normalize
      preproc_imgs float32 [N,H,W,3] : ภาพหลัง preprocessing จริง

    Saves scores_{split_name}_seed{N}.npz to SAVE_PATH with the same 7-key schema
    as the main repo — visualize.py and cross-repo scripts load the same
    keys immediately without modification.
    """
    out_path = Path(cfg.SAVE_PATH) / f"scores_{split_name}_seed{cfg.SEED}.npz"
    np.savez_compressed(
        out_path,
        scores       = scores.astype(np.float32),
        y_true       = y_true.astype(np.int64),
        labels       = np.array(labels),
        paths        = np.array(paths),
        heatmaps     = heatmaps.astype(np.float32),
        orig_imgs    = orig_imgs.astype(np.float32),
        preproc_imgs = preproc_imgs.astype(np.float32),
    )
    return out_path


def save_roc_csv(cfg, split_name: str,
                  scores: np.ndarray, y_true: np.ndarray) -> Path:
    """เซฟ roc_curve_data_{split_name}_seed{N}.csv ลง SAVE_PATH — จุดดิบทุกจุดบน
    ROC curve (fpr, tpr, threshold) สำหรับ:
      - เปิดดูใน Excel โดยไม่ต้องรัน Python เพิ่ม
      - multi-seed ROC aggregation (vertical averaging) — โหลด fpr/tpr
        ของแต่ละ seed มา np.interp() เข้าแกน FPR ร่วมก่อน average

    Save roc_curve_data_{split_name}_seed{N}.csv to SAVE_PATH — raw ROC curve
    points (fpr, tpr, threshold) for:
      - Direct viewing in Excel without any extra Python
      - Multi-seed ROC aggregation (vertical averaging): load each seed's
        fpr/tpr and np.interp() onto a shared FPR axis before averaging
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    out_path = Path(cfg.SAVE_PATH) / f"roc_curve_data_{split_name}_seed{cfg.SEED}.csv"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("fpr,tpr,threshold\n")
        for a, b, c in zip(fpr, tpr, thr):
            f.write(f"{a},{b},{c}\n")
    return out_path


# หมายเหตุ (ลบระหว่าง repo-analysis สำหรับความน่าเชื่อถือของผลทดลอง):
# เดิมไฟล์นี้มีฟังก์ชัน write_save_path_readme() อีกตัว (hardcode หัวข้อ
# "PatchCore" ตายตัว, ไม่ dynamic ตาม cfg.METHOD_NAME, ไม่ list
# cost_aware_sweep.csv/gallery_index.csv) — ไม่เคยถูกเรียกที่ไหนเลย
# (scripts/run_padim.py import จาก src/output_docs.py แทนเสมอ ดู
# คอมเมนต์ในไฟล์นั้น) ลบทิ้งเพราะเป็น dead code ที่เสี่ยงทำให้ README
# ระบุ method ผิด ถ้ามีใครแก้ import กลับมาใช้ตัวนี้โดยไม่ตั้งใจในอนาคต
# ใช้ src.output_docs.write_save_path_readme() แทนเสมอ
#
# Note (removed while reviewing the repo for experimental credibility):
# this file used to have a second write_save_path_readme() (hardcoded
# "PatchCore" title, not dynamic on cfg.METHOD_NAME, didn't list
# cost_aware_sweep.csv/gallery_index.csv) that was never called anywhere
# (scripts/run_padim.py always imports the one in src/output_docs.py
# instead — see that file's docstring). Removed as dead code that risked
# mislabeling the method in README.md if some future import accidentally
# pointed here. Always use src.output_docs.write_save_path_readme().