"""Unit test เชิงตัวเลขสำหรับ src/cost_aware.py — ครอบคลุมจุดที่คอมเมนต์
ในไฟล์เตือนไว้เองว่าเสี่ยง (cm -> tt/tf/ft/ff conversion ที่ cost_aware.py
ทำเองเพราะ evaluate.py ต้นแบบยังเป็นเวอร์ชันเก่า) และ threshold selection
บน val เท่านั้น

Numeric unit tests for src/cost_aware.py — covering the exact risk the
file's own comments flag (the cm -> tt/tf/ft/ff conversion cost_aware.py
does itself because the template evaluate.py is still the older version)
and val-only threshold selection.

Run with: pytest tests/test_cost_aware.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cost_aware import (cost_at_threshold, sweep_thresholds,
                            select_cost_optimal_threshold, cost_sweep_report,
                            find_elbow_r, select_recall_constrained_threshold)


# เหมือนชุดตัวอย่างใน test_evaluate.py: TN=3, FP=2, FN=1, TP=4 ที่ threshold=0.5
Y_TRUE = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
SCORES = np.array([0.10, 0.20, 0.30, 0.60, 0.70,
                   0.15, 0.80, 0.85, 0.90, 0.95])


def test_cost_at_threshold_matches_hand_computed_confusion_matrix():
    result = cost_at_threshold(SCORES, Y_TRUE, threshold=0.5, r=10.0)
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (3, 2, 1, 4)
    # total_cost = r*FN + FP = 10*1 + 2 = 12
    assert result["total_cost"] == pytest.approx(12.0)


def test_cost_at_threshold_extreme_thresholds_use_fixed_labels():
    """threshold สูงมากจน pred เป็น 0 ทั้งหมด (มี class เดียว) — ต้องไม่
    ทำให้ confusion_matrix คืน shape ผิด (นี่คือเหตุผลที่โค้ดล็อก
    labels=[0,1] เอาไว้เสมอ ตามคอมเมนต์ในไฟล์)"""
    result = cost_at_threshold(SCORES, Y_TRUE, threshold=999.0, r=10.0)
    assert result["tp"] == 0 and result["fp"] == 0
    assert result["fn"] == int((Y_TRUE == 1).sum())
    assert result["tn"] == int((Y_TRUE == 0).sum())


def test_select_cost_optimal_threshold_high_r_favors_fewer_false_negatives():
    """r สูงมาก (escape แพงกว่า false alarm มาก) -> threshold ที่เลือก
    ต้อง flag เยอะขึ้น (threshold ต่ำลง) เทียบกับ r ต่ำ"""
    best_low_r  = select_cost_optimal_threshold(SCORES, Y_TRUE, r=1.0)
    best_high_r = select_cost_optimal_threshold(SCORES, Y_TRUE, r=1000.0)
    assert best_high_r["fn"] <= best_low_r["fn"]
    assert best_high_r["threshold"] <= best_low_r["threshold"]


def test_select_cost_optimal_threshold_only_uses_val_not_test():
    """select_cost_optimal_threshold รับแค่ scores/y_true ชุดเดียว (ไม่มี
    พารามิเตอร์ test แยก) — เป็น structural guarantee ว่าการเลือก
    threshold ทำบน val อย่างเดียวเสมอ ไม่มีทางเผลอส่ง test เข้าไปตัดสินใจ"""
    import inspect
    sig = inspect.signature(select_cost_optimal_threshold)
    assert list(sig.parameters) == ["val_scores", "val_y", "r"]


def test_cost_sweep_report_tt_tf_ft_ff_matches_evaluate_convention():
    """tt/tf/ft/ff ที่ cost_aware.py แปลงเองจาก cm ต้องตรงกับ convention
    (actual, predicted) เดียวกับ src/evaluate.py::compute_metrics_from_predictions
    — tt=TP, tf=FN, ft=FP, ff=TN เท่านั้น ห้ามสลับ"""
    report = cost_sweep_report(SCORES, Y_TRUE, SCORES, Y_TRUE, r_values=[1.0])
    row = report[0]
    for prefix in ("val_", "test_"):
        tt, tf, ft, ff = (row[f"{prefix}tt"], row[f"{prefix}tf"],
                          row[f"{prefix}ft"], row[f"{prefix}ff"])
        assert tt + tf == int((Y_TRUE == 1).sum()), "tt+tf ต้องรวมเป็นจำนวน anomaly ทั้งหมด"
        assert ft + ff == int((Y_TRUE == 0).sum()), "ft+ff ต้องรวมเป็นจำนวน normal ทั้งหมด"


def test_find_elbow_r_requires_at_least_three_rows():
    with pytest.raises(ValueError):
        find_elbow_r([{"val_tf": 1, "val_ft": 1}, {"val_tf": 2, "val_ft": 2}])


def test_find_elbow_r_picks_middle_when_curve_is_flat():
    """ถ้าทุกแถวมี fn/fp เหมือนกันหมด (ไม่มี "โค้ง" ให้วัด) ต้อง fallback
    ไปจุดกึ่งกลาง ไม่ error และไม่เลือกแบบสุ่ม"""
    flat_report = [{"val_tf": 5, "val_ft": 5, "r": r} for r in [1, 10, 100]]
    result = find_elbow_r(flat_report)
    assert result["r"] == flat_report[1]["r"]  # จุดกึ่งกลางของ list ความยาว 3


def test_select_recall_constrained_threshold_raises_when_infeasible():
    """ตั้ง max_escape_rate เข้มกว่าที่ข้อมูลรองรับได้ (0.0 พอดี แต่มี FN
    เกิดขึ้นได้ที่ทุก threshold ในตัวอย่างนี้ยกเว้นที่ทำให้ TP=0 ด้วย)
    ต้อง raise ValueError ไม่ใช่คืนค่าอะไรเงียบๆ"""
    y = np.array([1, 1])  # ไม่มี normal เลย -> escape_rate ไม่มีทาง <= -1
    scores = np.array([0.1, 0.9])
    with pytest.raises(ValueError):
        select_recall_constrained_threshold(scores, y, max_escape_rate=-1.0)


def test_select_recall_constrained_threshold_feasible_case():
    result = select_recall_constrained_threshold(SCORES, Y_TRUE, max_escape_rate=0.5)
    assert result["escape_rate"] <= 0.5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
