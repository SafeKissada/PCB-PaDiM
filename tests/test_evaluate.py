"""Unit test เชิงตัวเลขสำหรับ src/evaluate.py — ต่างจาก tests/smoke_test.py
(เช็คแค่ว่า pipeline รันจบไม่ error) ไฟล์นี้ป้อนตัวเลขที่รู้คำตอบล่วงหน้า
แล้วเช็คว่าสูตร metric ตรงตามนิยามจริง — เป็นระดับการทดสอบที่ขาดหายไป
ตอนที่เจอบั๊ก confusion-matrix mismatch ในอดีต (ดู
investigation_confusion_matrix_mismatch_OPEN.md ในเอกสารโครงการ) กัน
regression แบบเดียวกันเกิดซ้ำในไฟล์นี้

Numeric unit tests for src/evaluate.py — unlike tests/smoke_test.py
(only checks the pipeline runs without error), this file feeds in
ground-truth-known numbers and checks the metric formulas against their
actual definitions. This is exactly the kind of test that was missing
when the earlier confusion-matrix mismatch bug was found (see the
project's investigation_confusion_matrix_mismatch_OPEN.md) — guards
against the same class of regression here.

Run with: pytest tests/test_evaluate.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import (compute_metrics, compute_metrics_from_predictions,
                          compute_naive_baseline_metrics,
                          select_percentile_threshold,
                          oracle_threshold_diagnostic)


# ── ตัวอย่างที่รู้คำตอบล่วงหน้า ────────────────────────────────────
# 10 ตัวอย่าง: y_true (0=normal,1=anomaly), score, threshold=0.5
#   -> pred = score >= 0.5
# คำนวณด้วยมือ:
#   idx  y_true  score  pred  ผล
#    0     0     0.10    0    TN
#    1     0     0.20    0    TN
#    2     0     0.30    0    TN
#    3     0     0.60    1    FP
#    4     0     0.70    1    FP
#    5     1     0.15    0    FN
#    6     1     0.80    1    TP
#    7     1     0.85    1    TP
#    8     1     0.90    1    TP
#    9     1     0.95    1    TP
# -> TN=3, FP=2, FN=1, TP=4  (n=10)
Y_TRUE = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
SCORES = np.array([0.10, 0.20, 0.30, 0.60, 0.70,
                   0.15, 0.80, 0.85, 0.90, 0.95])
THRESHOLD = 0.5
EXPECTED_TN, EXPECTED_FP, EXPECTED_FN, EXPECTED_TP = 3, 2, 1, 4


def test_compute_metrics_confusion_matrix_matches_hand_computed():
    """cm.ravel() ต้องคืน [TN, FP, FN, TP] ตรงกับที่คำนวณด้วยมือ —
    เป็น sklearn convention มาตรฐาน (labels=[0,1]) ที่ src/evaluate.py
    และ src/cost_aware.py ต้องตีความตรงกันทั้งคู่ (ดู tt/tf/ft/ff mapping
    ใน cost_aware.py::_tt_tf_ft_ff)."""
    metrics = compute_metrics(SCORES, Y_TRUE, THRESHOLD)
    tn, fp, fn, tp = metrics["cm"].ravel()
    assert (tn, fp, fn, tp) == (EXPECTED_TN, EXPECTED_FP, EXPECTED_FN, EXPECTED_TP)


def test_compute_metrics_derived_rates():
    """escape_rate = FN/(FN+TP), auto_clear_rate = TN/n, residual_fcr = FP/(FP+TP)."""
    metrics = compute_metrics(SCORES, Y_TRUE, THRESHOLD)
    n = len(Y_TRUE)
    assert metrics["escape_rate"] == pytest.approx(
        EXPECTED_FN / (EXPECTED_FN + EXPECTED_TP))
    assert metrics["auto_clear_rate"] == pytest.approx(EXPECTED_TN / n)
    assert metrics["residual_fcr"] == pytest.approx(
        EXPECTED_FP / (EXPECTED_FP + EXPECTED_TP))
    assert metrics["acc"] == pytest.approx(
        (EXPECTED_TN + EXPECTED_TP) / n)


def test_compute_metrics_from_predictions_tt_tf_ft_ff_convention():
    """tt/tf/ft/ff = (actual, predicted): tt=(anomaly,anomaly)=TP,
    tf=(anomaly,normal)=FN, ft=(normal,anomaly)=FP, ff=(normal,normal)=TN
    — ตรงตามนิยามที่ยืนยันไว้ใน investigation_confusion_matrix_mismatch_OPEN.md
    ห้ามสลับ ไม่ว่าจะแก้โค้ดส่วนไหนในอนาคต."""
    pred = (SCORES >= THRESHOLD).astype(int)
    m = compute_metrics_from_predictions(Y_TRUE, pred, scores=SCORES)
    assert m["tt"] == EXPECTED_TP
    assert m["tf"] == EXPECTED_FN
    assert m["ft"] == EXPECTED_FP
    assert m["ff"] == EXPECTED_TN


def test_compute_metrics_and_compute_metrics_from_predictions_agree():
    """สองฟังก์ชันคำนวณ confusion-matrix-derived metric คนละทาง (ผ่าน
    sklearn.confusion_matrix vs ผ่าน boolean mask ตรงๆ) — ต้องได้ผล
    เหมือนกันทุกตัวบนอินพุตเดียวกัน กันสองสูตรแยกออกจากกันแบบเงียบๆ."""
    pred = (SCORES >= THRESHOLD).astype(int)
    a = compute_metrics(SCORES, Y_TRUE, THRESHOLD)
    b = compute_metrics_from_predictions(Y_TRUE, pred, scores=SCORES)
    assert a["escape_rate"] == pytest.approx(b["escape_rate"])
    assert a["auto_clear_rate"] == pytest.approx(b["auto_clear_rate"])
    assert a["acc"] == pytest.approx(b["acc"])
    assert a["f1"] == pytest.approx(b["f1"])


def test_select_percentile_threshold_uses_only_normal_scores():
    """threshold ต้องมาจาก percentile ของ normal score ใน val เท่านั้น
    (ไม่ใช้ anomaly score เลย) — เปลี่ยนค่า anomaly score โดยไม่แตะ
    normal score ต้องไม่กระทบ threshold ที่ได้"""
    class _Cfg:
        THRESHOLD_PERCENTILE = 95.0

    threshold_a = select_percentile_threshold(SCORES, Y_TRUE, _Cfg())
    scores_perturbed = SCORES.copy()
    scores_perturbed[Y_TRUE == 1] = 999.0  # เปลี่ยนเฉพาะ anomaly score
    threshold_b = select_percentile_threshold(scores_perturbed, Y_TRUE, _Cfg())
    assert threshold_a == pytest.approx(threshold_b)


def test_oracle_threshold_diagnostic_returns_perfect_separator_when_separable():
    """ถ้า score แยกคลาสได้สมบูรณ์แบบ oracle threshold ต้องให้ f1=1.0"""
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])  # แยกขาดที่ 0.5
    _thr, f1 = oracle_threshold_diagnostic(scores, y)
    assert f1 == pytest.approx(1.0)


def test_compute_naive_baseline_metrics_reproducible_with_seed():
    """random_prior ต้อง reproduce ได้ทุกครั้งด้วย seed เดียวกัน (ใช้
    RandomState แยกต่างหาก ไม่แตะ global RNG) — สำคัญเพราะผลลัพธ์นี้ถูก
    เซฟลง final_results.json และต้อง reproduce ได้ข้าม run"""
    # หมายเหตุ: เทียบ dict ตรงๆ ด้วย == จะ fail เสมอเพราะ auc/ap เป็น
    # float('nan') และ nan != nan ตาม IEEE 754 — ใช้
    # np.testing.assert_equal ที่ถือว่า nan == nan แทน (ไม่ใช่บั๊กของ
    # evaluate.py ที่กำลังทดสอบ)
    #
    # Note: comparing the dicts directly with == always fails because
    # auc/ap are float('nan') and nan != nan per IEEE 754 — use
    # np.testing.assert_equal, which treats nan == nan, instead (not a
    # bug in the evaluate.py code under test).
    y_true = np.array([0] * 70 + [1] * 30)
    result_a = compute_naive_baseline_metrics(y_true, seed=42)
    result_b = compute_naive_baseline_metrics(y_true, seed=42)
    np.testing.assert_equal(result_a["random_prior"], result_b["random_prior"])

    result_c = compute_naive_baseline_metrics(y_true, seed=99)
    # ไม่ assert ว่าต้องต่างกันเสมอ (อาจบังเอิญเหมือน) แค่เช็คว่า field
    # ที่ deterministic (always_normal/always_anomaly) ไม่ขึ้นกับ seed
    # (nan-aware เหมือนด้านบน — residual_fcr เป็น nan ได้ตามนิยาม)
    np.testing.assert_equal(result_a["always_normal"], result_c["always_normal"])
    np.testing.assert_equal(result_a["always_anomaly"], result_c["always_anomaly"])


def test_compute_naive_baseline_metrics_always_normal_and_always_anomaly():
    """always_normal: ff=n(normal), tt=0 เสมอ. always_anomaly: tt=n(anomaly), ff=0 เสมอ."""
    y_true = np.array([0, 0, 0, 1, 1])
    result = compute_naive_baseline_metrics(y_true, seed=0)

    always_normal = result["always_normal"]
    assert always_normal["tt"] == 0 and always_normal["ft"] == 0
    assert always_normal["ff"] == 3 and always_normal["tf"] == 2

    always_anomaly = result["always_anomaly"]
    assert always_anomaly["ff"] == 0 and always_anomaly["tf"] == 0
    assert always_anomaly["tt"] == 2 and always_anomaly["ft"] == 3


def test_save_final_results_confusion_matrix_is_labeled_and_matches_raw():
    """confusion_matrix ใน final_results.json ต้องเป็น dict มี tt/tf/ft/ff
    label ชัดเจน ตรงกับ raw array เดิม (sklearn [[tn,fp],[fn,tp]]) — กัน
    การตีความผิดแบบเดียวกับที่เคยเจอมาก่อน (ดู
    investigation_confusion_matrix_mismatch_OPEN.md)"""
    import json
    import tempfile
    from src.io_utils import save_final_results
    from src.evaluate import compute_metrics

    class _Cfg:
        EXPERIMENT = "test"
        BACKBONE = "resnet18"
        METHOD_NAME = "PaDiM"
        THRESHOLD_PERCENTILE = 95.0
        SEED = 42  # save_final_results() ตอนนี้ใส่ seed suffix ในชื่อไฟล์ ต้องมี field นี้

    with tempfile.TemporaryDirectory() as tmp:
        _Cfg.SAVE_PATH = tmp
        metrics = compute_metrics(SCORES, Y_TRUE, THRESHOLD)
        out_path = save_final_results(_Cfg(), "val", metrics, THRESHOLD)
        saved = json.loads(out_path.read_text(encoding="utf-8"))

        cm = saved["confusion_matrix"]
        assert cm["tt"] == EXPECTED_TP
        assert cm["tf"] == EXPECTED_FN
        assert cm["ft"] == EXPECTED_FP
        assert cm["ff"] == EXPECTED_TN
        # raw array ต้องตรงกับ label ที่ให้ไว้ (tn=ff, fp=ft, fn=tf, tp=tt)
        raw = cm["raw"]
        assert raw == [[EXPECTED_TN, EXPECTED_FP], [EXPECTED_FN, EXPECTED_TP]]
        assert "convention" in cm  # ต้องมีคำอธิบาย convention แนบมาด้วยเสมอ


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
