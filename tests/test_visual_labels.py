"""Unit test สำหรับ src/visual.py::_method_label() — กัน regression ของ
บั๊กที่เจอจากภาพจริง (ผู้ใช้ส่ง gallery ของ PaDiM มาแต่หัวข้อภาพเขียนว่า
"ConvNeXt-Wide_resnet50_2 AE Gallery"): ไฟล์นี้ copy มาจาก repo หลัก
(EXPERIMENT 0 = ConvNeXt+Autoencoder) แล้ว hardcode label ไว้ตายตัว ไม่ได้
ใช้ cfg.METHOD_NAME ที่มีอยู่แล้ว ทำให้กราฟ/gallery ทุกใบของ PaDiM/
PatchCore ติดป้ายผิดว่าเป็นผลของ autoencoder

Unit test for src/visual.py::_method_label() — guards against the exact
bug found from a real screenshot (user sent a PaDiM gallery whose title
read "ConvNeXt-Wide_resnet50_2 AE Gallery"): this file is copied from the
main repo (EXPERIMENT 0 = ConvNeXt+Autoencoder) and hardcoded the label,
never using the already-present cfg.METHOD_NAME — so every PaDiM/
PatchCore plot/gallery was mislabeled as the autoencoder's output.

Run with: pytest tests/test_visual_labels.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visual import _method_label


class _PaDiMCfg:
    BACKBONE = "wide_resnet50_2"
    METHOD_NAME = "PaDiM"


class _PatchCoreCfg:
    BACKBONE = "resnet50"
    METHOD_NAME = "PatchCore"


class _MainRepoCfg:
    """จำลอง Config ของ repo หลัก (Anomaly-Detection-THESIS) ที่ไม่มี
    METHOD_NAME field เลย — ต้อง fallback กลับไปใช้ label เดิมเป๊ะ
    ไม่กระทบ behavior เดิมของ repo หลัก."""
    BACKBONE = "convnext_tiny"


def test_method_label_uses_method_name_when_present():
    assert _method_label(_PaDiMCfg()) == "PaDiM (wide_resnet50_2)"
    assert _method_label(_PatchCoreCfg()) == "PatchCore (resnet50)"


def test_method_label_never_says_ae_for_a_non_autoencoder_method():
    """กัน regression ของบั๊กที่เจอจริง: label ของ PaDiM/PatchCore ต้อง
    ไม่มีคำว่า 'AE' (autoencoder) โผล่มาเลย"""
    assert "AE" not in _method_label(_PaDiMCfg())
    assert "AE" not in _method_label(_PatchCoreCfg())


def test_method_label_falls_back_to_original_ae_label_for_main_repo():
    """Config ที่ไม่มี METHOD_NAME (เช่น repo หลักเอง) ต้องได้ label เดิม
    เป๊ะ ไม่เปลี่ยนพฤติกรรมของ EXPERIMENT 0"""
    assert _method_label(_MainRepoCfg()) == "ConvNeXt-Convnext_tiny AE"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
