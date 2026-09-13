"""
รัน PaDiM ซ้ำหลาย seed (multi-seed) — reuse OVERRIDES จาก single_run.py
ทุกประการ ไม่ copy ซ้ำ กัน 2 ไฟล์ไม่ sync กัน

แต่ละ seed รันครบ 3 step เหมือน single_run.py:
  [1/3] run_padim  — fit + score + save
  [2/3] visualize      — สร้างภาพ
  [3/3] cost_aware     — threshold sweep (toggle จาก single_run.RUN_COST_AWARE_ANALYSIS)

auto-detect "SEED {n}" ใน path ของ TEMPLATE_KEYS แล้วแทนที่ต่อ seed
fail-fast ถ้าไม่เจอ marker ก่อนเริ่ม seed แรก

Each seed runs all 3 steps identical to single_run.py:
  [1/3] run_padim  — fit + score + save
  [2/3] visualize      — generate all images
  [3/3] cost_aware     — threshold sweep (toggle from single_run.RUN_COST_AWARE_ANALYSIS)

Auto-detects "SEED {n}" in TEMPLATE_KEYS paths and substitutes per seed.
Fails fast if the marker is missing before the first seed starts.

── หมายเหตุ (แก้ไขระหว่าง repo-analysis สำหรับความน่าเชื่อถือของผลทดลอง) ──
เดิม TEMPLATE_KEYS รวม 'SPLIT_CACHE_PATH' ไว้ด้วย ทำให้ cfg.SEED คุมทั้ง
(1) train/val/test split (ผ่าน _stratified_group_split ใน dataset.py) และ
(2) random channel selection ของ PaDiM (ผ่าน torch.manual_seed ใน padim.py)
พร้อมกัน — แปลว่าแต่ละ seed จะได้ **test set คนละชุด** ไม่ใช่แค่โมเดล/
random channel ที่ต่างกันบน test set เดียวกัน ผลที่ได้คือ:
  - std ข้าม seed จะปนรวม (model variance + split variance) เข้าด้วยกัน
    ทำให้ตีความ "โมเดลเสถียรแค่ไหน" ไม่ได้ตรงๆ
  - test set ของทุก seed ยกเว้น 42 จะไม่ตรงกับ split_assignment.csv ของ
    repo หลัก (Anomaly-Detection-THESIS) อีกต่อไป → ผลเปรียบเทียบกับ
    EXPERIMENT 0 ไม่ valid สำหรับ seed เหล่านั้น

ค่า default ใหม่: แยก MODEL_SEEDS (คุม channel selection + naive
random_prior) ออกจาก SPLIT_CACHE_PATH ซึ่ง fix ไว้ค่าเดียวเสมอ (ตรงกับ
repo หลัก) — เป็นมาตรฐานทั่วไปของการรายงาน mean±std ข้าม seed: fix
evaluation set ไว้ วัดเฉพาะความแปรผันของโมเดล ถ้าต้องการวัด split
variance ด้วยจริงๆ (คนละคำถามวิจัย) ให้เปิด VARY_SPLIT_PER_SEED=True
ด้านล่างอย่างตั้งใจ พร้อมระบุในเล่มว่าเป็นการทดลองคนละแบบ ไม่ใช่ default

Fixed while reviewing the repo for experimental credibility: previously,
TEMPLATE_KEYS included 'SPLIT_CACHE_PATH', so a single cfg.SEED controlled
both (1) the train/val/test split and (2) PaDiM's random channel
selection at once — meaning every seed got a *different* test set, not
just a different model/channel-selection on the same test set. That
conflates model variance with split variance in the reported std, and
breaks comparability with EXPERIMENT 0 for every seed except 42 (whose
split happens to match the main repo's split_assignment.csv).

New default: SPLIT_CACHE_PATH is fixed across all seeds (matching the
main repo), only SAVE_PATH/OUTPUT_PATH vary per seed. This is the
standard way to report mean±std across seeds — fixed evaluation set,
model-only variance. Set VARY_SPLIT_PER_SEED=True below to opt into
measuring split variance too (a different research question — state
that explicitly in the thesis if used, don't treat it as the default).
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import single_run
from config.config import Config
import scripts.run_padim              as run_padim
import scripts.visualize_padim        as visualize_padim
import scripts.run_cost_aware_padim   as run_cost_aware_padim

# ── seed ที่จะรัน ────────────────────────────────────────────────────
SEEDS = [1, 14, 42, 99, 143]

# ── เปิดถ้าต้องการวัด split variance ด้วย (คนละคำถามวิจัยจาก default) ──
# False (ค่าเริ่มต้น): SPLIT_CACHE_PATH คงที่ทุก seed — วัดเฉพาะความ
# แปรผันของโมเดล (channel selection + naive random_prior), test set
# ตรงกับ repo หลักเสมอ เปรียบเทียบกับ EXPERIMENT 0 ได้ตรงๆ
# True: SPLIT_CACHE_PATH เปลี่ยนตาม seed ด้วย — วัด (model + split)
# variance ปนกัน ใช้เมื่อต้องการทดสอบความเสถียรของ split เอง ไม่ใช่
# เพื่อเทียบกับ EXPERIMENT 0
#
# False (default): SPLIT_CACHE_PATH stays fixed across seeds — measures
# model variance only (channel selection + naive random_prior); the test
# set always matches the main repo, so results compare directly against
# EXPERIMENT 0.
# True: SPLIT_CACHE_PATH also varies per seed — conflates model + split
# variance; use only when the research question is "how stable is the
# split itself", not for comparing against EXPERIMENT 0.
VARY_SPLIT_PER_SEED = False

# ── key ที่ต้องแยกตาม seed ──────────────────────────────────────────
# SAVE_PATH/OUTPUT_PATH แยกต่อ seed เสมอ (กันผลของ seed ต่างๆ เขียนทับกัน)
# SPLIT_CACHE_PATH แยกต่อ seed เฉพาะเมื่อ VARY_SPLIT_PER_SEED=True เท่านั้น
#
# SAVE_PATH/OUTPUT_PATH always vary per seed (so results don't overwrite
# each other). SPLIT_CACHE_PATH only varies per seed when
# VARY_SPLIT_PER_SEED=True.
TEMPLATE_KEYS = ['SAVE_PATH', 'OUTPUT_PATH']
if VARY_SPLIT_PER_SEED:
    TEMPLATE_KEYS.append('SPLIT_CACHE_PATH')
else:
    print("VARY_SPLIT_PER_SEED=False → SPLIT_CACHE_PATH คงที่ทุก seed "
          f"({single_run.OVERRIDES['SPLIT_CACHE_PATH']!r}) — วัดเฉพาะ model "
          "variance, test set ตรงกับ repo หลักเสมอ\n"
          "SPLIT_CACHE_PATH fixed across all seeds — measuring model "
          "variance only, test set always matches the main repo.")

# ── auto-detect template จาก OVERRIDES ─────────────────────────────
_current_seed = single_run.OVERRIDES['SEED']
_marker = f'SEED {_current_seed}'
_placeholder = 'SEED {seed}'

path_templates = {}
for key in TEMPLATE_KEYS:
    original_value = single_run.OVERRIDES[key]
    if _marker not in original_value:
        raise ValueError(
            f"ไม่เจอ '{_marker}' ใน single_run.OVERRIDES['{key}'] "
            f"(= {original_value!r}) — ต้องฝัง '{_marker}' ไว้ใน path "
            f"ให้ script แทนที่ด้วย seed อื่นได้\n"
            f"/ '{_marker}' not found in single_run.OVERRIDES['{key}'] "
            f"(= {original_value!r}) — embed '{_marker}' in the path "
            f"so this script can substitute other seeds.")
    path_templates[key] = original_value.replace(_marker, _placeholder)

print("Path templates ที่ตรวจพบ:")
for key, tmpl in path_templates.items():
    print(f"  {key} = {tmpl!r}")

results_log = []

for i, seed in enumerate(SEEDS, start=1):
    _n_steps = 3 if single_run.RUN_COST_AWARE_ANALYSIS else 2

    print(f"\n{'=' * 70}")
    print(f" MULTI-SEED RUN [{i}/{len(SEEDS)}] — SEED={seed}")
    print(f"{'=' * 70}")

    single_run.OVERRIDES['SEED'] = seed
    for key, tmpl in path_templates.items():
        single_run.OVERRIDES[key] = tmpl.format(seed=seed)

    print(f"  SPLIT_CACHE_PATH -> {single_run.OVERRIDES['SPLIT_CACHE_PATH']}")
    print(f"  SAVE_PATH        -> {single_run.OVERRIDES['SAVE_PATH']}")
    print(f"  OUTPUT_PATH      -> {single_run.OVERRIDES['OUTPUT_PATH']}")

    try:
        cfg = Config(**single_run.OVERRIDES)

        print(f"\n  --- [1/{_n_steps}] fit + score + save ---")
        run_padim.run(cfg)

        print(f"\n  --- [2/{_n_steps}] visualize ---")
        visualize_padim.visualize(cfg)

        if single_run.RUN_COST_AWARE_ANALYSIS:
            print(f"\n  --- [3/{_n_steps}] cost-aware sweep ---")
            run_cost_aware_padim.main()

        results_log.append((seed, 'OK', None))
        print(f"\n  ✅ seed={seed} เสร็จ -> {single_run.OVERRIDES['SAVE_PATH']}")

    except Exception as e:
        results_log.append((seed, 'FAILED', str(e)))
        print(f"\n  ❌ seed={seed} ล้มเหลว: {e}")
        traceback.print_exc()
        print("  ข้าม seed นี้ ไปทำ seed ถัดไปต่อ...")
        continue

# ── สรุปผล ─────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(" สรุปผล Multi-Seed Run")
print(f"{'=' * 70}")
for seed, status, err in results_log:
    line = f"  seed={seed:<4}  {status}"
    if err:
        line += f"  ({err})"
    print(line)

n_ok = sum(1 for _, s, _ in results_log if s == 'OK')
print(f"\nสำเร็จ {n_ok}/{len(SEEDS)} seed")
if n_ok < len(SEEDS):
    print("⚠️  เช็ค traceback ด้านบนก่อนเอาผลไปสรุปสถิติ")