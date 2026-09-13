# คู่มือ: เพิ่ม Baseline ใหม่เข้า `mvtec_repro` (MVTec Reproduction Harness)

> ใช้ไฟล์นี้ทุกครั้งที่จะเพิ่มวิธีใหม่ (paper ใหม่ที่ benchmark บน MVTec AD)
> เข้าไปในชุด baseline ของเล่ม — เขียนจากบทเรียนจริงที่เจอตอนทำ PatchCore /
> PaDiM / RD4AD / SimpleNet (บั๊ก/ความเข้าใจผิดที่เจอจริงถูกอ้างอิงไว้เป็น
> ตัวอย่างตลอดทั้งไฟล์ ไม่ใช่สมมติขึ้นมา)

---

## 0. หลักการที่ต้องยึดตลอด (สรุปจากทุกอย่างที่ผ่านมา)

1. **โค้ดที่ validate ต้องเป็นไฟล์เดียวกับที่ใช้ใน production pipeline**
   ห้ามเขียน `src/models/<method>.py` แยกอีกชุดสำหรับ MVTec — ต้อง `import`
   ตรงจากไฟล์เดียวกับที่ `scripts/run_<method>.py` ใช้ ไม่งั้นผล validate
   จะพิสูจน์แค่ว่า "โค้ดอีกชุดที่หน้าตาคล้ายกัน" ถูก ไม่ใช่ตัวที่ใช้จริง
2. **Reference table ต้องมี citation ระดับ table/column และผ่าน unit test**
   ทุกค่าที่ใส่ต้องตรวจสอบย้อนกลับได้ว่ามาจากไหน (ดูข้อ 3.4)
3. **Protocol ที่ใช้ validate (ตาม paper) กับ config ที่ใช้จริงกับ PCB
   (production) มักไม่เหมือนกัน — ต้องเช็คช่องว่างนี้เสมอ ไม่ใช่สมมติว่า
   เหมือนกันเพราะ import ไฟล์เดียวกัน** (ดูกรณี RD4AD/SimpleNet ที่ paper
   ใช้ 200/160 epoch แต่ production `config.py` ตั้งไว้แค่ 100 — คนละเรื่อง
   กับ "algorithm ถูกไหม" แต่กระทบว่าตัวเลขที่ validate ไว้สะท้อนของจริง
   แค่ไหน)
4. **1 การรันไม่พอสรุปอะไรทั้งนั้น** ต้อง ≥3 seed เสมอ (ยิ่ง method มี
   training loop ยิ่งสำคัญ)
5. **Fail loud ดีกว่า fail เงียบ** — ถ้า mask หาไม่เจอ/geometry ไม่ตรง/
   pixel map ถูก normalize ผิดจุด ต้อง raise error ทันที ไม่ fallback
   เงียบๆ แล้วให้ตัวเลขที่ดูดีเกินจริงหลุดออกไป

---

## 1. Checklist ก่อนเริ่ม (ตอบให้ครบก่อนเขียนโค้ดบรรทัดแรก)

| # | คำถาม | ทำไมสำคัญ |
|---|---|---|
| 1 | Algorithm นี้มี training loop ไหม (มี gradient descent จริงหรือเป็น training-free เช่น memory-bank/สถิติปิด) | กำหนดว่าต้องกังวลเรื่อง seed variance/epoch มากแค่ไหน (ดูหมวด 3) |
| 2 | Backbone อะไร, feature layer ไหน, pretrained-weight version ไหน (V1/V2/อื่นๆ) ตาม paper | ต้องตรงกับตอน paper ตีพิมพ์ ไม่ใช่ default ล่าสุดของ torchvision |
| 3 | Preprocessing ตาม paper คือ resize เฉยๆ หรือ resize→center-crop ที่ resolution เท่าไหร่ | กระทบตัวเลขโดยตรง ถ้าไม่ตรง diff ที่เห็นจะเป็น protocol ไม่ใช่ algorithm |
| 4 | Paper รายงาน metric อะไรบ้าง (I-AUROC เท่านั้น หรือมี P-AUROC/PRO ด้วย) | ถ้า paper ไม่รายงาน P-AUROC ก็ไม่ต้องฝืนทำ compute_pixel=True |
| 5 | ตัวเลข reference หาได้จาก **primary source** (paper ต้นฉบับ) ไหม หรือต้องพึ่ง secondary source | ถ้าต้องพึ่ง secondary ต้อง cross-check ≥2 แหล่ง (ดูกรณี PaDiM) |
| 6 | Image-level score รวมจาก pixel map ยังไง (max? ค่าเฉลี่ยถ่วงน้ำหนัก? sum ของหลาย scale?) | อย่า assume ว่าทุก method ใช้ `max(pixel_map)` เหมือน 4 ตัวเดิม (ดูหมวด 5) |
| 7 | `src/models/<method>.py` implement ไว้แล้วหรือยัง | ถ้ายัง ต้องเขียนตาม `BaseAnomalyModel` interface (`fit`/`score` คืน `ScoreResult`) ก่อน ตาม pattern เดิม |

---

## 2. ขั้นตอน implement (ทำตามลำดับ)

### 2.1 เขียน/ตรวจ `src/models/<method>.py`
- ต้อง inherit หรือมี interface ตรงกับ `BaseAnomalyModel` (`fit(normal_loader)`,
  `score(loader) -> ScoreResult`)
- `ScoreResult.pixel_maps` ต้องเป็น **ค่าดิบ ไม่ normalize ต่อภาพ**
  (ดูหมวด 4 — นี่คือบั๊กที่เจอจริงใน PaDiM)
- เพิ่ม `weights_override` support ใน `_FeatureExtractor`/backbone class
  ถ้ายังไม่มี (pattern เดียวกับ 4 ตัวเดิม — `getattr(cfg, "BACKBONE_WEIGHTS", None)`)

### 2.2 เพิ่ม `PublishedReference` ใน `mvtec_repro/core.py`
```python
"newmethod": PublishedReference(
    method="NewMethod (backbone-variant)",
    citation="Author et al., Title, Venue Year",
    image_table="Table N ('Ours' column)",
    pixel_table="Table M" ,   # หรือ None ถ้า paper ไม่รายงาน
    protocol=ProtocolSpec(
        method="newmethod",
        preprocess=PreprocessSpec(resize=..., center_crop=...),
        backbone="...", backbone_weights="IMAGENET1K_V1",
        feature_layers=(...,), epochs=... ,  # None ถ้า training-free
        hyperparams=(("key", value), ...)),
    image_auroc=_pct({...ครบ 15 category หรือเท่าที่ paper รายงาน...}),
    pixel_auroc=_pct({...}),
    image_quality=SourceQuality.PRIMARY,   # หรือ SECONDARY (ดูหมวด 3.4)
    notes="...",
),
```
แล้วเพิ่ม test คู่กันใน `tests/test_core.py`:
```python
def test_reference_mean_matches_paper_for_newmethod():
    ref = core.REFERENCES["newmethod"]
    assert abs(ref.mean("image_auroc") * 100 - PAPER_STATED_AVERAGE) <= 0.1
```

### 2.3 เขียน `adapter_<method>.py`
Copy โครงจาก `adapter_patchcore.py` (training-free) หรือ `adapter_rd4ad.py`
(training-based) แล้วแก้แค่: import model class, field ที่ `cfg` ต้องมี
(เช็คจาก `self.cfg.XXX` ทุกจุดในไฟล์ model จริง — **อย่าเดา field name**),
และ `deviations=[...]` (ดูหมวด 3.5)

### 2.4 รัน `pytest mvtec_repro/tests -q` ให้ผ่านก่อนรันจริง

### 2.5 รันจริง — `run_benchmark()` จะรัน smoke test บนข้อมูลปลอมให้อัตโนมัติ
ก่อนแตะ MVTec จริงเสมอ (เว้นแต่ `--skip-smoke-test`) — ถ้า smoke test fail
แปลว่า adapter เขียนผิด ไม่ใช่ algorithm ผิด แก้ตรงนี้ก่อนเสมอ

---

## 3. Do's / Don'ts แยกตามประเภท algorithm

### 3.1 Training-free (memory-bank / สถิติปิด — แบบ PatchCore, PaDiM)

**DO**
- ยังต้อง multi-seed แม้ไม่มี training loop ถ้ามี randomness ใดๆ อยู่
  (coreset random projection ของ PatchCore, random channel selection ของ
  PaDiM) — seed กระทบผลจริง แค่กระทบน้อยกว่า training-based
- ตั้ง `epochs=None` ใน `ProtocolSpec` — นี่คือ signal ให้ทั้งระบบรู้ว่า
  ไม่มี training loop (harness ใช้เช็คตอน build smoke test ด้วย)
- ใส่ chunk-size parameter ถ้ามีการคำนวณ pairwise distance กับของที่เก็บ
  ไว้เยอะ (memory bank/covariance) กัน OOM บน category ใหญ่

**DON'T**
- อย่าลืมว่า "ไม่มี training loop" ≠ "seed ไม่สำคัญ" (เคยเข้าใจผิดแบบนี้
  ตอนคิดว่า PatchCore ไม่ต้องสน seed เลย)
- อย่าใส่ `PATIENCE`/early-stopping logic เข้าไปถ้า paper ไม่มี training
  loop ให้ early-stop ตั้งแต่ต้น

### 3.2 Training-based (มี gradient descent จริง — แบบ RD4AD, SimpleNet)

**DO**
- **บังคับ ≥3 seed เสมอ ไม่มีข้อยกเว้น** — seed กระทบทั้ง weight
  initialization และ optimizer trajectory พร้อมกัน variance สูงกว่า
  training-free มาก
- เช็ค `EPOCHS`/`LR`/optimizer settings ใน `ProtocolSpec` ให้ตรง paper
  เป๊ะ แล้ว **เทียบกับ production `config.py` แยกต่างหาก** — ถ้าต่างกัน
  (เช่น production ใช้ epoch น้อยกว่าเพื่อประหยัด compute) ต้องบันทึกช่องว่าง
  นี้ไว้เป็นลายลักษณ์อักษร ไม่ใช่ปล่อยผ่าน (ดูกรณี RD4AD 200 vs 100,
  SimpleNet 160 vs 100 — เคยเจอจริงและตัดสินใจ "ไม่แก้อะไร" (option C)
  ซึ่งแปลว่าต้องเขียนข้อจำกัดนี้ในเล่มเสมอ)
- ถ้า repo เพิ่ม early-stopping ที่ paper ไม่มี (เช่น `PATIENCE` ที่เป็น
  ของ repo เองไม่ใช่ของ paper) ตั้งค่าให้สูงพอที่จะไม่ trigger ก่อนจบ
  epoch ตาม paper ตอน validate (ดู `adapter_rd4ad.py`/`adapter_simplenet.py`
  ที่ตั้ง `PATIENCE` เท่ากับ `EPOCHS` โดยตั้งใจ)

**DON'T**
- **ห้ามสรุปว่า "algorithm ผิด" จากการรันครั้งเดียวที่ fail** — ต้องแยก
  ตัวแปรก่อน: seed variance ปกติ / epoch ไม่พอ / loss ต่างจาก official /
  architecture ต่างจาก official จริง (แต่ละอย่างเช็คแยกได้ ดูตัวอย่าง
  วิธีแยกใน session ที่ทำ RD4AD/SimpleNet)
- ห้ามใช้ loss function/optimizer settings ที่ต่างจาก official code โดย
  ไม่บันทึกไว้ใน `deviations=[...]` — ถ้า paper มี ablation วัดผลกระทบของ
  loss ที่ต่างไว้แล้ว (เช่น SimpleNet Table 3 "Ours-CE") ให้อ้างอิงตัวเลข
  นั้นเป็นเกณฑ์ตัดสินว่า diff ที่เห็น "ตามคาด" หรือ "ผิดปกติ"
- ห้ามเชื่อว่า "shape/stride ถูกต้อง" (จาก unit test เชิงสถาปัตยกรรม) แปลว่า
  "สถาปัตยกรรมเทียบเท่า paper" — เป็นเงื่อนไขจำเป็นแต่ไม่พอ (ดูกรณี RD4AD:
  OCBE/student เป็น conv/deconv ธรรมดา ไม่มี skip connection แบบ residual
  block ที่ paper ใช้ — shape ถูกแต่ topology ต่าง ยังไม่เคยวัดผลกระทบจริง)

### 3.3 Pixel-level metric (P-AUROC / PRO) — ทุก paper ที่รายงานตัวนี้

**DO**
- Resize/crop ของ mask ต้อง**เหมือนภาพเป๊ะ** ใช้ `PreprocessSpec` ตัวเดียวกัน
- ใช้ **nearest-neighbor interpolation สำหรับ mask เท่านั้น** (ไม่ใช่
  bilinear — mask เป็น binary ไม่ใช่ intensity image)
- เรียก `core.assert_raw_pixel_maps()` ก่อนคำนวณ P-AUROC ทุกครั้ง (มีอยู่
  แล้วใน `torch_runner.run_category()` — อย่าลบออกถ้า copy pattern ไปทำ
  harness อื่น)

**DON'T**
- **ห้าม normalize `pixel_maps` เป็น [0,1] ต่อภาพก่อนคำนวณ P-AUROC** —
  ทำลาย scale สัมพัทธ์ระหว่างภาพ ทำให้ P-AUROC มี bias (นี่คือบั๊กจริงที่
  เจอใน PaDiM — `RAW_PIXEL_MAPS=True` ที่เพิ่มเข้าไปมีไว้แก้เรื่องนี้
  โดยเฉพาะ)
- ถ้า paper ไม่รายงาน P-AUROC (เช่นบาง ablation-only paper) อย่าฝืนทำ
  compute_pixel เพราะจะไม่มี reference ให้เทียบเลย (`compare_to_reference`
  จะได้ `NO_REFERENCE` ทุกแถว ไม่มีประโยชน์)

### 3.4 Reference table — วินัยเรื่อง citation

**DO**
- Primary source เสมอถ้าเป็นไปได้ (เปิด paper ต้นฉบับเอง อ่าน table
  ตรงๆ) ระบุเลข Table/column ให้ชัด
- ถ้าจำเป็นต้องใช้ secondary source (บาง paper ไม่ได้ตีพิมพ์ตาราง
  per-category ครบ — เจอกับ PaDiM) **ต้อง cross-check ≥2 แหล่งอิสระ** และ
  ถ้าขัดกันเอง ให้ mark `SourceQuality.SECONDARY` + เขียน `notes` อธิบาย
  ว่าขัดกันตรงไหน ไม่ปิดบัง
- เขียน unit test เช็คว่า **ค่าเฉลี่ยของตาราง** ตรงกับตัวเลขที่ paper
  ประกาศไว้เอง (ในบทคัดย่อ/ผลสรุป) ภายใน ~0.1 percentage point — ถ้าไม่
  ตรง แปลว่า transcribe ผิดแถวหรือผิดคอลัมน์แน่นอน (วิธีนี้จับได้จริงว่า
  RD4AD เคยหยิบตารางผิดทั้งชุด และ PaDiM หยิบ pixel-level มาใส่
  image-level)

**DON'T**
- **ห้ามปล่อยค่า reference เป็นค่าว่าง/`None` แล้วรันไปก่อน** (เคยเกิดกับ
  PatchCore ทั้ง 15 category — แปลว่าไม่เคยมีการเทียบเกิดขึ้นจริงเลย
  ทั้งที่ script รันแล้วดูเหมือนเทียบแล้ว)
- ห้ามพิมพ์ชื่อ category ผิด/สะกดผิด โดยไม่มี validation (เคยเกิด
  `"transitor"` ใน PaDiM — ใช้ `PublishedReference.__post_init__` ที่มี
  อยู่แล้วซึ่งจะ raise ทันทีถ้าเจอ key ที่ไม่อยู่ใน `MVTEC_CATEGORIES`)
- ห้าม hardcode ตัวเลขจากความจำ (ของ LLM หรือของคนเขียนเอง) โดยไม่เปิด
  แหล่งที่มาจริงมาเทียบ

### 3.5 `deviations=[...]` — วินัยการประกาศความต่าง

ทุกจุดที่รู้ตัวว่าโค้ด/protocol ต่างจาก official implementation หรือ
production config ต้องใส่ไว้ตรงนี้เป็นประโยคอ่านง่าย ไม่ใช่แค่ comment ใน
โค้ดที่ไม่มีใครเห็น — ตัวอย่างที่ควรมี:
- backbone weight version ต่างจาก official (ถ้าจงใจต่าง)
- loss function ต่างจาก official + อ้างอิงว่า paper วัดผลกระทบไว้เท่าไหร่
  (ถ้ามี)
- architecture component ที่ verified แค่ shape ไม่ได้ verified topology
- epoch/hyperparameter ที่ adapter ใช้ (ตาม paper) ต่างจาก production
  `config.py` ที่ใช้จริงกับ PCB

---

## 4. Output format — ใช้ schema เดียวกับ 4 ตัวเดิมเป๊ะ

เรียก `core.build_run_record(reference, protocol, categories, started_at,
finished_at, declared_deviations=[...])` เสมอ — **อย่าประกอบ dict เอง** จะ
ได้ schema 4 ส่วนเหมือนกันทุก method โดยอัตโนมัติ:

```json
{
  "run_info": {"started_ict": "...", "finished_ict": "...", "environment": {...}},
  "protocol": {...ที่ใช้รันจริง...},
  "results": {
    "paper_citation": "...", "paper_image_table": "...",
    "categories": [
      {"category": "...", "paper_image_auroc": ..., "paper_pixel_auroc": ...,
       "n_train": ..., "n_test": ..., "per_seed": [...ค่าดิบ...]}
    ]
  },
  "summary": [{"category": "...", "metric": "...", "mean": ..., "std": ...,
               "paper": ..., "diff_vs_paper": ..., "status": "PASS|FAIL|..."}],
  "pass_rate": {"passed": N, "checkable": M, "percent": X}
}
```

Console output (Colab cell) ใช้ `_print_summary()` ที่มีอยู่แล้วใน
`torch_runner.py` — เรียกอัตโนมัติท้าย `run_benchmark()` ไม่ต้องเขียนซ้ำ

---

## 5. ตัวอย่าง "algorithm คนละแบบ" ที่ harness ปัจจุบันอาจไม่รองรับตรงๆ —
## ต้องอ่านให้ดีก่อนสมมติว่า pattern เดิมใช้ได้เลย

| ประเภท algorithm | ตัวอย่าง paper | จุดที่ต่างจาก 4 ตัวเดิม | ต้องปรับอะไร |
|---|---|---|---|
| **Normalizing flow** | CFLOW-AD, FastFlow | ไม่มี memory bank, มี invertible network ต้องเทรน, image score อาจไม่ได้มาจาก `max(pixel_map)` ตรงๆ | เช็คสูตร aggregate score ของ paper นั้นเองก่อน อย่า assume ว่า `ScoreResult.image_scores` คำนวณแบบเดียวกับ 4 ตัวเดิม |
| **Few-shot / zero-shot (prompt-based)** | WinCLIP | ไม่ fit ทีละ category แบบ one-class — อาจใช้ prompt เดียวกันข้ามหลาย category หรือต้องการแค่ few-shot ตัวอย่าง | `run_category()` assumption ที่ว่า "fit(normal_loader) เต็ม train split ต่อ 1 category" อาจไม่ตรง ต้องปรับ loop หรือเขียน adapter ที่ inject ข้อมูลน้อยกว่านั้น |
| **Diffusion-based** | DiAD | "fit" อาจหมายถึง fine-tune จาก diffusion checkpoint ขนาดใหญ่ ไม่ใช่ train from scratch | smoke test แบบเดิม (fit()+score() บนข้อมูลปลอมไม่กี่ภาพ) อาจไม่ครอบคลุมค่าใช้จ่ายจริง — พิจารณาลด scope ของ smoke test เฉพาะ diffusion-based ให้เบากว่านี้ |
| **Multi-class / unified model** | UniAD | เทรนโมเดลเดียวข้ามทุก category พร้อมกัน ไม่ใช่โมเดลแยกต่อ category | ต้องปรับ harness ทั้งชุด (ตอนนี้ออกแบบมาสำหรับ one-class-per-category ล้วนๆ) — เป็นงานใหญ่ ไม่ใช่แค่เขียน adapter |

**กฎทั่วไป**: ถ้า paper ใหม่ไม่ใช่ one-class, ไม่ fit ต่อ category เดี่ยวๆ,
หรือ image score ไม่ได้มาจาก pixel-map เดียวกันกับที่ใช้ทำ P-AUROC ให้
หยุดก่อนเขียน adapter ตาม pattern เดิม — อ่าน `torch_runner.run_category()`
ให้ทะลุก่อนว่า assumption ไหนจะพังกับ algorithm ใหม่นี้

---

## 6. Checklist สุดท้ายก่อนเชื่อผล (ก่อนเอาเข้าตาราง SOTA comparison ของเล่ม)

- [ ] `pytest mvtec_repro/tests -q` ผ่านหมด (รวม test ของ reference table ใหม่)
- [ ] Smoke test ผ่านก่อนรันจริง (อัตโนมัติ ไม่ต้องเช็คเอง)
- [ ] รัน ≥3 seed
- [ ] รัน ≥6 category รวม category ที่ paper เองก็ทำได้ไม่เต็ม 100%
      อย่างน้อย 2-3 ตัว (อย่าใช้แต่ bottle/wood/transistor — ไม่มีพลัง
      แยกว่า implementation ถูกจริงหรือบังเอิญ)
- [ ] `declared_deviations` ครบทุกจุดที่รู้ (หมวด 3.5)
- [ ] เทียบ `ProtocolSpec` (paper) กับ production `config.py` แล้ว **ตัดสินใจ
      ชัดเจน**ว่าจะทำ:
  - **A** — เพิ่ม validate อีกรอบด้วย config production จริง (เช่น
    `--epochs 100`) เพื่อดูช่องว่าง
  - **B** — แก้ production `config.py` ให้ตรง paper ไปเลย
  - **C** — ไม่แก้อะไร แต่ต้องเขียนข้อจำกัดนี้ในเล่มเสมอว่า "MVTec
    validation ใช้ config ตาม paper ไม่ใช่ config ที่รันจริงกับ PCB"
  - บันทึกว่าเลือกอะไรไว้ในไฟล์นี้เอง (เพิ่ม section ต่อท้ายตาม method)
- [ ] ผลลัพธ์ `.json` commit เข้า repo (`mvtec_repro_results/<method>.json`)
      พร้อม `git commit` message อ้างอิงว่ารันวันไหน seed อะไรบ้าง

---

## 7. Log การตัดสินใจต่อ method (เพิ่ม section ใหม่ทุกครั้งที่เพิ่ม baseline)

### PatchCore — เพิ่มเมื่อ 2026-09
- Training-free, tier ความเชื่อมั่นสูงสุด (ไม่มี training loop, สูตรตรวจ
  แล้วตรงสมการ paper)
- Config gap ระหว่าง paper protocol กับ production: resize/crop +
  weight version เท่านั้น ไม่มี epoch เกี่ยวข้อง
- ตัดสินใจ: **C** (ยังไม่แก้ config gap เรื่อง crop/weight version)

### PaDiM — เพิ่มเมื่อ 2026-09
- Training-free, reference table เป็น secondary source ที่ขัดกันเองใน
  บาง category — ใช้ class-mean + pixel-AUROC เป็นหลักในการตัดสิน
- ตัดสินใจ: **C**

### RD4AD — เพิ่มเมื่อ 2026-09
- Training-based, EPOCHS paper=200 vs production=100 (ยังไม่แก้)
- Architecture: OCBE/student verified แค่ shape/stride ไม่ verified
  residual/skip-connection topology เทียบ official
- ตัดสินใจ: **C** — ต้องเขียนข้อจำกัดนี้ในเล่มเสมอ

### SimpleNet — เพิ่มเมื่อ 2026-09
- Training-based, EPOCHS paper=160 vs production=100 (ยังไม่แก้)
- Loss BCE แทน truncated-L1 (paper วัดผลกระทบไว้แล้ว ~0.2-0.3pp)
- Adaptor bias: แก้แล้วให้ตรง paper (`ADAPTOR_BIAS=False` ตอน validate)
- ตัดสินใจ: **C**

### `<เพิ่ม method ใหม่ตรงนี้>`
- Training-free / training-based?
- Config gap กับ production คืออะไร?
- ตัดสินใจ A/B/C?
