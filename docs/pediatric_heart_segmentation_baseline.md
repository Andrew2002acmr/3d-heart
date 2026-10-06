# Pediatric Heart segmentation baseline

Дата: 2026-10-06. Ветка `feature/pediatric-heart-segmentation`, audit base `66c5804`.
Основной checkout и пользовательские изменения не трогались.

**Готовы QA cohort 60, frozen split 42/9/9, train-only preprocessing, собственная
2.5D U-Net и короткий CPU benchmark. Full training и test evaluation не было.**
Подробный отчёт: [pediatric_heart_training_readiness.md](pediatric_heart_training_readiness.md).
Продолжение: [pediatric_heart_segmentation_resume.md](pediatric_heart_segmentation_resume.md).

## 1. Цель

Сегментировать исходный экспертный **Heart OAR** на детских CT разных возрастов
и scanner/protocol. Цепочка: полная DICOM CT → подтверждённые HU/geometry →
original RTSTRUCT Heart → original-grid GT → QA → patient split → train-fitted
preprocessing → собственная модель → короткий benchmark. Камеры, сосуды, диагнозы
и clinical decision logic не входят в этот эксперимент.

## 2. Источник

[TCIA Pediatric-CT-SEG](https://www.cancerimagingarchive.net/collection/pediatric-ct-seg/),
DOI [10.7937/TCIA.X0H0-1706](https://doi.org/10.7937/TCIA.X0H0-1706),
[source publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC9090951/).
Clinical CT + экспертный OAR RTSTRUCT. Census: 327 с reported age 2–17,
фактически **2–16**, без 17-летних. Рабочие группы 2–5 / 6–11 / 12–17:
143 / 105 / 79. DICOM NNNY не exact birthday. **healthy_status=unknown**;
отсутствие CHD label не подтверждает норму. MRI/Atlas/healthy classifier не используются.

## 3. DICOM QA

`heart3d/dicom/ct.py`: classic single-frame CT, patient/age/series/frame/SOP,
inventory completeness, IOP/IPP/PixelSpacing, slope/intercept/RescaleType.
Enhanced CT, irregular grids, mixed series и gantry shear автоматически не исправляются.
Sort: `dot(IPP,cross(IOP[:3],IOP[3:]))`. Z spacing определяется соседними проекциями,
SliceThickness сохраняется как nominal metadata. Regularity tolerance:
max(0.02 mm, 1% step), grid residual ≤0.05 mm. Inventory подтверждает полноту
опубликованной series, не неопубликованной acquisition.

Массив `[column,row,slice]`, LPS affine axes: IOP[:3]*PixelSpacing[1],
IOP[3:]*PixelSpacing[0], normal*actual Z. Origin — первый sorted IPP, voxel center 0.
HU = stored*slope+intercept каждого slice при подтверждённых tags.
NIfTI RAS = `diag(-1,-1,1,1) @ LPS`, qform=sform, mm; массив не переворачивается.
Independent GDCM/SimpleITK sorting/affine и все HU pixels проверены.
[Image Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html).

## 4. RTSTRUCT → mask

`heart3d/dicom/rtstruct.py`: единственный nonempty Heart ROI, original patient,
series/frame/SOP class и global/Heart references. LPS points → inverse CT affine;
slice определяется SOP, missing reference не заменяется ближайшим. FOV/plane
residual ≤0.05 mm, union CLOSED_PLANAR / XOR CLOSEDPLANAR_XOR; mixed types rejected.
Заполнение voxel centers **без межсрезовой GT interpolation**. CT/GT shape,
spacing/origin/affine совпадают. Independent VTK stencil: Dice ≥0.99,
differences только на boundary. [ROI Contour Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.8.6.html).
Original DICOM не изменяются; census Heart extracts не являются full RT.

## 5. Включение и visual QA

Необходимы age/full inventory/HU/geometry/references gates, continuous Heart OAR,
independent agreement и visual alignment/coverage. Face contact, margin ≤1 slice,
internal gaps/components требуют review; unusual size/spacing не критерий ranking.
Sampling v2: axial first−2 / first / first+2 / median active / last−2 / last /
last+2 (clamp), native coronal/sagittal median active; CT+mask, original contour
на axial, physical aspect без resampling.

Различаем scan truncation, annotation scope, technical error и ROI endpoint.
Planar cap внутри полного scan сохраняется как original OAR; anatomical chamber
completeness не утверждается. Неоднозначные cases оставлены review, repair нет.
`approved` — technical visual suitability, не новый clinician sign-off или healthy.

## 6. Census и полученные данные

[327-patient registry](../metadata/pediatric/heart_segmentation_candidates.json):
327 Heart definitions, 324 nonempty, 322 reference candidates, 2 reference failures,
3 empty. Census CT probe/Heart item не заменяют full-stack QA.
Scanner LightSpeed/SOMATOM/Revolution = 134/128/65; contrast evidence/unknown = 301/26.

**117 complete CT/RT pairs**: 9 development +60 initial +48 replacements.
**114 DICOM и 114 RTSTRUCT gates pass**, все 114 native visual reviews выполнены.
**62 approved**, 50 coverage_incomplete, 3 geometry_failure, 1 coverage_review,
1 identity_review. Annotation/reference/empty statuses в этой wave =0; исходные
reference/empty failures не выбирались.
[Full audit](../metadata/pediatric/heart_qa_audit_v1.json),
[reviews](../metadata/pediatric/heart_cohort_reviews_v1.json). Historical pilot
summaries сохранены отдельно; ещё два original reference-only RT без full CT.
Все 234 source ZIP SHA совпадают с receipts, recovery завершён, originals не менялись.
Крупные raw/derived/ML artifacts на E через config/CLI; D-copy сохранена.

## 7. Frozen cohort и split

**60**: первые 20 каждой age group по predeclared seeded metadata queue;
два approved young cases в reserve. Volume/beauty не входят в ranking.
Seed 20261006, **42/9/9**, внутри каждой age group 14/3/3. Development IDs вне test.
[Cohort](../metadata/pediatric/heart_approved_cohort_v1.json),
[frozen split](../configs/splits/pediatric_ct_heart_v1.json) с cohort SHA.
Public patient IDs не пересекаются; exact HU audit 114, duplicate groups 0.
Unresolved related pair не включена как два units; clinical identity по anonymized
данным полностью не доказуема. Frozen v1 не менять.
Train scanner 29/12/1, held-out 6/3/0 каждый; contrast 39/3, held-out 9/0.
Revolution и confirmed non-contrast generalization этим split не оцениваются.

## 8. Preprocessing

Только 42 train: CT512XY, FOV 200/280/~500 mm, 30 Z2 mm /12 Z0.625 mm,
standard axial IOP. Heart OAR volume 133/343/744 ml, median voxel fraction 0.844%;
original positive/negative slices 2830/14350.
[Statistics](../metadata/pediatric/heart_train_statistics_v1.json).
[Frozen preprocessing](../configs/pediatric_heart_preprocessing_v1.json):
HU **−1000..969** по train CT0.5th/body99.5th percentiles → [-1,1]; full physical
FOV fit/pad **256×256**, **Z2 mm**, no GT crop. Image linear/mask nearest, original
GT сохранён. Context **[-4,-2,0,2,4] mm**, central target; missing offsets
replicate-edge +flags. Native orientation сохранена; другие IOP требуют adapter.
Inverse XYZ/RAS affine и pixel-center mapping сохранены; future probability
linear обратно, threshold на original grid. Train mmapcache: 10280 slices,
1742 positive /8538 negative; epoch all-positive +equal-negative per patient =3484.
Future held-out/inference all-slices. Ни held-out fitting, ни GT crop нет.

## 9. Архитектура

[Own model](../heart3d/ml/model.py), с нуля, **488993 parameters**. Input5,
encoder16/32/64/128, 3 MaxPool2d(2), double Conv3×3 bias=True /GN8 /ReLU;
decoder bilinear до skip shape, concat +double block, output Conv1×1→1 logits.
Готовая segmentation network не импортируется. CPU backend реально проверен,
CUDA=false, Radeon не проверен. Odd H/W и target output shape протестированы.

## 10. Loss

[Implemented losses](../heart3d/ml/losses.py): BCEWithLogits, own soft Dice, combined1/1.
Для p=sigmoid(logits), epsilon=1e−6:

`BCE = -mean[y log(p)+(1-y) log(1-p)]` (stable BCEWithLogits).

`DiceLoss = mean_batch[1-(2 sum(p*y)+epsilon)/(sum(p)+sum(y)+epsilon)]`.

Empty targets сохранены, epsilon smoothing; BCE даёт основной background signal.
Synthetic tests passed; benchmark combined, три full experiments не запускались.

## 11. Augmentation

Train only: rotation ±5°, scale 0.95..1.05, normalized shift ±0.03 (~29.5 HU),
noise sigma 0.01 (~9.85 HU), clamp. Shared spatial transform пяти planes и target,
image bilinear/mask nearest. Flips запрещены guard. Seed/epoch/index deterministic;
val/test без augmentation.

## 12. Training setup

[Config](../configs/pediatric_heart_baseline_v1.json), own bounded optimizer loop
`heart3d.ml.benchmark`: CPU6 threads, batch2, workers0, Adam0.001, no scheduler,
seed20261006. Реально 5 warmup +50 measured +20 fixed-train sanity =**75 updates**.
History/config/environment/resources/last_benchmark.pt на E. Full train/validation
loop best/last — следующий отдельно разрешаемый этап. Benchmark weights не final
model; full baseline должен стартовать с нуля.

## 13. Hardware и измеренный бюджет

Ryzen5 4500, 6 physical/12 logical, RAM~16 GiB, Windows11, Python3.14.4,
torch2.14.1+cpu. **0.4150 sec/batch**, p95 0.4497, **4.819 samples/sec**;
peak sampled RSS **624 MB**, CPU579% (~48.3% machine), min available RAM7.69 GB,
checkpoint5.95 MB. [Measured summary](../metadata/pediatric/heart_cpu_benchmark_v1.json).
1742 batches/epoch → **12.05 min**, 30 training epochs → **6.02 h**, linear
extrapolation **без validation/save overhead**. Full wall time больше,
эти составляющие не измерены. E~281.48 GB free, reserve80 decimalGB;
require_space перед allocations, автоматического удаления нет.

## 14. Метрики будущего baseline

Dice/IoU/precision/recall per patient и age/scanner/contrast/spacing aggregates.
HD95/ASSD mm только confirmed geometry, explicit empty-mask/surface conventions.
**Test evaluation не было**, scientific scores отсутствуют.

## 15. Технические результаты

**120 tests passed**, 320 прежних NumPy/skimage warnings. DICOM/HU/RAS-LPS,
inventory/references/planes, rasterization, own model/losses/gradients, physical
neighbors, seed/indexing, patient isolation, shared augmentation, inverse landmarks.
Three-age train preprocessing mosaic просмотрен, context/central GT aligned.
114 independent reader checks HU difference0; rasterizer Dice min0.9999947272818347,
nonboundary differences0 — conversion agreement, не ML metric.
Fixed2 train slices loss **1.4544→1.0210→0.8787**, finite gradients/weight updates,
no NaN/Inf. Это technical learning sanity, не generalization.

## 16. Failures

50 truncated scans исключены без repair; E03568A6 coverage review.
376/37058120/EB1DCBAA irregular spacing rejected. 176261A0 identity review:
похож на92891A2F, voxels различаются, identity unknown; только92891A2F включён.
Source planar caps сохранены в OAR target.
Historical full-RT failures:272B6C5D missing17 Heart references, 34ECBB32 missing8:
[evidence](../metadata/pediatric/heart_segmentation_reference_review.json).
Они и3 empty не candidates. Interrupted download cache сохранён/recovered,
OS root lock предотвращает overlap; originals untouched, resolution в readiness.

## 17. Ground Truth vs Prediction

Independent NiiVue `scripts/view_pediatric_ct_qa.py`: loopback, allowlisted CT+GT,
pinned CDN; native mosaics reproducible. Benchmark predictions не экспортировались.
GT/Prediction/Comparison и test masks — после отдельно разрешённого full baseline.

## 18. 3D comparison

GT/Prediction surfaces/volume/area/components/distances/topology пока отсутствуют.
После training/inference existing mask→mesh с explicit binary Heart mapping;
старый label1=LV нельзя выдавать за Heart. Physical units только verified geometry.

## 19. Ограничения и рекомендация

OAR extent не гарантирует полную chamber/vascular anatomy, technical approval
не clinician sign-off; clinical status unknown, reported age limited, age17 нет.
Coverage selection меняет scanner/protocol distribution, held-out два scanner models,
confirmed non-contrast нет. Public IDs/hashes не доказывают все repeat identities.
**Технически data/model готовы** к full source-OAR baseline, CPU/RAM/storage
достаточны. Нужно отдельное согласование full training и полноценный
train/validation loop с original-grid evaluation/failure review. Clinical reasoning
этим экспериментом не валидируется; полное обучение здесь не выполнялось.

## 20. Воспроизведение и следующий этап

Pinned requirements-lock/pediatric/ml. Current Python
`D:/codexProjects/3d-hearts/.venv/Scripts/python.exe`; external pydicom PYTHONPATH
`D:/codexProjects/3d-hearts/data/pediatric_audit/python_deps`. Data paths config/CLI.

```powershell
$dataRoot='E:/3d-heart-data/pediatric_ct_heart'
python -m heart3d.ml.prepare --config configs/pediatric_heart_baseline_v1.json --data $dataRoot --partition train
python -m heart3d.ml.benchmark --config configs/pediatric_heart_baseline_v1.json --data $dataRoot --batches 50 --run-name new_unique_bounded_run
python -m pytest -q --basetemp "$dataRoot/temporary/pytest_new_unique"
```

Это bounded reproduction, не разрешение full training. Полные команды/artifacts
в readiness. Frozen v1 не менять, approved reports не refresh, originals не
перезаписывать, data/weights не коммитить. Следующий отдельно согласуемый шаг:
full train/validation, один untouched-test pass, failures, GT/Prediction viewer/3D.
