# Pediatric Heart segmentation baseline

Дата: 2026-10-06. Работа только в `feature/pediatric-heart-segmentation`.
**QA cohort60, frozen42/9/9, train-only preprocessing и собственная2.5D U-Net
готовы; один30-epoch RunPod baseline завершён, original-grid test n=9 оценён.**
Training commit `7c93cb9f8b284a9a526e78c9c67432e26d15e794`; mean test Dice0.9194.
Critical results скачаны на E и SHA-verified; Pod можно остановить.
Полный отчёт: [RunPod baseline v1](pediatric_heart_runpod_v1.md).
Подготовка: [training readiness](pediatric_heart_training_readiness.md).
Продолжение: [resume](pediatric_heart_segmentation_resume.md).
Основной checkout/пользовательские изменения, main и audit branch не менялись.

## 1. Цель

Сегментировать исходный экспертный **Heart OAR** на детских CT разных возрастов
и scanner/protocol. Цепочка: полная DICOM CT → подтверждённые HU/geometry →
original RTSTRUCT Heart → original-grid GT → QA → patient split → train-fitted
preprocessing → собственная модель → benchmark → full training → original-grid evaluation. Камеры, сосуды, диагнозы
и clinical decision logic не входят в этот эксперимент.

Уточнение цели от 2026-10-09: **0–17 лет**, с отдельной проверкой новорождённых
и младенцев. Baseline v1 остаётся 2–16/Heart OAR; split, preprocessing,
checkpoints и оценки не менялись.
[Clinical workflow](clinical_reconstruction_workflow.md) и
[аудит внешних CT/SEG](external_ct_import_audit.md) — отдельный engineering этап.

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
Inverse XYZ/RAS affine и pixel-center mapping сохранены; prediction probability
linear обратно, threshold на original grid. Train mmapcache: 10280 slices,
1742 positive /8538 negative; epoch all-positive +equal-negative per patient =3484.
Held-out/inference all-slices. Ни held-out fitting, ни GT crop нет.

## 9. Архитектура

[Own model](../heart3d/ml/model.py), с нуля, **488993 parameters**. Input5,
encoder16/32/64/128, 3 MaxPool2d(2), double Conv3×3 bias=True /GN8 /ReLU;
decoder bilinear до skip shape, concat +double block, output Conv1×1→1 logits.
Готовая segmentation network не импортируется. CPU и RTX4090 CUDA backends реально проверены. Odd H/W и target output shape протестированы.

## 10. Loss

[Implemented losses](../heart3d/ml/losses.py): BCEWithLogits, own soft Dice, combined1/1.
Для p=sigmoid(logits), epsilon=1e−6:

`BCE = -mean[y log(p)+(1-y) log(1-p)]` (stable BCEWithLogits).

`DiceLoss = mean_batch[1-(2 sum(p*y)+epsilon)/(sum(p)+sum(y)+epsilon)]`.

Empty targets сохранены, epsilon smoothing; BCE даёт основной background signal.
Synthetic tests passed; один full combined experiment, три loss experiments не запускались.

## 11. Augmentation

Train only: rotation ±5°, scale 0.95..1.05, normalized shift ±0.03 (~29.5 HU),
noise sigma 0.01 (~9.85 HU), clamp. Shared spatial transform пяти planes и target,
image bilinear/mask nearest. Flips запрещены guard. Seed/epoch/index deterministic;
val/test без augmentation.

## 12. Training setup

Full config [pediatric_heart_runpod_v1.json](../configs/pediatric_heart_runpod_v1.json):
FP32 CUDA, batch16, workers0, CPUthreads6, Adam0.001, без scheduler, seed20261006.
30 epochs с нуля, balanced3484samples/218batches per train epoch.
Validation — все central positions и mean patient Dice на preprocessing grid.
Best epoch18; last epoch30; benchmark/pilot weights не загружались.
TF32 off, CUDA deterministic warn-only; actual environment/pip freeze сохранены.
CLI: `python -m scripts.run_pediatric_gpu --config ... --data ... --run-name ... --export ...`.
Trainer/evaluate/bundle/infer реализованы внутри проекта; test не используется для fitting.

## 13. Hardware и измеренный бюджет

RTX4090 24564MiB, torch2.8.0+cu128/runtime12.8, Python3.12.3;32logical CPU,
RAM124.9GiB, persistent /workspace. GPU benchmark50+5warmup+20sanity:
**0.3344s/batch16,47.84samples/s**, VRAM allocated1.916GB/reserved3.127GB.
Peak sampled benchmark RSS1.790GB; full-train RSS2.072GB.
Измеренный benchmark estimate30train epochs36.45min без overhead;
**фактические30epochs с validation/save 41.10min**.
Train/validation epoch в среднем72.95/9.11s.
NPY handles cache устранил повторные network opens; pixels lazy, максимум2patientmaps.
Начальный медленный pilot остановлен и сохранён; final restarted from scratch.

Historical CPU readiness: Ryzen54500/16GiB/torch2.14.1+cpu, batch2,0.4150s,
4.819samples/s,RSS624MB;30train-only extrapolation6.02h. Это не full CPU run.
[CPU summary](../metadata/pediatric/heart_cpu_benchmark_v1.json).

## 14. Метрики

Patient-weighted original DICOM-derived XYZ test n=9, probabilities inverse-linear,
threshold0.5, без postprocessing. Geometry/spacing подтверждены в каждом case.

| Metric | Mean | Median | Range |
|---|---:|---:|---:|
| Dice | 0.9194 | 0.9251 | 0.8579–0.9658 |
| IoU | 0.8521 | 0.8606 | 0.7512–0.9339 |
| precision | 0.9472 | 0.9636 | 0.8518–0.9900 |
| recall | 0.8977 | 0.9101 | 0.7569–0.9799 |
| HD95_mm | 17.9010 | 14.0000 | 3.1250–70.6495 |
| ASSD_mm | 3.0077 | 3.1639 | 1.4295–4.5990 |


HD95=max двух directed95th percentiles; ASSD=pooled directed surface-voxel mean,
6-neighbour boundary, units mm. Empty convention и valid counts сохранены;
в этом test пустых prediction нет. Per-patient и age/scanner/contrast/Z-spacing
таблицы находятся в [полном отчёте](pediatric_heart_runpod_v1.md).

## 15. Результаты и проверки

Полный локальный suite **127passed** (322 существующих NumPy/skimage warnings);
после расширения failure review дополнительно prediction-QA test1passed.
DICOM/HU/order/LPS→RAS, references/planes, rasterization, forward/loss/gradients,
physical contexts, augmentation, mmap eviction, seed, patient isolation, inverse
geometry, full synthetic trainer, original metrics и SHA bundle проверены.
GPU sanity fixed loss1.4544→0.7031; finite gradients/optimizer update подтверждены.
Это technical sanity; quality относится только к untouched-test pass финального run.

[GPU experiment summary](../metadata/pediatric/heart_gpu_baseline_v1.json).
Best validation Dice0.92967 на preprocessing grid; mean test Dice0.91937 на
original grid. Это первый source-OAR baseline, не validation clinical support.

## 16. Failures

Data QA:50 scan truncations excluded,3 irregular geometry rejected,1coverage review
и1identity review сохранены. Planar source caps не достраивались.
Prediction failures:792705D7 HD95=70.65mm, off-heart islands;423F282F islands
в области CT table. CA967BD7 (2y) Dice0.8579/recall0.7569/volume−23.54%:
недосегментация верхней области и внутренние gaps. F50AD62F precision0.8518,
volume+12.97%. Все ошибки сохранены без repair и без test-based параметров.
Age-group mean Dice2–5/6–11/12–17=0.8973/0.9315/0.9293, n3each: descriptive only.
Scanner LightSpeed/SOMATOM=0.9118(n6)/0.9345(n3); effects не отделены от protocol/age.

## 17. Ground Truth vs Prediction и inference

Original-grid masks доступны для всех9test patients, GT unchanged.
Minimal NiiVue viewer имеет Ground Truth / Prediction / Comparison; реальные
F22EFF95 три modes проверены headless Edge, JS errors0, screenshot просмотрен.
Static native axial/coronal/sagittal + endpoint comparisons просмотрены для
трёх заранее выбранных age cases и отдельно lowest-Dice CA967BD7.
[QA summary](../metadata/pediatric/heart_prediction_qa_v1.json).

Image-only CLI `python -m heart3d.ml.infer_dicom --config ... --checkpoint ...
--ct-series ... --output ... --device cpu --reserve-GB 80`: inspect full series,
HU/geometry gates, preprocessing, model, inverse mask + provenance.
На train/development06722123 проверено реально: shape512×512×320,639667voxels;
GT/RTSTRUCT не использовались, источники не менялись. Это technical smoke.
Predictions/checkpoints всегда отдельны от original CT/GT, no overwrite.

## 18. 3D comparison

Existing mask→mesh выполнен для3preselected test +1posthocfailure. Binary label1
явно переименован Heart OAR, не LV; RAS/mm field metadata подтверждены.

| Patient suffix | GT / prediction volume, ml | GT / prediction area, mm² | GT / prediction components (26) |
|---|---:|---:|---:|
| 423F282F | 661.20 / 642.08 | 47867.6 / 55413.0 | 1 / 3 |
| 54BC2D63 | 159.64 / 150.26 | 16950.6 / 16932.8 | 1 / 3 |
| 792705D7 | 358.77 / 321.66 | 29990.0 / 31448.8 | 1 / 5 |
| CA967BD7 | 162.29 / 124.09 | 18334.8 / 19663.8 | 1 / 1 |


Все четыре GT имеют1voxel26 component; prediction3/3/5/1 соответственно.
Boundary/nonmanifold/inconsistent winding edges0, self-intersections не проверены.
Topology/euler/component diagnostics и mesh vertex-distance metrics сохранены;
последние отличаются от voxel HD95/ASSD. No smoothing/cleanup/component removal.
Наличие закрытого mesh не означает корректную heart anatomy.

## 19. Ограничения

Source OAR scope не гарантирует полную anatomy камер/сосудов, approvals технические
без нового clinician sign-off, healthy_status=unknown. Reported ages2–16, age17нет.
Coverage selection меняет domain distribution; held-out n9,2scanner models,
Revolution/confirmed noncontrast не проверены. Public IDs/content SHA не доказывают
все repeat identities. Shape256²/Z2mm теряет native detail; planar boundaries
влияют на learning/metrics. Высокий Dice сопровождается островками и miss boundaries.
Test теперь просмотрен: для следующих tuned methods нужен заранее определённый
evaluation protocol; нельзя вновь называть этот test untouched.
CT/MRI joint training, diagnosis/healthy classifiers и СППР здесь не реализованы.

## 20. Воспроизведение, сохранность и следующий этап

Все тяжёлые artifacts E-selected data root,80GBreserve. Results под
`runpod_results/gpu_full_v1_20261006_io/`, QA/plots/meshes под
`experiments/heart_baseline_v1/gpu_full_v1_20261006_io/`. Все31result+9audit SHAverified,
best/last CPU load verified. Pod можно остановить, автоматически не останавливался.
В Git только code/config/tests/docs/small metadata. Main merge не выполняется.
Команды, pinned commit/config hashes и artifact SHA: [RunPod report](pediatric_heart_runpod_v1.md).

Следующий отдельно согласуемый experiment: экспертно уточнить source target,
исследовать background islands/3D continuity на train/validation, зафиксировать
v2 method и hold-out plan до новых экспериментов. Не менять frozen v1 и его scores.
Многоклассовые cardiac CT из НИИ требуют отдельных согласованных anatomical labels/
clinical metadata; никаких автоматических classifiers или MRI intensity mixing.
