# RunPod: завершённый Heart baseline v1

Дата: 2026-10-06. Один разрешённый пользователем полный baseline завершён.
Run: `gpu_full_v1_20261006_io`; training commit `7c93cb9f8b284a9a526e78c9c67432e26d15e794`.
Ветка `feature/pediatric-heart-segmentation`. Main и другие feature-ветки не менялись.
Модель обучена с нуля; test оценён один раз, результаты сохранены локально и проверены.

## Frozen experiment

Когорта 60, train/validation/test **42/9/9**, по 20 пациентов каждой рабочей
возрастной группы. Реальный возраст 2–16 лет; клинический статус unknown.
Target — original expert Heart OAR. Это не полный набор камер/сосудов.
Preprocessing fit только на 42 train: HU [−1000,969] → [−1,1], полный physical FOV
fit/pad 256², Z 2 mm, axial offsets [−4,−2,0,2,4] mm, image linear / mask nearest.
GT-guided crop отсутствует, original GT сохраняется отдельно.

Own 2.5D U-Net: 5→16→32→64→128→1, double Conv3×3/GN8/ReLU, три down/up stages,
bilinear decoder, **488993 parameters**. FP32, batch16, Adam LR0.001, 30 epochs,
без scheduler, BCE + own Soft Dice (1+1), seed20261006, workers0, CPU threads6.
Train augmentation: rotations ±5°, scale ±5%, shift ±0.03, noise0.01; без flips.
Balanced train epoch 3484 samples / 218 batches; validation/test все позиции.
Никаких loss sweeps, threshold fitting, TTA или postprocessing.

Best checkpoint — maximum mean patient validation Dice на preprocessing grid.
Выбран epoch **18**, validation Dice **0.92967**.
После inverse linear probability transform threshold0.5 применён на original grid.
Validation и test scores вычислены на разных grids, это явно сохранено в provenance.
CUDA seeds/order фиксированы, TF32 off, cudnn benchmark off; deterministic
algorithms warn-only из-за возможного bilinear backward. Побитовую идентичность
между CUDA/PyTorch версиями не обещаем; actual environment и pip freeze сохранены.

## Проверенные ресурсы и benchmark

RTX4090 **24564 MiB**, driver580.173.02, torch2.8.0+cu128, CUDA runtime12.8,
Python3.12.3. 32 logical CPUs / 16 physical, RAM134.12 decimal GB (~124.9 GiB).
`/workspace` — отдельный persistent FUSE mount `euro-2.runpod.net`.
df сообщает размер shared storage pool, а не индивидуальную квоту volume.
Репозиторий, venv, cache/tmp, training artifacts и exports находятся под /workspace.
Проверены nvidia-smi, torch CUDA, df, nproc, free; CPU torch на Pod не устанавливался.

5 warmup + 50 measured + 20 fixed train sanity steps:
**0.3344 s/batch**, p95 **0.3973 s**,
**47.84 samples/s** при batch16.
GPU peak allocated **1.916 GB**, reserved
**3.127 GB**; sampled process RSS
**1.790 GB**.
Fixed-train loss **1.4544 → 0.7031**,
finite gradients и optimizer updates подтверждены. Это technical sanity, не quality score.
Benchmark weights не сохранялись и не инициализировали полный run.

Начальная попытка `gpu_full_v1_20261006` прошла benchmark1.0310 s/batch, но
была остановлена SIGTERM из-за повторного NPY open/header чтения с network volume
(~90% времени пяти profiled batches). Test в ней не запускался. Logs, interruption
receipt и pilot benchmark сохранены. Исправлено кеширование маленьких file handles/
headers; read-only pixel maps ограничены двумя пациентами. Samples до/после eviction
проверены на совпадение. Скорость при одинаковом batch16 выросла примерно в3.08 раза.
Final `gpu_full_v1_20261006_io` снова начат с seeded random weights, не из pilot checkpoint.

Benchmark прогнозировал train epoch **72.9 s**
и 30 train epochs **36.5 min** без validation/save.
Реально 30 epochs с validation/checkpoints: **41.10 min**.
Средний train epoch **72.95 s**, validation
**9.11 s**. Peak sampled full-training RSS
**2.072 GB**;
CPU mean **224.7%** процесса
(100% = одна logical CPU), **7.02%** машины.
Новые estimates для иной hardware/cohort требуют нового измерения.

## Test: original DICOM-derived grid, n=9

| Metric | Mean | Median | Range |
|---|---:|---:|---:|
| Dice | 0.9194 | 0.9251 | 0.8579–0.9658 |
| IoU | 0.8521 | 0.8606 | 0.7512–0.9339 |
| precision | 0.9472 | 0.9636 | 0.8518–0.9900 |
| recall | 0.8977 | 0.9101 | 0.7569–0.9799 |
| HD95_mm | 17.9010 | 14.0000 | 3.1250–70.6495 |
| ASSD_mm | 3.0077 | 3.1639 | 1.4295–4.5990 |


HD95 — максимум двух направленных 95th percentiles; ASSD — pooled mean
surface-voxel distances, 6-neighbour boundary. Units mm подтверждены DICOM QA.
Пациенты имеют одинаковый вес в aggregate; voxels между пациентами не объединяются.
Пустые prediction дают overlap0 и undefined surface distances, здесь пустых нет.

| Patient suffix | Age | Dice | HD95, mm | ASSD, mm |
|---|---:|---:|---:|---:|
| 423F282F | 15 | 0.9384 | 18.17 | 3.79 |
| 54BC2D63 | 2 | 0.9345 | 5.62 | 1.43 |
| 572C61DF | 12 | 0.9076 | 14.00 | 3.16 |
| 792705D7 | 10 | 0.9251 | 70.65 | 4.60 |
| 9B29D191 | 15 | 0.9419 | 7.34 | 1.85 |
| C26BBB5F | 6 | 0.9658 | 3.12 | 2.32 |
| CA967BD7 | 2 | 0.8579 | 17.09 | 3.70 |
| F22EFF95 | 3 | 0.8995 | 7.11 | 2.21 |
| F50AD62F | 6 | 0.9037 | 18.00 | 4.00 |


| Group | n | Mean Dice | Mean HD95, mm |
|---|---:|---:|---:|
| by_age_group: 12-17 | 3 | 0.9293 | 13.17 |
| by_age_group: 2-5 | 3 | 0.8973 | 9.94 |
| by_age_group: 6-11 | 3 | 0.9315 | 30.59 |
| by_scanner: SOMATOM Definition AS+ | 3 | 0.9345 | 9.47 |
| by_scanner: LightSpeed VCT | 6 | 0.9118 | 22.12 |
| by_native_z_spacing: 2.0 | 7 | 0.9164 | 12.12 |
| by_native_z_spacing: 0.625 | 2 | 0.9298 | 38.14 |


Все9 held-out cases имеют reported contrast agent; нет confirmed non-contrast
comparison. Subgroups малы (2–7), scanner/age/spacing effects здесь описательные,
причинных выводов и подтверждения domain robustness нет.

## Failures и visual / 3D QA

Средний Dice0.9194 не устраняет boundary failures. `792705D7` имеет HD95
70.65 mm и вне-сердечные islands в верхней части живота; `423F282F` —
также отдельные islands в области CT table. `CA967BD7` (2y, lowest Dice0.8579)
недосегментирован: recall0.7569, volume error−23.54%, верхняя часть OAR и
внутренние пропуски видны coronal/sagittal. Никакие islands/holes не исправлялись.

Для первых lexical IDs каждой age group:423F282F/54BC2D63/792705D7 сохранены
axial/coronal/sagittal и slices около OAR endpoints, GT/Prediction binary Heart
surfaces с существующим mask→mesh. CA967BD7 просмотрен отдельно post-hoc.
CT/GT/pred shape/affine и source SHA совпадают. Original GT неизменён.

| Patient suffix | GT / prediction volume, ml | GT / prediction area, mm² | GT / prediction components (26) |
|---|---:|---:|---:|
| 423F282F | 661.20 / 642.08 | 47867.6 / 55413.0 | 1 / 3 |
| 54BC2D63 | 159.64 / 150.26 | 16950.6 / 16932.8 | 1 / 3 |
| 792705D7 | 358.77 / 321.66 | 29990.0 / 31448.8 | 1 / 5 |
| CA967BD7 | 162.29 / 124.09 | 18334.8 / 19663.8 | 1 / 1 |


Meshes closed, boundary/non-manifold/inconsistent winding edges0 в четырёх
сравнениях; self-intersections не тестировались. Predictions имеют islands и
изменённую топологию, Euler/component diagnostics сохранены. Voxel26 components
и vertex-connected surface components — разные определения.
Mesh distances vertex-to-vertex сохранены отдельно от voxel HD95/ASSD;
никаких smoothing, component removal или автоматического repair.

Viewer `scripts/view_pediatric_ct_qa.py` показывает original CT+GT+prediction:
Ground Truth / Prediction / Comparison. Headless Edge/NiiVue0.69 на реальном
F22EFF95 подтвердил все три opacity modes и отсутствие JS errors; screenshot
Comparison просмотрен. Viewer loopback-only,9 test predictions доступны.
Raw DICOM→model→original mask отдельно проверен на engineering/train06722123
без RTSTRUCT/GT; это smoke check, не дополнительный test experiment.

## Transfer, integrity и локальная сохранность

Minimal input bundle **201 files**, source4,784,231,262 bytes, archive2,352,906,707 bytes.
SHA256 `8c8c2e7981cea6497e77462c403b396429ba4f840504d7818b6650bb29ad164e`.
Raw DICOM/RTSTRUCT не передавались. Repo configs, frozen split/cohort/preproc и
все201 content SHA проверены на Pod до benchmark. Репозиторий public clone/fetch;
GitHub tokens в repo не сохранялись. SSH использовал только заданный RunPod key,
IdentitiesOnly=yes, StrictHostKeyChecking=yes; passphrase введена локально в agent.
Private key не выводился, не копировался и не отправлялся.

Result archive **31 files**, 11,956,542 bytes, SHA256
`6dc78afef3f645134424d5035da0a12908c88d32a35d9a58954b74b2abbef8f5`.
Audit archive9 files, SHA256
`549194d82e0807f69b38fbae0ca83bf9d6ad48f5ad436ccabdd2b0a2c95d6e94`.
Archive SHA, manifest SHA и все40 file SHA совпали после download на E.
Best/last checkpoints успешно загружены CPU weights_only: epoch18/30,488993params.

Best5,954,592 bytes SHA `e6b28f1b87d581fde2dbc2e055ec58f0353c98c913f62c10b96ba3338690e8f0`;
last5,965,984 bytes SHA `080c57f1471bc6b93d268a28fc37a702353bd454f4949af3119cbc523f9e5489`.
Сохранены history, config, actual environment/pip freeze, resource samples,
per-patient/group metrics, provenance и9 original-grid predictions.
Резерв E80 decimal GB соблюдён (~277.7 GB free), ничего автоматически не удалено.
Pod можно остановить; автоматическая остановка Pod этим кодом не выполнялась.

## Артефакты и воспроизведение

Small reports:
[GPU summary](../metadata/pediatric/heart_gpu_baseline_v1.json),
[prediction QA](../metadata/pediatric/heart_prediction_qa_v1.json),
[transfer status](../metadata/pediatric/heart_runpod_preparation_v1.json).
Все NIfTI/weights/meshes/screenshots/logs вне Git под CLI-selected data root.

```powershell
$dataRoot = 'E:/3d-heart-data/pediatric_ct_heart'
$resultsRoot = "$dataRoot/runpod_results/gpu_full_v1_20261006_io"
$testPredictions = "$resultsRoot/experiments/heart_baseline_v1/gpu_full_v1_20261006_io/test"
python -m heart3d.ml.bundle verify --root $resultsRoot --manifest "$resultsRoot/bundle_manifest.json"
python -m scripts.summarize_pediatric_gpu_run --results-root $resultsRoot --run-name gpu_full_v1_20261006_io --output-json metadata/pediatric/heart_gpu_baseline_v1.json --figure-dir "$dataRoot/experiments/report_recheck"
python -m scripts.view_pediatric_ct_qa --data $dataRoot --predictions $testPredictions --reviews metadata/pediatric/heart_cohort_reviews_v1.json --port 8766
# QA outputs требуют нового каталога; не перезаписывать существующий.
python -m scripts.qa_pediatric_predictions --data $dataRoot --predictions $testPredictions --output "$dataRoot/experiments/prediction_qa_recheck"
```

Repeat training — только как новый отдельно согласованный run, не автоматическое
продолжение. Pin training commit7c93cb9; install requirements-runpod (без torch),
проверить CUDA4090/persistent mount и SHA bundle, затем:

```bash
/workspace/venvs/heart/bin/python -m scripts.run_pediatric_gpu --config configs/pediatric_heart_runpod_v1.json --data /workspace/pediatric_ct_heart --run-name NEW_UNIQUE_RUN --export /workspace/exports
```

Frozen v1 split/cohort/preprocessing не менять. Уже просмотренный test теперь не
является новым blind set для следующих изменений. Следующий experimental version:
уточнить OAR scope с экспертом, анализировать continuity/background по train/val,
зафиксировать метод до новой held-out оценки. Многоклассовая сегментация, диагнозы,
healthy/CHD reasoning, Atlas fitting и CT+MRI joint training остаются вне v1.
