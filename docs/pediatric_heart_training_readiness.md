# Pediatric Heart: готовность и завершённый baseline

Дата:2026-10-06. QA cohort60 / split42/9/9 заморожены, train-only preprocessing
и собственная2.5D U-Net488993params проверены. CPU readiness успешно завершён;
позднее пользователь разрешил один RunPod full experiment после GPU benchmark.
**GPU benchmark прошёл,30epochs завершены за 41.10min, test n9 оценён
один раз; mean Dice0.9194, HD95mean17.90mm/max70.65mm.**
Results SHAverified на E, best/last load verified; Pod можно остановить.
Доступ SSH исправлен, блокеров подключения больше нет.
Full local suite127passed; actualDICOM inference + viewer/3D QA также выполнены.

Технически полный baseline запускается воспроизводимо. Качество первого run
ещё недостаточно для clinical использования: islands/undersegmentation и небольшой
held-out требуют дальнейшей проверки. Новые runs автоматически не запускаются.
[Финальный GPU отчёт](pediatric_heart_runpod_v1.md),
[общий baseline](pediatric_heart_segmentation_baseline.md),
[small metrics](../metadata/pediatric/heart_gpu_baseline_v1.json).

## Исторический CPU readiness report

Следующие sections фиксируют состояние **до** последующего CUDA transfer/
validation/test caching и full training. Frozen selection/preprocessing не менялись.
Указанные CPU estimates и train-only benchmark относятся к этому прежнему этапу;
актуальные GPU timings/results приведены выше и в отдельном отчёте.

## Target и критерии до отбора

Target v1 — **исходный экспертный Heart OAR**, как определён RTSTRUCT источника.
Это общая Heart область; анатомическую полноту камер/сосудов не утверждаем.
[Публикация](https://pmc.ncbi.nlm.nih.gov/articles/PMC9090951/) подтверждает
экспертные OAR contours и исходную QA radiation oncologist. Точный cranial landmark
для Heart в статье не задан. Поэтому ровная planar граница сама по себе не ошибка:
она сохраняется как scope исходной аннотации, если references/planes корректны,
ROI непрерывен и scan не обрезает очевидную Heart область. Неоднозначность
аннотации оставляется `annotation_scope_review`, не достраивается автоматически.

`approved` означает техническую и визуальную пригодность для воспроизведения
source OAR protocol, не новую clinical annotation sign-off или healthy label.
Для всех пациентов healthy_status остаётся unknown.

Полный CT gate: исходные SOP inventory, возраст/ID/series/frame, IPP/IOP/spacing,
HU tags, regular grid, independently sorted GDCM/SimpleITK geometry + HU.
RT gate: original full RT, Heart definitions/nonempty contours, all source references,
FOV/plane residual, rasterization и independent VTK boundary agreement.

Review classes: `approved`, `coverage_incomplete`, `coverage_review`,
`annotation_scope_review`, `reference_failure`, `empty_roi`, `geometry_failure`,
`identity_review`.
Automatic technical pass никогда не означает visual approval. Scan face contact,
margin <=1 CT slice, внутренние пропуски или несколько components требуют review.
Необычный contour count, объём или spacing — признаки для внимательного просмотра,
а не автоматическое исключение по размеру/возрасту.

## Sampling

Заранее фиксирован seed 20261006. Начальные scanner квоты — existing preparation
config; replacements заранее упорядочены SHA256(seed|patient_id), round-robin strata.
Volume/красота маски не входят в selection. Metadata reference/empty failures
отсеиваются до full download. Цель 20/20/20 по рабочим age groups, не любой ценой.
Девять прежних engineering cases повторно проверяются; из будущего test исключены.

QA sampling v2: axial first−2, first, first+2, median active, last−2, last, last+2
(clamp к CT grid), native coronal/sagittal median active. Отдельные CT и
CT+mask panels, source contour overlay на axial. Original spacing aspect, без GT
interpolation. Скриншоты вне Git. Review описывает coverage, annotation scope,
alignment, причины и reviewer type.

## Фактическая полная QA

Получены **117 полных CT/оригинальных RTSTRUCT пар**: 9 прежних development
studies, 60 начальных кандидатов и 48 metadata-selected replacements только в
младших двух возрастных группах. Дополнительные два original RT предыдущего
reference audit не являются полными CT/RT парами. Данные, native review mosaics,
receipts и fingerprints находятся на E, в Git только обезличенные summaries.

| Gate / review | Число |
|---|---:|
| Полные CT/RT pairs | 117 |
| DICOM geometry/HU + независимый reader | 114 |
| Heart RTSTRUCT gate + независимый rasterizer | 114 |
| `approved` для исходного OAR target | 62 |
| `coverage_incomplete` | 50 |
| `coverage_review` | 1 |
| `geometry_failure` | 3 |
| `identity_review` | 1 |
| `annotation_scope_review` / reference failure / empty ROI в full-pair wave | 0 / 0 / 0 |

Все 114 технически успешных случаев визуально просмотрены на sampling v2;
3 geometry failures не обходились. Все 62 approvals имеют SHA256 report, native
QA image, original-grid CT и GT. Минимальный Dice двух способов rasterization
0.9999947272818347; nonboundary differences = 0. Это согласованность конвертации,
не точность ML. Full gate дополнительно проверяет оригинальные RT patient,
series, frame и SOP class. Геометрия соответствует опубликованной CT series;
полноту неопубликованной исходной acquisition доказать невозможно.

Review reasons: [heart_cohort_reviews_v1.json](../metadata/pediatric/heart_cohort_reviews_v1.json).
Case evidence: [heart_qa_audit_v1.json](../metadata/pediatric/heart_qa_audit_v1.json).

Три irregular-spacing failures: `376`, `37058120`, `EB1DCBAA`. `E03568A6`
остаётся coverage review из-за одного slice над ROI. Сходные `92891A2F` и
`176261A0` имеют разные HU voxels, но совпадающую age/geometry; клиническая
идентичность не установлена. Оставлен более ранний в заданной очереди `92891A2F`,
`176261A0` не используется ни в одной partition. Это precaution к repeat
acquisition leakage, а не утверждение о подтверждённом дубле.

Exact decoded HU fingerprint audit: **114**, точных duplicate groups **0**.
Public patient ID isolation и content fingerprint не доказывают отсутствие
повторных исследований человека под разными anonymized IDs.

## Целостность download и storage

**234 ZIP** проверены по download receipt SHA256, расхождений нет; downloader
проверяет ZIP CRC и существующие DICOM перед extraction без перезаписи различий.
Исторический overlap двух preparation jobs привёл к четырём incomplete/corrupt
cache files. Они сохранены отдельно, повторный download завершён и проверен;
исторические failures сохранены с отдельным resolution. Existing DICOM не
изменялись. Добавлен OS-exclusive lock общего data root для prevention.
Evidence: [source cache verification](../metadata/pediatric/heart_source_cache_integrity_v1.json),
[incident/resolution](../metadata/pediatric/heart_download_integrity_incident_v1.json).
Ничего с D автоматически не удалялось/переносилось. Резерв E — 80 decimal GB.
Wave budgets и подробные receipts находятся в external cache.

## Frozen cohort и split

Из 62 approved взяты первые 20 каждой age group в заранее объявленной metadata
очереди: **60**, возраст фактически 2–16 лет. Два дополнительных approved young
cases оставлены вне baseline v1. Маска/объём не используются для ранжирования.

Manifest: [heart_approved_cohort_v1.json](../metadata/pediatric/heart_approved_cohort_v1.json).
Frozen split: [pediatric_ct_heart_v1.json](../configs/splits/pediatric_ct_heart_v1.json).
Seed **20261006**, cohort manifest SHA256 записан в split, все public patient IDs
уникальны и partitions не пересекаются. Повторное создание v1 запрещено helpers;
review нельзя менять после freeze. Девять engineering IDs исключены из test.

| Partition | Пациентов | 2–5 / 6–11 / 12–17 | LightSpeed / SOMATOM / Revolution | Contrast evidence / unknown |
|---|---:|---|---|---|
| train | 42 | 14 / 14 / 14 | 29 / 12 / 1 | 39 / 3 |
| validation | 9 | 3 / 3 / 3 | 6 / 3 / 0 | 9 / 0 |
| test | 9 | 3 / 3 / 3 | 6 / 3 / 0 | 9 / 0 |

Revolution full coverage мало; в held-out его нет, поэтому scanner robustness для
него этим baseline не оценивается. Contrast evidence = reported agent, пустой
tag = unknown; подтверждённой non-contrast held-out группы нет. Reformat artifacts
сохранялись при технической пригодности; индивидуальный исходный reformat flag
не везде доступен и по шуму/spacing автоматически не присваивался.

## Train-only statistics

[heart_train_statistics_v1.json](../metadata/pediatric/heart_train_statistics_v1.json)
получен только по 42 train patients после freeze. HU — fixed XYZ stride [4,4,2],
voxel-weighted aggregate, 1 HU histogram bins; это sampled distribution,
не точные extrema всех voxel. Body для статистики = HU > −500, это не crop.
Review CT/GT hashes и native affine проверены до чтения. Ни один held-out volume
не открывался при fitting, cache preparation или benchmark.

| Train quantity | Min / median / max |
|---|---|
| Original XYZ shape | 512×512×71 / 512×512×285 / 512×512×1041 |
| XY spacing, mm | 0.390625 / 0.546875 / 0.976562 |
| Z spacing | 30 patients 2 mm; 12 patients 0.625 mm |
| XY FOV, mm | 200 / 280 / ~500 |
| Z FOV, mm | 142 / 506 / 696 |
| Source Heart OAR volume, ml | 133.04 / 343.48 / 743.52 |
| Heart fraction of full CT voxels | 0.389% / 0.844% / 2.928% |
| Heart axial extent, mm | 46 / 81 / 122 |

Все 42 train IOP = [1,0,0,0,1,0]. Original slices 17,180: positive 2,830,
negative 14,350. Sampled HU percentiles (0.5% / median / 99.5%):
CT −1000 / −902 / 706; body −475 / 24 / 969; Heart −121 / 156 / 322.

## Frozen preprocessing

[pediatric_heart_preprocessing_v1.json](../configs/pediatric_heart_preprocessing_v1.json)
фиксирует fitting patient IDs, split/cohort/statistics SHA256. HU clip
**[−1000,969]**, linear normalization [−1,1]: train CT 0.5th percentile сохраняет
air, train body 99.5th ограничивает яркие outliers и оставляет Heart intensity
range. Это воспроизводимый baseline выбор, не клинически оптимизированное окно.
QA window −150..250 применяется только к screenshots.

Full physical FOV fit/pad **256×256**, без body/GT crop. Effective XY spacing
~0.78125..1.953125 mm (median 1.09375), зависит от source FOV; small boundaries
теряют разрешение. Target Z **2 mm**, выбран по большинству native train CT;
image linear / mask nearest, исходный GT сохранён. Native orientation сохраняется;
unexpected orientation требует явного отдельного adapter, не silent flip.

Пять physical offsets **[−4,−2,0,2,4] mm**, центральный target. Neighbors выбираются
по physical positions с linear weights, отсутствующий context replicate-edge
с явными flags. Target-grid последняя позиция floor(span/2)*2; хвост original
grid <2 mm восстанавливается replicate-edge. Inverse transform сохраняет native
XYZ shape, RAS affine, pixel-center resize/pad и relative physical Z. Future
prediction probability linear обратно на original grid, затем threshold; GT не
перезаписывается. Unit tests проверяют identity, anisotropic inverse landmarks,
physical neighbors и неправильные grids.

Подготовлен **только train** disk-backed .npy cache, один patient за раз:
10,280 central positions, 1,742 positive и 8,538 negative. Training epoch —
all positive + столько же negative на patient (seeded, меняются с epoch),
**3,484 samples / 1,742 batch при batch=2**. Это GT-based train sampling,
не GT-guided crop; future val/test/inference используют все central positions.
LRU mmap открывает максимум два patients, volumes не собираются вместе в RAM.
Cache summary: [heart_train_cache_summary_v1.json](../metadata/pediatric/heart_train_cache_summary_v1.json).

Дополнительно просмотрен train-only mosaic по одному patient каждой age group:
06722123, 0D53FEEE, 1360E26F. Five-slice context и central GT совпадают визуально;
файлы находятся под external `experiments/heart_baseline_v1/preprocessing_qa/`.

## Model, loss и augmentation

Собственный [model.py](../heart3d/ml/model.py), никаких imported ready-made networks
или pretrained weights. Input BCHW=2×5×256×256, encoder **16/32/64/128**, три
MaxPool2d(2); каждый block две bias-enabled Conv2d(3,pad1) → GroupNorm(8) → ReLU.
Decoder bilinear align_corners=False до skip shape, concat + такой же block.
Conv2d(16,1,1) output logits, target B×1×H×W. **488,993 trainable parameters**,
1,955,972 bytes FP32 tensor payload. Odd spatial sizes поддержаны unit test.

[losses.py](../heart3d/ml/losses.py): BCEWithLogits, project soft Dice, combined.
Для p=sigmoid(logits), y binary, ε=10⁻⁶:

`BCE = −mean[y log(p) + (1−y) log(1−p)]` (численно stable PyTorch formulation).

`DiceLoss = mean_batch[1 − (2 sum(p y)+ε)/(sum(p)+sum(y)+ε)]`.

`Combined = BCE + DiceLoss` (оба weights=1). Empty-target slices сохраняются:
Dice сглажен ε; на них BCE даёт основной background signal. Variants протестированы,
трёх full loss experiments не было. Benchmark использует combined.

Train augmentation: rotation uniform ±5°, scale 0.95..1.05, normalized intensity
shift ±0.03 (~29.5 HU), noise σ=0.01 (~9.85 HU), clamp [−1,1]. Одинаковое spatial
преобразование всех пяти planes и target: image bilinear / target nearest.
Flips запрещены guard; val/test augmentation отключена. Seed/index/epoch определяют
sampling и augmentation; synthetic test проверяет shared spatial alignment.

## Короткий benchmark

Config: [pediatric_heart_baseline_v1.json](../configs/pediatric_heart_baseline_v1.json).
CPU, 6 intra-op threads, batch 2, workers 0, Adam lr=0.001, стандартные
betas/eps, scheduler отсутствует в этом коротком run. Seed 20261006;
torch deterministic algorithms. Только 5 warmup + 50 measured batches,
затем 20 fixed-train sanity steps на двух train patients без augmentation.
Нет epoch loop, validation/test evaluation или model-quality claims.
Bench entry point ограничивает measured batches ≤200 и sanity/warmup budget.
История, environment, config, resource samples и last_benchmark checkpoint на E;
многочасовой full training запрещён текущим заданием.

Run `cpu_b50_v1_20261006`, code revision `686fe95`, успешный exit 0.
Machine-readable [heart_cpu_benchmark_v1.json](../metadata/pediatric/heart_cpu_benchmark_v1.json).

| Измерение | Результат |
|---|---:|
| CPU | AMD Ryzen 5 4500, 6 physical / 12 logical |
| RAM installed / Python / torch | ~16 GiB / 3.14.4 / 2.14.1+cpu |
| Device / intra-op threads | CPU / 6; CUDA available=false |
| Measured batches / warmup / fixed sanity | 50 / 5 / 20 |
| sec/batch mean / median / p95 (data+augmentation+backward+Adam) | 0.4150 / 0.4137 / 0.4497 |
| samples/sec | 4.8191 |
| Peak sampled process RSS (0.2 s sampling) | 623,996,928 bytes (~624 MB, 595 MiB) |
| Mean process CPU / share of logical machine | 579.1% / 48.3% |
| Minimum available system RAM | 7,694,491,648 bytes (~7.69 GB) |
| Maximum system RAM utilization | 54.9% |
| Checkpoint size | 5,947,446 bytes (~5.95 MB) |
| Extrapolated training epoch, 1,742 batches | 722.96 s (~12.05 min) |
| Extrapolated 30 training epochs | 21,688.88 s (~6.02 h) |

CPU 100% соответствует одному logical core; benchmark после завершения
conversion/tests. RSS sampled working set, не OS-wide memory и не точный
непрерывный high-water mark; file cache отдельно. Radeon backend не проверен,
эксперимент автоматически на GPU не переключался. 80 GB E reserve выполнен:
после cache/benchmark свободно ~281.48 GB.

**6.02 h — только training compute/data budget**, linear extrapolation этого run.
Validation all-slices, checkpoint I/O, загрузка машины и future original-grid
evaluation отдельно, поэтому полный wall time будет больше; они здесь не измерены.
Это не фактические 30 epochs и не гарантия постоянной скорости на долгом запуске.

Sanity: no NaN/Inf, ненулевые finite gradients, weight absolute delta 3261.254.
На двух центральных slices train patients 06722123/0D53FEEE, без augmentation:
combined loss **1.454400 → 1.021004 после random benchmark → 0.878711 после
20 fixed steps**. Это техническая обучаемость, не scientific Dice и не оценка
generalization. Всего 75 optimizer steps; loss variants full runs отсутствуют.
Benchmark weights нельзя принимать за final baseline или использовать как
pretrained initialization: следующий baseline должен стартовать с нуля по seed.

## Проверки и ограничения

**120 tests passed**, полный suite, 320 прежних NumPy/scikit-image deprecation
warnings. Проверены model forward/shape/gradients, Dice/BCE, seed, physical
neighbors/edge context, shared augmentation, lazy indexing, patient isolation,
manifest hashes и inverse restoration. Existing DICOM/geometry/mesh tests проходят.
Дополнительный train-only preprocessing mosaic просмотрен; receipt:
[heart_train_preprocessing_review_v1.json](../metadata/pediatric/heart_train_preprocessing_review_v1.json).

Крупные artifacts вне Git, исходная D-копия и основной checkout не менялись.
Frozen split не менялся. Все healthy_status unknown. Coverage selection ограничивает
generalization: source OAR boundaries могут завершаться внутри анатомического
сердца по исходному protocol. Target v1 не whole-chamber/vascular segmentation.
Техническая visual QA не заменяет нового clinician annotation sign-off. Scanner
и contrast held-out ограничения выше; ages 17 фактически нет. Диагнозов нет.

**Технически data/model готовы к следующему full baseline для source Heart OAR**:
CPU/RAM/storage достаточны, оптимизация и sanity learning работают. Полный run
потребует отдельного разрешения пользователя. Следующий этап — full train/validation
loop с best/last checkpoint, original-grid metrics и failure review; текущий
entry point намеренно ограничен benchmark. Затем один test evaluation без выбора
параметров по test. Диагнозы, healthy classifier, Atlas fitting и CT/MRI mixing
не входят в этот baseline. Сейчас val/test cache, test metrics/predictions и
GT/Prediction surfaces отсутствуют в соответствии с запретом.

## Воспроизведение

Из worktree в Python environment проекта, после установки pinned requirements:

```powershell
python -m pip install -r requirements-lock.txt -r requirements-pediatric.txt -r requirements-ml.txt
$dataRoot = 'E:/3d-heart-data/pediatric_ct_heart'
python -m scripts.verify_pediatric_source_cache --data $dataRoot --out "$dataRoot/cache/source_integrity_recheck.json"
python -m heart3d.ml.analyse --data $dataRoot --split configs/splits/pediatric_ct_heart_v1.json --cohort metadata/pediatric/heart_approved_cohort_v1.json --out "$dataRoot/experiments/train_statistics_recheck.json"
python -m heart3d.ml.prepare --config configs/pediatric_heart_baseline_v1.json --data $dataRoot --partition train
python -m scripts.qa_pediatric_preprocessing --config configs/pediatric_heart_baseline_v1.json --data $dataRoot --out "$dataRoot/experiments/heart_baseline_v1/preprocessing_qa/train_context_recheck.png"
# Only a bounded benchmark; choose a NEW run name to preserve current artifacts.
python -m heart3d.ml.benchmark --config configs/pediatric_heart_baseline_v1.json --data $dataRoot --batches 50 --run-name cpu_b50_recheck_unique
python -m pytest -q --basetemp "$dataRoot/temporary/pytest_recheck_unique"
```

Текущий environment использует Python executable
`D:/codexProjects/3d-hearts/.venv/Scripts/python.exe`; external pydicom подключён
через PYTHONPATH `D:/codexProjects/3d-hearts/data/pediatric_audit/python_deps`.
Новый environment ставит pinned pediatric dependencies обычным pip.
Нужен existing public inventory `<data_root>/cache/tcia_series_v1.json` для
повторного full-source download. Восстановить выбранные case IDs можно из audit
и predeclared queue через `prepare_pediatric_cohort.py`, но **не refresh approved
reports после freeze**, поскольку их SHA входит в immutable manifest.
Raw sources/CT/GT не перезаписывать. Baseline helper reuse cache проверяет hashes;
при изменении preprocess нужен новый experiment/version, не silent overwrite v1.
Checkpoint/history/config/environment: external
`experiments/heart_baseline_v1/cpu_b50_v1_20261006/` и
`checkpoints/heart_baseline_v1/cpu_b50_v1_20261006/last_benchmark.pt`.
