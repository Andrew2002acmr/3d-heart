# CHD68: результат resolution384 v2

## Вывод — 2026-10-10

Увеличение XY-входа **256×256 → 384×384** в этом запуске не улучшило общий
результат: на тех же 10 validation cases, на одинаковой original release grid,
mean-case macro foreground Dice **0.797833 → 0.795231**, paired delta
**−0.002602 (−0.26 процентного пункта)**. Улучшились 4 случая, ухудшились 6.
PA выросла **0.6055 → 0.6340 (+2.85 п.п.)**, остальные шесть средних Dice снизились.
**Сохраняем v1 как текущий общий baseline.** V2 остаётся исследовательским вариантом;
одного запуска недостаточно для вывода об устойчивом эффекте resolution.

Обучение 30 epochs, paired validation и export завершены. Best v2 — epoch10,
prepared-grid validation Dice0.795064; v1 reference — best epoch23.
Все **64/64 critical files** скачаны на E и проверены SHA-256, включая best/last,
history, config, environment, provenance и обе группы validation predictions.
Test v2 не запускался. GPU-процесс завершён; **Pod можно остановить**.

Это публичный CHD68 pilot. Возраст, независимость patient identity и физический
масштаб release не подтверждены. Результат не является проверкой точности на
новорождённых или клинической валидацией. Частные НИИ CT не передавались в cloud.

## Протокол и воспроизводимость

Изменён только размер XY. Cache384 построен из original NIfTI, без upsample256.
Frozen split сохранён: **48 train / 10 validation / 10 sealed test**, seed20261009.
Test arrays/GT отсутствуют в training bundle, config запрещает partition test.
Гипотеза возникла после просмотра ошибок v1 test; будущая независимая проверка
потребует новой внешней выборки. Здесь сопоставляются только validation с validation.

Train-only intensity clip[0,2015] и linear normalization[-1,1] сохранены без refit.
Это source intensity, не подтверждённые HU. Full-FOV fit/pad, CT linear XY, GT nearest;
no GT crop, no Z resampling. Контекст: **5 index slices −2/−1/0/1/2**, не миллиметры,
явная edge replication. Original grid/inverse transform записаны в provenance;
probabilities восстановлены на original grid **до argmax**.

Свои **489112 parameters**, 2.5D U-Net: Conv3×3/GroupNorm8/ReLU blocks,
channels16/32/64/128, 3 downsampling stages, bilinear decoder и skips;
8 logits: background + LV/RV/LA/RA/MYO/AO/PA. Scratch initialization;
v1 weights использованы только для reference inference.
Loss: CE + собственный foreground Soft Dice, ignore255. Adam, fixedLR0.001,
batch16, 30 epochs, FP32, TF32 off, workers0. Augmentation train-only:
rotation±5°, scaling±5%, intensity shift±0.03, noise0.01, no flips.
Lazy/memmap strategy; **12381 train slices / 774 batches per epoch**.
Slice sampling прежний. Validation ct_1033 slice48 изменила foreground extent
при nearest resampling; это записано при подготовке, протокол ради результата не менялся.

Execution checkout `/workspace/3d-heart-resolution384-v2`, branch
`feature/pediatric-heart-segmentation`, commit
`948c4bb70da7c9f494e7dbfe84bcf09c8446271b`.
Позднейшие локальные docs/QA/viewer commits не обновляли execution checkout.
Config `configs/cardiac_chd68_resolution384_v2.json`; frozen split
`configs/splits/cardiac_chd68_v1.json`. Их SHA, cache/cohort/preprocessing SHA
сохранены в [машинном отчёте](../metadata/pediatric/cardiac_chd68_resolution384_results_v2.json).

## Сравнение на original grid

Сначала оба best checkpoint оценены на **одних и тех же 10 validation cases**.
Comparator проверил split/cohort IDs, original grids и GT voxel counts.
Macro считается по reference-present foreground classes, без empty-empty reward;
затем mean по cases. Все 7 классов присутствуют во всех 10 случаях.
Prepared-grid score выбора checkpoint отдельно от приведённых ниже метрик.
Ни smoothing, ни component removal, ни исправление GT не применялись.

| Структура | Dice v1 | Dice v2 | Δ, п.п. |
|---|---:|---:|---:|
| LV | 0.8722 | 0.8594 | -1.28 |
| RV | 0.8185 | 0.8090 | -0.95 |
| LA | 0.7893 | 0.7821 | -0.72 |
| RA | 0.8464 | 0.8365 | -0.99 |
| MYO | 0.8537 | 0.8488 | -0.50 |
| AO | 0.7992 | 0.7968 | -0.24 |
| PA | 0.6055 | 0.6340 | +2.85 |

Средние по cases; пары в каждой ячейке: **v1 / v2**.
Mean IoU вычислен по individual cases, не преобразованием mean Dice.

| Структура | IoU | Precision | Recall |
|---|---:|---:|---:|
| LV | 0.7780 / 0.7606 | 0.8435 / 0.8641 | 0.9068 / 0.8627 |
| RV | 0.6994 / 0.6878 | 0.8645 / 0.8660 | 0.7986 / 0.7755 |
| LA | 0.6630 / 0.6553 | 0.8165 / 0.8190 | 0.8000 / 0.7817 |
| RA | 0.7414 / 0.7288 | 0.8797 / 0.8304 | 0.8277 / 0.8609 |
| MYO | 0.7475 / 0.7406 | 0.8621 / 0.8549 | 0.8555 / 0.8559 |
| AO | 0.6717 / 0.6680 | 0.8076 / 0.7872 | 0.8124 / 0.8236 |
| PA | 0.4654 / 0.4925 | 0.6108 / 0.7005 | 0.6514 / 0.6249 |

| Case | Macro Dice v1 | Macro Dice v2 | Δ, п.п. |
|---|---:|---:|---:|
| ct_1027 | 0.6498 | 0.6513 | +0.15 |
| ct_1028 | 0.8347 | 0.8294 | -0.53 |
| ct_1033 | 0.8700 | 0.8649 | -0.51 |
| ct_1036 | 0.8797 | 0.8845 | +0.48 |
| ct_1046 | 0.8650 | 0.8628 | -0.22 |
| ct_1056 | 0.7735 | 0.7370 | -3.65 |
| ct_1079 | 0.7901 | 0.8001 | +0.99 |
| ct_1099 | 0.7833 | 0.7720 | -1.13 |
| ct_1125 | 0.7525 | 0.7747 | +2.22 |
| ct_1127 | 0.7798 | 0.7757 | -0.40 |

## Ошибки и визуальная QA

PA остаётся самым слабым классом. Её mean precision вырос0.6108→0.7005,
но recall снизился0.6514→0.6249: улучшение overlap не означает восстановления всех ветвей.
В ct_1027 PA Dice вырос0.3117→0.5621, но LV/LA/RA ухудшились;
RV recall v2 лишь0.4800, LA precision0.3462 — остаются крупные ошибки камер.
В ct_1099 PA Dice **0.2999→0.2398**, precision0.1608, recall0.4711:
predicted PA объём примерно2.93×GT, одновременно есть FP и пропуски.
В ct_1125 PA Dice0.3054→0.3381, но recall0.2285: значительная часть reference пропущена.
Наибольшее ухудшение общего score — ct_1056: **0.7735→0.7370**.
Наибольшее улучшение — ct_1125:0.7525→0.7747, при сохраняющейся слабой PA.

Воспроизводимый sampling: axial20/50/80% GT-positive extent и central coronal/sagittal.
GT используется только для выбора review slices после inference, не как input crop.
Best/middle-rank/worst выбираются по v2 validation: **ct_1036 / ct_1079 / ct_1027**.
Middle rank — 6-й из10, не арифметическая median двух средних score.
CT/GT/v1/v2 показаны на одинаковых slices/window, с легендой семи структур.
Original source и prediction SHA/grid проверяются перед rendering.

Изображения и QA_summary находятся на E:
`remote_runs/cardiac_chd68_resolution384_v2_20261010/visual_QA`.
Все три panels просмотрены: у ct_1036 центральный overlap хороший, но сохраняются
ошибки границ и ложная AO-метка на периферии axial80%; у ct_1079 остаются
внутренние фрагменты неверных labels. У ct_1027 видны крупная путаница камер
и неровные межсрезовые границы на coronal/sagittal у обеих моделей.
[QA sampling и SHA изображений](../metadata/pediatric/cardiac_chd68_resolution384_QA_v2.json).
Визуальный осмотр компактных panels не заменяет полный послойный/3D review.
HD95/ASSD в mm не вычислялись из-за неподтверждённого физического масштаба.

## Фактическое время и ресурсы

RTX4090 24564MiB, torch2.8.0+cu128/CUDA12.8, driver580.95.05;
32 logical CPUs, RAM124GiB, persistent `/workspace`, выделено120GB пользователем.
Benchmark:5warmup +50measured batches +20fixed-train sanity updates.
**0.34384s/batch, 46.53samples/s**; train-only прогноз7983.98s/30epochs
исключал validation, checkpoint I/O и contention. Sanity loss3.3290→1.8320,
finite gradients, weights updated; это техническая проверка, не качество сегментации.

Фактический полный trainer wall time **9194.41s =153.24min (2h33m14s)**,
включая epoch validation и сохранение checkpoint. Сумма training_seconds8676.98s,
epoch validation513.19s; mean train epoch289.23s.
От запуска runner04:48:22UTC до готового export07:26:58.927UTC — около158m37s;
это включает две original-grid validation evaluations и упаковку.
V1 trainer wall117.74min: v2 занял примерно на30.2% больше времени.

Peak process RSS **2.69GB**; CUDA allocated **4.31GB**, reserved **6.36GB**.
Mean processCPU256.1% =2.56 logicalCPU equivalents /8.0% всех32.
Минимум system availableRAM122.47GB. RSS не включает OS file cache.
Train loss1.1649→0.0787, лучшие validation weights на epoch10, дальнейший val score
колебался ниже максимума. Такая динамика совместима с переобучением; она не доказывает
его единственную причину. NaN/Inf и failed stages не обнаружены.
CUDA CE предупредил об отсутствии deterministic implementation;
битовая воспроизводимость GPU не заявляется, seed/provenance фиксируются.

## Проверки и сохранность

Перед запуском: 185 tests passed, remote focused11passed, bash syntax/guard passed.
Archive входа7.847GB и217/217payload SHA проверены. Ни private data, ни test payload
не передавались. Full runner использует exclusive lock и отказывается повторно запускать
существующий prefix/output. Протокол и execution commit во время run не менялись.

Results archive **25332556bytes**, 64files, unpacked38867960bytes.
SHA-256 archive:
`1b01cf066c18edee13affeb75fa5a6ec6b7651c2c41ebaeab95d316650698ef4`.
Manifest SHA и gzip CRC проверены, safe unpack повторно сверил все64payload.
Best/last checkpoint hashes дополнительно совпали с training_summary:

- best.pt6006560bytes: `fc24b90d6a7cf3a8f14130e44b82e78bcff7de7fa30d0cbb84c00474dd8760df`
- last.pt6143520bytes: `17a8cfd502992d78310ee2c5b4af16747315ee3f8d47df6080143be82039dd9d`

Local verified backup:
`E:/3d-heart-data/pediatric_ct_heart/remote_runs/cardiac_chd68_resolution384_v2_20261010`.
Там сохранены оба checkpoint, обе группы predictions/provenance, config/split,
metrics/comparison/history, environment/pip freeze, resource samples и compute log.
Receipt `results_SHA_verified_20261010.json`; свободно около225GB, резерв80GB соблюдён.
В Git только docs и небольшие публичные summaries; images/weights/cache/predictions на E.

Проверка и распаковка:

~~~powershell
python -m scripts.verify_training_archive --archive RESULTS.tar.gz --receipt RESULTS.tar.receipt.json
python -m heart3d.ml.bundle unpack --archive RESULTS.tar.gz --destination NEW_ARTIFACT_DIR --reserve-GB 80
python -m scripts.qa_cardiac_validation_pair --data DATA_ROOT --baseline V1_VALIDATION --variant V2_VALIDATION --output NEW_QA_DIR
~~~

Полный runner: `scripts/runpod_cardiac_resolution_full.sh`; benchmark runner:
`scripts/runpod_cardiac_resolution_benchmark.sh`. Не повторять завершённый prefix.
RunPod после проверенного backup можно остановить; автоматически Pod не останавливался.

## Следующий шаг

Рекомендуем сохранить v1 и спланировать отдельный validation-only эксперимент
по train-derived image-only crop или большему/3D spatial context, с прежним split
и новым заранее записанным протоколом. Не добавлять denoising без проверки потери
мелких сосудов; не удалять компоненты агрессивно. Resolution alone не решила ошибки PA
и камер. Новое обучение, tuning и test evaluation в этом этапе не выполняются.
Для оценки на новых НИИ CT нужна отдельная локальная разметка/проверка: этот публичный
эксперимент не подтверждает точность на младенцах и не восстанавливает утраченную
из-за motion информацию. Полуручная коррекция остаётся необходимой частью проекта.
