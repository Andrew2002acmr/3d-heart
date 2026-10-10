# CHD68: resolution384 v2 — подготовка следующего эксперимента

## Цель и статус

Проверить влияние увеличения XY-входа с 256×256 до **384×384** на сегментацию
семи структур, прежде всего AO/PA. Остальные параметры прежнего baseline сохранены.
Это public CHD68 pilot, не подтверждённая возрастная когорта 0–17 лет.
Подготовка выполнена локально; **GPU benchmark, full training и test v2 не запускались**.
Pod пользователя выключен, подключения к прежнему endpoint не выполнялись.

Исходный [результат v1](cardiac_chd68_results_v1.md) остаётся неизменным.
Его уже просмотренные test failures послужили основанием для гипотезы:
тот же test не является совершенно новым confirmatory holdout последующих вариантов.
В текущем этапе test pixels не открывались; они исключены из bundle,
новая конфигурация запрещает partition test. Подбор — только train/validation;
для независимого подтверждения потребуется новая проверяемая внешняя выборка.

## Протокол: один изменяемый фактор

| Параметр | v1 | resolution384 v2 |
|---|---|---|
| Input XY | 256×256 | **384×384** |
| Источник pixels | original CHD68 NIfTI | те же оригиналы, не upsample cache256 |
| Frozen split | 48 train / 10 validation / 10 test | тот же split; в комплекте только 48/10 |
| Context | 5 slices, indices −2/−1/0/1/2 | без изменения |
| Clip / normalization | train48 fit [0,2015] / linear [-1,1] | тот же frozen fit, без refitting |
| Image / mask interpolation | linear / nearest XY | без изменения |
| Z / crop | no Z resampling / full-FOV fit-pad | без изменения |
| Model | own 2.5D U-Net 16/32/64/128, GN8 | та же, **489112 parameters** |
| Target | background + LV/RV/LA/RA/MYO/AO/PA | те же labels; >7/pad ignore255 |
| Initialization | scratch | scratch; v1 weights только для reference inference |
| Batch / epochs | 16 / 30 | 16 / 30; full run пока не запущен |
| Optimizer / LR | Adam / fixed0.001 | без изменения |
| Loss | CE + own foreground Soft Dice | без изменения |
| Augmentation | rotation±5°, scale±5%, shift±.03, noise.01; no flips | без изменения |
| Seed / workers / precision | 20261009 / 0 / FP32, TF32 off | без изменения |

Sampling использует SHA-проверенные v1 positive/negative indices:
**12381 train samples/epoch, 774 batches при batch16**.
Изменение foreground extent после nearest resampling найдено только у validation
ct_1033, slice48; оно записано в summary. Train slice selection не менялся.
Body/heart crop, denoising, larger context, class reweighting, LR scheduler
не добавлялись: это отдельные последующие эксперименты.

HU/physical scale release остаются неподтверждёнными. Не вводились mm spacing,
physical context или HD95/ASSD mm. Клинические НИИ CT не используются и не передаются в cloud.

## Реально подготовленные данные и проверки

Root задаётся CLI; на текущем ПК E:/3d-heart-data/pediatric_ct_heart.
Новый cache: preprocessing/cardiac_chd68_resolution384_v2.
**58 cases, 174 prepared files, 10924627741 bytes (10.925 GB)**.
Source SHA, CT/GT shape/affine/RAS, finite values, integer labels проверены;
ни один из семи классов не исчез после resampling. Original grid/inverse transform
сохранены в provenance. Старые caches, sources и результаты не перезаписывались.

[Prepared summary](../metadata/pediatric/cardiac_chd68_resolution384_preparation_v2.json).
[CPU smoke](../metadata/pediatric/cardiac_chd68_resolution384_smoke_v2.json):
три optimizer updates на двух fixed train samples ct_1001/ct_1002.
Input 2×5×384×384, output 2×8×384×384, target 2×384×384.
Loss **3.3290 → 3.0915 → 2.9850**, finite gradients, weights changed.
Scratch weights не сохранялись. Это technical sanity, не quality и не GPU timing.

**179 tests passed**, 322 прежних skimage/NumPy warnings; bash -n passed.
Новые regression tests: one-factor contract, sealed test даже при отсутствующих
test sources, original-pixel preparation, fixed slice sampling, ignore255,
archive SHA round-trip, validation-only reference и восстановление probabilities384 на несquare original grid без перестановки осей.
Первый полный pytest collection требовал прежнего pydicom через
PYTHONPATH=D:/codexProjects/3d-hearts/data/pediatric_audit/python_deps;
после подключения весь suite прошёл. Новые пакеты не устанавливались.
Первый smoke до завершения train cache остановился на отсутствующей provenance;
после подготовки48train существующий checker выполнен успешно.

Визуально проверены ct_1001/ct_1002: original/256/384 на одинаковых axial indices
25/50/75% v1 positive range, единые colors/window. Систематического смещения на этих
panels не обнаружено. Это preprocessing QA, не predictions.
Helper scripts/qa_cardiac_resolution.py; images на E:
experiments/cardiac_chd68_resolution384_v2/preprocessing_QA.

## Комплект для RunPod

Archive: training_bundles/cardiac_chd68_resolution384_v2.tar.gz,
receipt: cardiac_chd68_resolution384_v2.tar.receipt.json.
**7847032263 bytes (7.847 GB), 217 payload files, unpacked11.733 GB.**
SHA-256:
2135116d4e9bc770cfbedb948a0619f30b19d57ca27f9f0a21976caa385b4d02.

Внутри train/validation384, их provenance, original10validation GT,
prepared10validation v1 и SHA-проверенный v1 best checkpoint epoch23.
Это минимальное дополнение для original-grid comparison, не инициализация обучения.
Нет raw CT/DICOM/RTSTRUCT, test arrays/GT, клинических источников, last checkpoint v1
или новых predictions. Шесть repository configs/manifests сверяются по SHA.

[Readiness summary](../metadata/pediatric/cardiac_chd68_resolution384_readiness_v2.json)
содержит receipt, storage budget и результат полной потоковой проверки payload/CRC.
На E резерв минимум80GB соблюдён. Для Pod учитывать отдельно фактическую quota:
archive+unpack+environment+10GBreserve требуют около32GB; разумно volume40GB или больше.
df общего FUSE pool не доказывает собственную quota.

Повтор подготовки использует scripts.prepare_cardiac_resolution prepare/bundle
с --config, --baseline-config, --data, --reference-checkpoint, --summary
и --archive/--manifest соответственно. Existing output автоматически не перезаписывается.
Config/preprocessing/split v1 не менять.

## После включения Pod

Получить актуальные SSH Host/Port. Использовать только D:\.ssh\id_ed25519_runpod,
IdentitiesOnly=yes, без передачи private key на сервер. Проверить RTX4090,
torch CUDA, persistent /workspace, nproc/free и quota до передачи bundle.
Для сопоставимости желательно окружение v1 torch2.8.0+cu128/CUDA12.8;
другие версии фиксировать как ограничение сравнения, CPU torch на Pod не устанавливать.

Новый checkout /workspace/3d-heart-resolution384-v2, та же feature-ветка.
Передать archive+receipt; сохранить git rev-parse HEAD.
Не обновлять checkout между benchmark и train: receipt проверяет точный commit/config/cache.

~~~bash
cd /workspace/3d-heart-resolution384-v2
git branch --show-current
git status --porcelain
git rev-parse HEAD
bash scripts/runpod_cardiac_resolution_benchmark.sh \
  /workspace/incoming/cardiac_chd68_resolution384_v2.tar.gz \
  /workspace/incoming/cardiac_chd68_resolution384_v2.tar.receipt.json
~~~

Script проверяет CUDA/mount, archive SHA/size, все payload/repository hashes;
unpack в НОВЫЙ /workspace/cardiac_chd68_resolution384_v2, stage baseline validation manifest.
Только **5 warmup + 50 measured batches + 20 fixed-train sanity updates**.
Не запускает full training/test. Измерить sec/batch, VRAM/RSS и оценку стоимости;
при failed gate сначала изучить причину. Оценка длительности v2 ещё отсутствует.

После successful GPU receipt и решения о полном запуске:

~~~bash
source /workspace/venvs/cardiac_chd68_resolution384_v2/bin/activate
DATA_ROOT=/workspace/cardiac_chd68_resolution384_v2
python -m heart3d.ml.cardiac_train train \
  --config configs/cardiac_chd68_resolution384_v2.json --data "$DATA_ROOT" \
  --run-name gpu_full_resolution384_v2 \
  --benchmark "$DATA_ROOT/experiments/cardiac_chd68_resolution384_v2/gpu_benchmark_resolution384_v2/summary.json" \
  --allow-full-training
~~~

## Честное validation-сравнение

Нельзя сравнивать prepared-grid Dice256 с Dice384 как одинаковое измерение.
После обучения оценить оба best checkpoint на **одной original release grid**
тех же10validation cases. Каждый best по-прежнему выбирается mean-case foreground
validation Dice на своей prepared grid; отдельно вычисляются original-grid метрики.

~~~bash
BASELINE_CKPT="$DATA_ROOT/remote_runs/cardiac_chd68_v1_full_20261009/artifacts/cardiac_chd68_v1/checkpoints/cardiac_chd68_v1/gpu_full_v1_20261009/best.pt"
python -m heart3d.ml.cardiac_train evaluate \
  --config configs/cardiac_chd68_runpod_v1.json --data "$DATA_ROOT" \
  --checkpoint "$BASELINE_CKPT" --partition validation \
  --output "$DATA_ROOT/evaluation/validation_original_baseline_v1"
python -m heart3d.ml.cardiac_train evaluate \
  --config configs/cardiac_chd68_resolution384_v2.json --data "$DATA_ROOT" \
  --checkpoint "$DATA_ROOT/checkpoints/cardiac_chd68_resolution384_v2/gpu_full_resolution384_v2/best.pt" \
  --partition validation --output "$DATA_ROOT/evaluation/validation_original_resolution384_v2"
~~~

Primary: paired per-case change macro foreground Dice.
Также Dice/IoU/precision/recall всех структур, особенно AO/PA; visual failure review
и continuity в3D. Smoothing/component removal или правка GT не применяются.
Один запуск не доказывает устойчивого эффекта. Новый test evaluation не запускать
для выбора параметров. Все future best/last/history/config/environment и validation
predictions/metrics скачать на E с SHA-проверкой до остановки Pod.
Прирост качества v2 и время GPU обучения пока **не измерены**.

Финальная проверка archive: **217/217 payload SHA-256 и gzip CRC passed**.
Свободно на E после упаковки: **225.06 GB**, резерв80GB соблюдён.


## Разрешён новый full run, preflight — 2026-10-10

Пользователь включил Pod и разрешил обучение resolution384. Актуальный endpoint
213.173.109.80:15827, root, только прежний ключ. Старый port12476 закрывал SSH
до authentication; port15827 publickey authentication успешно. Для нового endpoint
host key сохранён, последующие connections strict=yes. Windows ssh-keyscan оказался
несовместим с предложенным sntrup761 KEX; обычный ssh curve25519 подключился успешно.

RTX4090/torch2.8.0+cu128/CUDA подтверждены; persistent /workspace mount подтверждён.
Прежние v1 repos/data сохранены. На volume по du занято около18.41GB.
Новый archive7.847GB + unpack11.733GB +10GBreserve дают минимум48.0GB
до дополнительного environment/results. Для сохранения старых данных нужен volume
минимум50GB, лучше60GB. df общего FUSE pool не показывает личную quota.
Вопрос о выделенном размере отправлен пользователю; mass transfer/GPU benchmark/
full training до подтверждения capacity не запускались. Данные не удалялись.

Готов scripts/runpod_cardiac_resolution_full.sh: pinned commit, exclusive lock,
refusal of reused prefix/output, train30scratch, v1/v2 original-grid validation,
paired metrics, SHA export. Test не оценивается. Comparator проверяет одинаковые
split/cohort/cases/GT voxel counts и original grid. 185 tests passed;17focused passed;
bash -n passed. Actual execution commit будет зафиксирован при benchmark/launch;
не обновлять checkout между ними. [Preflight/launch metadata](../metadata/pediatric/cardiac_chd68_resolution384_launch_v2.json).
