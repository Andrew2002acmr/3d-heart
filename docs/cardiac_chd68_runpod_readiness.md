# Подготовка multiclass CHD68 v1 к RunPod RTX 4090

Дата: 09.10.2026. Ветка feature/pediatric-heart-segmentation.
Обновлено после миграции Pod: SSH, RTX4090/CUDA и persistent /workspace проверены.
Минимальный публичный bundle передан и проверен, короткий GPU benchmark выполнен.
Полное multiclass training и одна test evaluation завершены; critical results на E: проверены SHA-256.
Все тяжёлые локальные данные на E:, исходные CT и старый Heart baseline не изменены.

## Готовый эксперимент

Цель — собственная segmentation model: background/LV/RV/LA/RA/MYO/AO/PA.
Используется только полный публичный CHD68 release, 68 пар с прошедшим pair-grid gate.
ImageCHD не добавлен повторно. НИИ CT не используются для обучения или настройки.
Исходные labels >7 сохраняются и исключаются из loss/метрик через ignore_index=255.
Статус Normal не превращается в confirmed healthy; классификатора диагнозов нет.

Frozen split **48 train / 10 validation / 10 test**, seed 20261009.
ct_1001/1002/1004/1008/1072 — development, только train; в test не входят.
Split по released case/group, без смешивания slices между partitions.
Отдельные case IDs и одинаковые image hashes не могут пересекаться между partitions.
Между различными case IDs независимость пациентов не подтверждена — split не выдаётся
за независимо верифицированный patient-level cohort. После уточнения metadata можно
планировать следующий протокол; v1 во время эксперимента не менять.

Это **pilot на публичном release**, а не подтверждённый результат на детях 0–17.
Age linkage, physical scale и HU release не подтверждены. Для обучения segmentation
в index coordinates они не являются обязательными входами. Age/мм/неонатальную
точность этот эксперимент не доказывает. Запрос авторам остаётся отдельным шагом.

## Train-only preprocessing и данные

[Config](../configs/cardiac_chd68_runpod_v1.json),
[fitted preprocessing](../configs/cardiac_chd68_preprocessing_v1.json),
[split](../configs/splits/cardiac_chd68_v1.json),
[cohort](../metadata/pediatric/cardiac_chd68_cohort_v1.json),
[train statistics](../metadata/pediatric/cardiac_chd68_train_statistics_v1.json).

Глобальные percentiles 1/99.5 на 65536 воспроизводимых равномерных samples
каждого из 48 train случаев, с одинаковым весом каждого случая:
**source intensity clip [0,2015] → [-1,1]**, без заявления HU.
Validation/test не участвовали в fit или выборе параметров по результатам модели.
256×256 — заранее выбранный compute budget и full-FOV fit/pad, без GT crop.
XY image linear / labels nearest. Z не resample; context [-2,-1,0,1,2] по индексам,
поскольку physical release spacing не подтверждён. За границами — явный replicate-edge.
Все семь исходных cardiac classes сохраняются после подготовки каждой пары.
Pad target=255, image=-1. Inverse mapping сохраняется; при evaluation
вероятности интерполируются на original XY до argmax, original Z неизменён.
Prediction NIfTI сохраняет original release grid/affine/units, без подмены unknown на mm.

Train: 10617 positive и 1764 negative slices; 12381 samples/epoch,
774 batches при batch16. Train-only выбор positives + не более равного числа negatives.
Validation/test используют все slices. Lazy NPY memory maps, не более двух cases
с pixels одновременно; маленькие headers/file handles кешируются для network volume.
Workers=0 для явного epoch state. Файлы готового cache проверены SHA-256.

Augmentation только train: rotation ±5°, scale ±5%, intensity shift ±0.03,
noise std 0.01; flips отсутствуют. Spatial transform общий для всех channels/target,
ignore pixels сохраняются. Это варианты исходных cases, не новые независимые пациенты.
Generative augmentation и имитация neonatal motion в v1 не включены.

## Модель и запуск

Own 2.5D U-Net, 5 input channels, 16→32→64→128, double Conv3×3/GN8/ReLU,
3 downsampling stages, bilinear decoder, 8 output logits, **489112 parameters**.
С нуля; benchmark weights в training не загружаются.
FP32, batch16, Adam LR0.001, 30 epochs, fixed LR, CE + own foreground Soft Dice.
Формула Dice для c: (2 sum(p*g)+epsilon)/(sum(p)+sum(g)+epsilon),
по B,H,W после исключения ignore; foreground class mean. All-ignore loss=0.

Best — максимальный mean-case foreground validation Dice на preprocessing grid.
Empty-reference class не получает автоматическую награду 1.
После training test оценивается отдельно на original grid: Dice/IoU/precision/recall
по структурам и cases. HD95/ASSD в mm выключены до подтверждения geometry.
Выводы об age/scanner/contrast effects требуют подтверждённой metadata.

Benchmark: 5 warmup + 50 measured training batches + 20 fixed-train sanity steps;
validation/test не используются. Реальное CUDA время измеряется с synchronize,
сохраняются peak allocated/reserved VRAM, RSS/CPU, environment, config и history.
Измеренная оценка: 3.91 min/train epoch, 117.30 min/30 train epochs; validation/save I/O не включены.
[CUDA timing](https://docs.pytorch.org/docs/2.14/notes/cuda.html) требует учёта
асинхронного выполнения; CPU smoke timings для GPU estimates не используются.

Full training требует отдельного --allow-full-training и matching successful GPU receipt:
тот же GPU, commit, config, split, preprocessing, cohort и prepared manifest.
CLI не имеет CPU fallback. Сохранит best.pt/last.pt, history, environment, pip freeze,
config, provenance, resource samples. Evaluation сохранит original-grid predictions/metrics.
Автоматический RunPod script выполняет только benchmark.

## Локальные файлы для передачи

```powershell
$dataRoot = 'E:/3d-heart-data/pediatric_ct_heart'
$bundle = "$dataRoot/training_bundles/cardiac_chd68_v1_portable.tar.gz"
$receipt = "$dataRoot/training_bundles/cardiac_chd68_v1_portable.tar.receipt.json"
```

Cache: 5742884046 bytes (5.743 GB); bundle **3332761224 bytes (3.333 GB)**,
225 payload files, source payload 5750421761 bytes.
Внутри prepared data/provenance, prepared manifest и original validation/test masks.
Нет raw CT, DICOM/RTSTRUCT, исходных dataset archives, NII данных или checkpoints.
Конфиги/код/маленькие summaries берутся из Git и проверяются против bundle manifest.

Archive SHA-256:
`456fa3f28bae409c33665014d25325bdccbc17cd251b953208a1b88d057cd596`.
После копирования проверяются archive SHA/size, затем каждый extracted file SHA.
Оригиналы на D: и E: не удалялись; запас E: значительно больше 80 GB.
Активный архив — *_portable.tar.gz. Прежний архив сохранён в training_bundles; прежние metadata bytes —
в training_bundles/portability_previous для provenance. Нормализация
LF не меняла split membership или pixels; обновлены только serialization hashes.
Четыре repository configs/manifests имеют одинаковые bytes в Windows и Linux.

## После аренды Pod

Получить текущие SSH Host/Port у RunPod; прежний endpoint не считать действующим.
Ключ только D:\.ssh\id_ed25519_runpod с IdentitiesOnly=yes; ключ на сервер не копировать.
Host key проверять для конкретного endpoint, без глобального отключения проверки.

```powershell
$podHost = '<actual-host>'
$podPort = <actual-port>
$identity = 'D:/.ssh/id_ed25519_runpod'
ssh -i $identity -o IdentitiesOnly=yes -p $podPort "root@$podHost"
```

Путь identity выше соответствует D:\.ssh\id_ed25519_runpod; пароль вводить
локально в ssh-agent/terminal, не в чат и не в scripts/config.

Сначала в SSH выполнить preflight до передачи данных:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'); assert torch.cuda.is_available(); assert '4090' in torch.cuda.get_device_name(0)"
df -h /workspace
findmnt -T /workspace
mountpoint -q /workspace
nproc
free -h
exit
```

При успешном preflight и достаточной persistent volume quota — из Windows:

```powershell
ssh -i $identity -o IdentitiesOnly=yes -p $podPort "root@$podHost" 'mkdir -p /workspace/incoming'
scp -P $podPort -i $identity -o IdentitiesOnly=yes $bundle $receipt "root@${podHost}:/workspace/incoming/"
```

На Pod, с имеющимся CUDA PyTorch:

```bash
git clone --branch feature/pediatric-heart-segmentation --single-branch \
  https://github.com/Andrew2002acmr/3d-heart /workspace/3d-heart-cardiac-v1
cd /workspace/3d-heart-cardiac-v1
git rev-parse HEAD
bash scripts/runpod_cardiac_benchmark.sh \
  /workspace/incoming/cardiac_chd68_v1_portable.tar.gz \
  /workspace/incoming/cardiac_chd68_v1_portable.tar.receipt.json
```

Если repo уже есть, сначала inspect status/branch/commit; не overwrite/reset.
Script проверит RTX4090, CUDA, df/findmnt/mountpoint, CPU/RAM; создаст venv
с system-site-packages и persistent pip/temp dirs. CPU torch не устанавливает.
Shared volume df может показывать pool size — отдельно учитывать фактическую
квоту Pod volume. Для bundle+unpack+checkpoints нужен volume хотя бы около 25 GB;
реальный free/quota проверить до transfer. Весь код и data/output — /workspace.

После отдельного решения о полном запуске (эти команды пока не выполнялись):

```bash
source /workspace/venvs/cardiac_chd68_v1/bin/activate
python -m heart3d.ml.cardiac_train train \
  --config configs/cardiac_chd68_runpod_v1.json \
  --data /workspace/cardiac_chd68_v1 --run-name gpu_full_v1 \
  --benchmark /workspace/cardiac_chd68_v1/experiments/cardiac_chd68_v1/gpu_benchmark_v1/summary.json \
  --allow-full-training

python -m heart3d.ml.cardiac_train evaluate \
  --config configs/cardiac_chd68_runpod_v1.json --data /workspace/cardiac_chd68_v1 \
  --checkpoint /workspace/cardiac_chd68_v1/checkpoints/cardiac_chd68_v1/gpu_full_v1/best.pt \
  --partition test --output /workspace/cardiac_chd68_v1/evaluation/test_v1
```

Не использовать test для подбора параметров. Новая evaluation не перезапишет GT
или существующий output. До остановки Pod выгрузить critical checkpoints,
experiments и evaluation в новый каталог E:; сопоставить SHA-256 всех файлов.
Full-training checkpoints/quality metrics/test predictions ещё отсутствуют.
Benchmark receipt/environment/history/logs уже сохранены локально и проверены SHA-256.

## Проверки готовности

Полный suite с новым training/data pipeline/archive verifier: 174 passed,
322 прежних warnings, 62.81 s. После byte-level LF normalization выполнены
семь focused regression tests, включая Windows/Linux serialization.
Реальный cache smoke: 3 batches, 2 train development cases; input 2×5×256×256,
output 2×8×256×256, target 2×256×256; finite loss/gradients, weights обновлены.
Это была локальная техническая проверка; последующий GPU benchmark описан ниже. Full training ещё предстоит.

## Миграция Pod и фактический GPU benchmark — 09.10.2026

Новый endpoint пользователя: 213.173.109.80:12476, root; доступ подтверждён
только прежним явно заданным RunPod ключом. Новый host key сохранён для этого
endpoint; глобальная SSH-проверка не отключалась. Старый /workspace/3d-heart
остался на 7c93cb9, его данные и результаты binary Heart v1 не изменены.
Новый чистый checkout /workspace/3d-heart-cardiac-v1:
**d89ea289124501654232b8ed6333d671410244d4**, feature/pediatric-heart-segmentation.

RTX4090 24564 MiB, driver 580.95.05; Python 3.12.3,
PyTorch 2.8.0+cu128 / CUDA12.8. Доступны 32 logical CPU, 16 physical CPU,
134.12 GB RAM. /workspace — отдельный persistent FUSE network mount.
df отражает общий pool, не индивидуальную купленную квоту; её значение независимо
не проверено. Дополнительно переданы archive 3.333 GB и unpacked payload 5.750 GB.
Все записи и integrity checks завершились успешно; GPU environment/temp/cache
и результаты находятся под /workspace. НИИ CT не передавались.
Локально E: остаётся около 243.89 GB свободно, резерв 80 GB соблюдён.

SHA-256 archive совпал с локальным receipt; 225 payload files и четыре
repository configs/manifests проверены. Prepared cache дополнительно прошёл
проверку 204 файлов. Linux targeted suite: **7 passed, 33.18 s**.
Модель, preprocessing и frozen split 48/10/10 не менялись.

[Machine-readable GPU report](../metadata/pediatric/cardiac_chd68_gpu_benchmark_v1.json).
FP32/batch16: 5 warmup, 50 measured batches, 20 fixed-train sanity updates;
validation/test не использовались, benchmark weights не сохранялись.

| Показатель | Измеренное значение |
|---|---:|
| Mean sec/batch, включая загрузку/augmentation | 0.303102 |
| Median sec/batch | 0.301406 |
| Samples/sec | 52.79 |
| Train batches/epoch | 774 |
| Estimated train epoch | 234.60 s / 3.91 min |
| Estimated 30 train epochs | 7038.02 s / 117.30 min |
| Peak process RSS | 1.883 GB |
| Peak CUDA allocated | 1.921 GB |
| Peak CUDA reserved | 3.127 GB |
| Mean process CPU | 193.68% / около 1.94 logical CPU |
| Fixed-train loss before → after | 3.33890 → 1.83319 |

Loss/gradients конечны, weights обновились, sanity_passed=true.
Это технический learning check, не результат качества segmentation.
Оценка около **1 h 57 min** относится только к train batches; validation,
checkpoint I/O и изменение нагрузки network filesystem требуют дополнительного
времени. Benchmark использовал уже прочитанный для SHA cache; будущая нагрузка
сетевого тома может отличаться. Проверку quota проводить перед full launch.

CUDA nll_loss2d в torch2.8 выдаёт предупреждение о недетерминированной реализации
при deterministic warn_only. Seed/config/environment фиксированы, но побитовая
идентичность повторных запусков не гарантируется. Age/patient identity/physical
scale ограничения CHD68 остаются: neonatal accuracy и mm метрики не заявляются.

Восемь критичных файлов (summary/history/config/environment/resources/pip freeze,
preflight/log) выгружены в новый локальный каталог
E:/3d-heart-data/pediatric_ct_heart/remote_runs/cardiac_chd68_v1_20261009.
Archive и каждый payload проверены SHA-256. Results archive SHA:
e034ebb090d3a937bb93089b1926863c94f2f2275c7651045c54aa54d3488b5e.

**Рекомендация:** технически можно запускать полный публичный multiclass baseline.
Текущий этап остановлен после benchmark; full training/test не запускались.
Pod после backup verification свободен и может быть остановлен пользователем.
При запуске использовать matching receipt и execution commit d89ea28 в оставленном
checkout. Более поздний docs-only commit не является execution commit; после
обновления Git HEAD safety gate потребует новый короткий benchmark.

## Разрешённый полный запуск — 09.10.2026

После команды пользователя «Запускай обучение» стартовал отдельный процесс
**gpu_full_v1_20261009**: 30 epochs с нуля, прежние config/split/preprocessing,
execution commit d89ea28. Benchmark weights не загружались, test не участвует
в выборе checkpoints. PID runner1487 / trainer1526; detached process сохраняется
при закрытии SSH. Наблюдались GPU67%, 3188 MiB, stage training.
Это подтверждение работы, а не segmentation quality.

Run outputs:
/workspace/cardiac_chd68_v1/experiments/cardiac_chd68_v1/gpu_full_v1_20261009;
checkpoints:
/workspace/cardiac_chd68_v1/checkpoints/cardiac_chd68_v1/gpu_full_v1_20261009.
После training runner автоматически выполнит **одну** evaluation best checkpoint
на 10 frozen test cases в original release grid, затем SHA-256 export.
Test evaluation output: /workspace/cardiac_chd68_v1/evaluation/test_v1_20261009.
Results archive prefix:
/workspace/exports/cardiac_chd68_gpu_full_v1_20261009_results.
При ошибке runner отметит failed и сохранит созданные артефакты; повторный запуск
в тот же output не предусмотрен.

[Launch/state summary](../metadata/pediatric/cardiac_chd68_full_run_v1.json).
Launch receipt и runner script скопированы на E: в
remote_runs/cardiac_chd68_v1_full_20261009; final checkpoints/metrics ещё не готовы.
После отдельного разрешения пользователя настроена временная проверка в этом
чате каждые 5 минут (cardiac-chd68-v1), с уведомлением об ошибке/итоге и отключением
после результата. После verified backup будут подготовлены CT/GT/prediction
примеры и отчёт. Пока обучение работает, Pod не останавливать.

## Завершение full run и test evaluation

30 epochs завершены, best epoch23 по validation0.8021. Full training117.74min.
Одна original-grid evaluation10test: mean case macro foreground Dice0.7897,
median0.8205, range0.5794–0.8783. PA0.6847, AO0.7474 в среднем;
ct_1083 PA0.3375 и выраженное смешение структур. Клиническую готовность не заявлять.
Все39 critical файлов на E SHA-verified, best/last hashes отдельно совпали.
Визуально проверены best/upper-median/worst examples, orthogonal/error panels
и train/test charts; helper scripts/qa_cardiac_predictions.py.
Heartbeat cardiac-chd68-v1 PAUSED, GPU process завершён; Pod можно остановить.
[Итоговый отчёт](cardiac_chd68_results_v1.md),
[машинные результаты](../metadata/pediatric/cardiac_chd68_results_v1.json).
Старые разделы о launch/benchmark описывают состояние на соответствующий момент.
