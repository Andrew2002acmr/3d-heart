# RunPod baseline v1

Пользователь разрешил один полноценный baseline после успешного GPU benchmark.
Когорта и preprocessing предыдущего этапа неизменны: 60 пациентов, 42/9/9,
5 axial planes с offsets −4/−2/0/2/4 mm, 256×256, Z 2 mm, HU [−1000,969].
Это исходный экспертный Heart OAR, клинический статус unknown.

## Зафиксированный эксперимент

Собственная 2.5D U-Net, 488993 параметра, FP32, batch 16, Adam LR 0.001,
30 epochs, без scheduler и без поиска гиперпараметров, BCE + Soft Dice (1+1).
Seed 20261006; workers=0; train-only augmentation без flips. Полное обучение
начинается с нуля. Benchmark weights не сохраняются и не используются.
GPU seed/order фиксированы, TF32 отключён. Deterministic algorithms warn-only:
в некоторых CUDA/PyTorch версиях bilinear backward не гарантирует побитовую
воспроизводимость. Фактические версии сохраняются вместе с результатами.

Best checkpoint выбирается по среднему patient-level Dice validation на
preprocessing grid, threshold 0.5. Проверяются все validation slices без
балансирования. Test не участвует в обучении и выборе параметров. После 30
epochs best checkpoint оценивается один раз на original DICOM-derived grid.
Probabilities восстанавливаются linear, затем threshold 0.5; postprocessing
отсутствует. Dice/IoU/precision/recall по пациентам; HD95 — максимум двух
направленных 95-х percentiles; ASSD — pooled mean surface-voxel distances,
граница 6-connected. Пустая prediction даёт Dice 0 и undefined surface distances,
без исключения пациента из overlap aggregate. Physical spacing подтверждён QA.

## Минимальный bundle и запуск

Все локальные тяжёлые файлы — во внешнем data root на E:, резерв 80 GB.
Bundle содержит 60 preprocessed image/mask/provenance, partition indices и
18 original GT masks для held-out evaluation. Raw CT/RTSTRUCT не переносятся.
Manifest фиксирует SHA-256 каждого файла и точные config/split/cohort/preproc.
Репозиторий находится в /workspace/3d-heart, данные в /workspace/pediatric_ct_heart.
Remote storage reserve 10 GB — отдельный бюджет /workspace; локальный резерв
80 GB не изменяется. Перед запуском проверяются mount, свободное место, CUDA,
RTX 4090, commit, branch и все SHA. requirements-runpod.txt не устанавливает
torch: используется проверенная CUDA installation Pod.

```powershell
python -m heart3d.ml.bundle training --config configs/pediatric_heart_runpod_v1.json --data $dataRoot --archive $archive --manifest $manifest
```

```bash
python -m heart3d.ml.bundle unpack --archive /workspace/transfer/frozen_v1_20261006.tar.gz --destination /workspace/pediatric_ct_heart --reserve-GB 10
python -m scripts.run_pediatric_gpu --config configs/pediatric_heart_runpod_v1.json --data /workspace/pediatric_ct_heart --run-name gpu_full_v1_20261006 --export /workspace/exports
```

Driver проверяет bundle, выполняет 5 warmup + 50 measured batches и 20
fixed-train sanity steps, требует уменьшения loss и finite nonzero gradients.
Только успешный matching config/commit GPU benchmark разрешает full training.
Оценка времени benchmark исключает validation/I/O; фактическое время каждого
train/validation epoch записывается отдельно. Сохраняются best/last, history,
config, environment, pip freeze, per-patient test metrics, predictions на original
grid и provenance. Results archive скачивается на E:, SHA archive и каждого
файла проверяются. До этой проверки считать remote результаты сохранёнными
локально нельзя. Под можно остановить после подтверждения локальной сохранности.

## Результаты

Здесь будут зафиксированы только фактические результаты завершённого запуска.
Локальная проверка до подключения: 124 теста проходят, включая synthetic full
trainer, best/last checkpoint, patient-level validation, mm distances и безопасный
SHA bundle round-trip. CUDA benchmark и обучение ещё не выполнялись на момент
фиксации исходного training code.

## Фактическая готовность подключения, 2026-10-06

Prepared train/validation/test: 42/9/9, параметры preprocessing и frozen SHA
не изменены. Minimal bundle: **201 файл**, 4,784,231,262 bytes исходных данных,
**2,352,906,707 bytes** compressed archive. SHA-256 archive:
`8c8c2e7981cea6497e77462c403b396429ba4f840504d7818b6650bb29ad164e`.
Manifest SHA: `197b646938806242ea4d98cd5ecdbd9f1b26d1706df6ed58df441a2ce6f920d2`.
Files `runpod_transfer/frozen_v1_20261006.tar.gz` и manifest находятся под local
data root. Свободно ~277.7 decimal GB, резерв 80 GB выполнен; raw DICOM не входят.
Machine-readable status: [heart_runpod_preparation_v1.json](../metadata/pediatric/heart_runpod_preparation_v1.json).

SSH TCP endpoint доступен. Использован только явно заданный RunPod key,
IdentitiesOnly=yes. Host key впервые сохранён per-connection accept-new;
повторная проверка с StrictHostKeyChecking=yes подтвердила совпадение. Это
trust-on-first-use, не независимая проверка fingerprint через провайдера.
Сервер **отклоняет публичный ключ до запроса passphrase**. Интерактивная попытка
запрашивает пароль серверного root; пользователь подтвердил, что располагает
только паролем приватного ключа. Неверный password prompt отменён.
Пользователь добавил key в RunPod Credentials, но повторная попытка также получила
отказ. Account keys внедряются при старте Pod: [официальная инструкция RunPod](https://github.com/runpod/runpod-plugins-official/blob/main/plugins/runpod/skills/runpod/golden-paths/06-dev-pod.md).
Требуется restart с проверкой нового endpoint или ручное добавление .pub в
authorized_keys через Web Terminal. Приватный ключ не выводился и не копировался.

**Данных на RunPod ещё нет; GPU/CUDA/mount не проверены, GPU benchmark/full
training/test evaluation не выполнялись.** GPU throughput/epoch/full-time estimates
и scientific scores отсутствуют. Нельзя утверждать, что cloud baseline прошёл
или что результаты cloud сохранены локально. Пользователь предупреждён о простое
оплачиваемого Pod. После применения key требуется ввести passphrase локально
(например ssh-add для этого key в Windows ssh-agent), затем проверить hardware,
mount и storage, pin feature commit, передать archive с SHA проверкой и выполнить
driver. При перезапуске Pod уточнить endpoint; host-key mismatch не обходить.
