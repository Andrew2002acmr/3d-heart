# Восстановление локального inference CHD68

Дата: 22.10.2026. Базовый код: `feature/pediatric-heart-segmentation`, commit `9a56fb9e7182057ce0b4a8b8c18c1662d4f93600`; новая локальная inference-команда и её тесты пока не закоммичены.

**Результат:** CHD68 v1 и Resolution384 v2 восстановлены, их обученные веса загружаются штатным загрузчиком и выполняют image-only inference на RTX 2060 6 GiB. Обе модели прошли синтетический путь «CT array → frozen preprocessing → GPU → probabilities → original-grid mask». Реальный CT в этой сессии не использовался: подходящей публичной development-пары локально нет. Поэтому техническая готовность подтверждена, качество на настоящем CT здесь ещё не проверено.

Новые клинические исследования не читались. Обучающих экспериментов, подбора по test, изменения checkpoint и записи в S3 не было. Восстановленные модели работают в `eval` и `inference_mode`; optimizer для inference не создаётся.

## 1. Целостность checkpoint

Исходные файлы оставлены в `C:/Users/Andre/heart-model-recovery/` без изменений. SHA-256 вычислен по всему файлу, размер проверен отдельно.

| Файл | Bytes | SHA-256 | Результат |
|---|---:|---|---|
| `chd68_v1_best.pt` | 6094048 | `c946ae1787583ff323e1620d82f6d069742c180c40754efb414a18d21b7145ee` | Точное совпадение с историческим best epoch 23 и result manifest |
| `chd68_v2_best.pt` | 6006560 | `fc24b90d6a7cf3a8f14130e44b82e78bcff7de7fa30d0cbb84c00474dd8760df` | Точное совпадение с историческим best epoch 10 и result manifest |
| `heart_binary_v1_best.pt` | 5938400 | `27e52ec9e9a25792cab4996cd0845d8a68e2862ed619076c802b4f4063dd3932` | **Не совпадает** с финальным Heart v1 |

Для финального бинарного Heart v1 ожидаются **5954592 bytes**, SHA `e6b28f1b87d581fde2dbc2e055ec58f0353c98c913f62c10b96ba3338690e8f0`, best epoch 18. Полученный бинарный файл содержит epoch 1, одну запись history, другой config/cohort/split/preprocessing SHA и один выходной канал. Это не восстановленный финальный Heart baseline. Точное происхождение не установлено; файл не использовался для inference и не заменялся автоматически. Нельзя приписывать ему исторический Dice 0.9194.

`torch.load(..., map_location="cpu", weights_only=True)` использован для безопасного чтения checkpoint. Для CHD68 дополнительно выполнен strict `load_state_dict` через существующий `cardiac_train.load_checkpoint`; missing/unexpected parameters не было.

### Предоставленный `manifest_v1.json`

- Размер: 42750 bytes.
- SHA: `c89baa87c256ef276628a7cc09fd783c7ba87a941f80dc49ef16eb01c347f8c7`.
- Совпадает с `bundle_receipt.manifest_SHA256` в [историческом readiness JSON](../metadata/pediatric/cardiac_chd68_runpod_readiness_v1.json).
- Это **входной training bundle manifest**, 225 payload entries и четыре repository references, а не manifest обученных весов; `.pt` entries отсутствуют.
- Все четыре repository SHA проверены и совпали. Сами 225 payload files здесь отсутствуют и повторно не проверялись.

## 2. Read-only восстановление manifests из Runpod S3

AWS CLI 2.38.4 найден в `C:/Program Files/Amazon/AWSCLIV2/aws.exe`; в текущем PATH команды `aws` не было, поэтому использован абсолютный путь. Существующий профиль `runpod` работает. Регион: `EU-RO-1`, endpoint: `https://s3api-eu-ro-1.runpod.io/`, bucket: `2n3wohq954`. Настройки и credentials не изменялись и не экспортировались.

Использовались только `list-buckets`, `list-objects-v2` и **`s3api get-object`**. Скачаны три JSON, всего **31842 bytes**:

| S3 key | SHA-256 локального JSON | Проверка |
|---|---|---|
| `exports/cardiac_chd68_gpu_full_v1_20261009_results_manifest.json` | `1db5ba86a1b2ac0bb38210cba2d92d40160b0338a28a627474896ee265a6571f` | Совпадает с историческим result manifest SHA v1 |
| `exports/cardiac_chd68_resolution384_v2_20261010_results_manifest.json` | `7d4bc6f79558267d302662d89c0a6c0e6cb7bd1308175cc51f6f20b12c37fd2b` | Совпадает с историческим result manifest SHA v2 |
| `exports/gpu_full_v1_20261006_io_manifest.json` | `90bc227fc1c72c73c47a00bdd1ab38ad506f47b224ab0ab378754b0e7f90d1ff` | Best entry совпадает с историческим финальным Heart SHA; для самого JSON ранее закреплённый SHA отдельно не найден |

Пример реально использованной команды, запуск из корня `repository`:

```powershell
& 'C:\Program Files\Amazon\AWSCLIV2\aws.exe' s3api get-object `
  --bucket 2n3wohq954 `
  --key exports/cardiac_chd68_gpu_full_v1_20261009_results_manifest.json `
  --profile runpod --region EU-RO-1 `
  --endpoint-url https://s3api-eu-ro-1.runpod.io/ --no-cli-pager `
  outputs\model_recovery_20261022\manifests\chd68_v1_results_manifest.json
```

Веса, большие tar/NPY volumes, test GT и другие данные из S3 не скачивались. PUT/DELETE/COPY и иные изменения cloud storage не выполнялись. Небольшие конфигурации не требовали скачивания: их локальные байты уже совпадают с checkpoint hashes.

## 3. Архитектура и frozen protocol

| Свойство | CHD68 v1 | Resolution384 v2 |
|---|---|---|
| Модель | `HeartUNet25D` | Та же архитектура |
| Параметры | 489112 | 489112 |
| Вход | 5 axial index planes, 256×256 | 5 axial index planes, 384×384 |
| Encoder | 16/32/64/128, Conv2d/GroupNorm/ReLU | Без изменений |
| Decoder | Bilinear upsample + skip concatenation | Без изменений |
| Выход | 8 logits | 8 logits |
| Выбранная эпоха | 23 | 10 |
| Source intensity clip | [0, 2015] | [0, 2015] |
| Normalization | Linear → [-1, 1], padding -1 | Без изменений |
| Context offsets | [-2, -1, 0, 1, 2] по индексам | Без изменений |
| Resampling | Full-FOV XY fit/pad, linear CT; без Z resampling | То же, целевая XY сетка 384 |
| Fit preprocessing | Только исходные train 48 | Те же train statistics, без refit |
| Frozen split | 48 train / 10 validation / 10 test | Тот же; test в v2 sealed |
| Geometry | Координаты исходного release; physical scale не подтверждён | То же |

Метки: `0 background`, `1 LV`, `2 RV`, `3 LA`, `4 RA`, `5 MYO`, `6 AO`, `7 PA`. Восемь logits означают семь foreground классов, а не восемь анатомических структур. Наличие всех классов в каждом конкретном prediction не гарантируется.

Проверенные SHA конфигураций:

| Файл | SHA-256 |
|---|---|
| `configs/cardiac_chd68_runpod_v1.json` | `7bf1064769004e0a24fa1ee49103dd5e7418d17576098ffb8848c8a24e9451a1` |
| `configs/cardiac_chd68_resolution384_v2.json` | `ebc9add57e812383028cab7e1ab01634d90832ee7430f362df950b93fddc3bfb` |
| `configs/cardiac_chd68_preprocessing_v1.json` | `a7e1252702113aac3b79ac505293c5d9e5db7807e4bf0da03b4635d2ba7bdf2c` |
| `configs/cardiac_chd68_preprocessing_resolution384_v2.json` | `7acf0d113f95e41408a905c5e924ea3dda3aea61e41db0af97bda84ea445decf` |
| `configs/splits/cardiac_chd68_v1.json` | `a291dd9d1cb5ce9437191e0260e590235cd0c5d5a34223ae8bb6a73cd14e3bdc` |
| `metadata/pediatric/cardiac_chd68_cohort_v1.json` | `bf60efae0c766af11893e7414361e66a09c78f325c2c658f61f88de8839244e0` |

Штатный loader проверил config/split/cohort/preprocessing hashes. Embedded config и preprocessing дополнительно сравнены с JSON как структуры данных. Исходные frozen files не менялись. Размер batch для локального inference задаётся отдельно и не меняет training config.

Source intensity **не объявляется HU**. Автоматически применять обученный clip [0, 2015] к произвольному клиническому DICOM/HU без проверки domain adapter нельзя. Масштаб и анатомическая ориентация release остаются прежним исследовательским ограничением.

## 4. Локальная команда image-only inference

Добавлен [heart3d/ml/infer_cardiac.py](../heart3d/ml/infer_cardiac.py). Он переиспользует:

- [cardiac_train.load_checkpoint / probability_batches](../heart3d/ml/cardiac_train.py);
- [index_transform / resize_plane / restore_probability_plane](../heart3d/ml/cardiac_data.py);
- существующую [HeartUNet25D](../heart3d/ml/model.py) и frozen JSON.

Старый `cardiac_train.evaluate` требует полный prepared manifest, кэш, GT и штатно разрешённую Runpod RTX 4090. Эти ограничения для benchmark/training не удалялись. Новая команда делает только inference: не вызывает setup/train/evaluate, не требует GT и не создаёт optimizer.

Выход: NIfTI `uint8` на исходной XYZ-сетке и `inference.json` с SHA/provenance/time/VRAM. Probabilities восстанавливаются на original XY grid **до argmax**, исходное Z сохраняется. Для публичного CT сохраняются affine, qform/sform codes и spatial units. Новые output directories обязательны; старые результаты не перезаписываются. Постобработка, удаление компонент, smoothing и подстройка порогов не применяются.

В рамках восстановления команда принимает synthetic input или **точную SHA-verified публичную development/train CT** из frozen cohort. Test и clinical inputs отклоняются; это намеренная граница текущего задания. Для обычного получения prediction нужен только CT, маска GT используется позднее для проверки качества.

Повтор synthetic checks из `C:\Users\Andre\Desktop\3D-heart\repository`:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -m heart3d.ml.infer_cardiac `
  --variant v1 --checkpoint C:\Users\Andre\heart-model-recovery\chd68_v1_best.pt `
  --synthetic --batch-size 2 --repeats 5 --out outputs\recovery_v1_repeat

.\.venv\Scripts\python.exe -m heart3d.ml.infer_cardiac `
  --variant v2 --checkpoint C:\Users\Andre\heart-model-recovery\chd68_v2_best.pt `
  --synthetic --batch-size 2 --repeats 5 --out outputs\recovery_v2_repeat
```

Если соответствующий output существует, выбрать новое имя. CPU fallback не выполняется молча. Для RTX 2060 выбран batch 2; существующий training batch 16 не переносился автоматически. Локальная команда ограничивает batch 1–4.

## 5. Замеры на RTX 2060

Окружение: Windows 10 x64, i5-9300H, 16 GiB RAM, NVIDIA RTX 2060 **6144 MiB**, driver 595.79, PyTorch **2.14.1+cu130**, CUDA runtime **13.0**. Вычисления FP32, TF32 отключён, `model.eval()`, `torch.inference_mode()`. Обе модели проверены последовательно в отдельных процессах, без параллельной GPU-нагрузки от другой модели.

Одинаковый искусственный source volume: **128×112×16 XYZ**, seed 20261022, значения [0, 2015]. Это числовой массив, не изображение сердца. После preprocessing: 16×256×256 или 16×384×384; batch 2, edge replication контекста, один первый запуск и пять повторов после прогрева.

| Измерение | CHD68 v1 | Resolution384 v2 |
|---|---:|---:|
| Загрузка и проверка checkpoint + CUDA sync | 0.422 s | 0.391 s |
| Preprocessing source volume | 0.057 s | 0.165 s |
| Первый полный inference volume | 0.607 s | 0.635 s |
| Среднее пяти прогретых запусков | **0.246 s** | **0.388 s** |
| Медиана | 0.240 s | 0.390 s |
| Диапазон прогретых запусков | 0.236–0.259 s | 0.373–0.402 s |
| Peak torch allocated, включая модель | **78.75 MiB** | **174.85 MiB** |
| Peak torch reserved | **98 MiB** | **264 MiB** |
| `nvidia-smi memory.used` после запуска | 199 MiB | 365 MiB |

Для времени выполнена CUDA synchronization перед началом и после завершения каждого замера. Полный inference time включает forward, GPU→CPU transfer, обратное преобразование probabilities и argmax; excludes checkpoint load, preprocessing и сохранение NIfTI — они измерены отдельно либо не входят в таблицу времени.

Torch memory peaks не включают весь CUDA context/driver и чужие приложения. `nvidia-smi` — общий снимок после запуска, не измерение исторического максимума. Нельзя напрямую сравнивать эти inference peaks с прежними training reserved 2.85–6.36 GB.

На настоящем CT глубина и original XY grid могут быть существенно больше; CPU restoration/IO и RAM будут другими. Значения 0.246/0.388 s нельзя выдавать за время полного клинического CT. Качество, Dice и полнота семи структур по синтетическим данным не оценивались. Некоторые foreground классы на искусственном input отсутствуют; это не ошибка числа выходов и не медицинский вывод.

## 6. Проверки и artifacts

**16 tests passed, 16.40 s**: четыре новых теста плюс существующие resolution, multiclass training и runner guard regressions. Новые тесты проверяют отказ при неверном checkpoint до deserialization, отказ для test/clinical case до CT IO, image-only preprocessing без изменения CT/frozen statistics и сохранение prediction на исходной геометрии.

Существующие regression-тесты включают короткие optimizer/trainer checks на временных синтетических fixtures; обучающие эксперименты на наборах не проводились, восстановленные checkpoint не обновлялись. Проверки GPU выше использовали только inference. `pip check` ошибок не обнаружил, `git diff --check` чистый. Training guards, architecture, старые configs и data splits сохранены.

Локальные artifacts вне Git:

- [integrity.json](../outputs/model_recovery_20261022/integrity.json) — фактические и ожидаемые SHA/bytes;
- [recovery_summary.json](../outputs/model_recovery_20261022/recovery_summary.json) — замеры и ограничения;
- [v1 inference.json](../outputs/model_recovery_20261022/v1_synthetic/inference.json), [v2 inference.json](../outputs/model_recovery_20261022/v2_synthetic/inference.json);
- [тесты/JUnit](../outputs/model_recovery_20261022/tests.xml), [манифесты](../outputs/model_recovery_20261022/manifests/);
- `synthetic_prediction.nii.gz` в каждой папке — технический output искусственного input, не клиническая сегментация и не GT.

В Git предназначены только новая команда, [тесты](../tests/test_cardiac_inference.py) и этот отчёт. Модели и большие данные не добавлялись, commit/push не выполнялись.

## 7. Что минимально нужно для настоящего CT inference

Уже есть веса, architecture, label mapping, exact configs, frozen preprocessing, source cohort/split и CUDA environment. Для первого реального запуска не нужны весь dataset, полный training cache, optimizer state, training bundle или GT для самого предсказания.

Минимально нужны:

1. Один source-compatible CT NIfTI с подтверждёнными SHA, XYZ shape и orientation, из заранее объявленного public development/train case.
2. Его provenance и подтверждение source intensity convention; код сохраняет geometry, но не исправляет неопределённый физический масштаб release.
3. Для оценки качества — соответствующая исходная GT mask, точно та же сетка и mapping 0–7; дополнительные IDs игнорировать по frozen policy, не превращать в background.
4. Новый output directory. GT не передаётся inference-команде и не участвует в crop/normalization/model input.

Проверены каталоги `data`, `datasets`, `checkpoints`, `outputs` проекта и `C:/Users/Andre/heart-model-recovery`: подходящих NIfTI/NPY CT нет. Полный поиск компьютера не проводился. Наличие CT новых клинических пациентов не использовалось как замена development data.

Проверены только два ожидаемых S3 prefixes для `ct_1001` в roots v1/v2: отдельных original CT/GT объектов там не обнаружено. Это не поиск по всему bucket и не доказательство полного отсутствия пары в иных archives/locations; большие archives автоматически не скачивались.

## 8. Минимальный план получения одной CT + GT пары

Выбран заранее **`ct_1001`**, который входит в frozen development и train, отсутствует в validation/test. Для этого case ранее подтверждена полная SHA-идентичность ImageCHD/CHD68 CT и GT; это один и тот же development пример, не независимая внешняя оценка.

Ожидаемые файлы:

| Файл | SHA-256 |
|---|---|
| `ct_1001_image.nii.gz` | `03c6c49cb914704b68672cc92f3327dd50304c24cbe8fdbdd0ef34fe7fcd8962` |
| `ct_1001_label.nii.gz` | `a8d04123b65b782e38010ff3b0c168437a76ae086b2849b622373dcc4fe4a6c2` |

Ожидаемая shape: **512×512×221**. NIfTI header axes RAS; spatial units unknown — не подменять их mm. Источник: [существующий cohort](../metadata/pediatric/cardiac_chd68_cohort_v1.json), [аудит overlap](cardiac_structure_segmentation_pilot.md).

Предпочтительно перенести ровно эти два файла из прежнего verified backup. Если backup недоступен, существующий helper [fetch_sample.py](../scripts/fetch_sample.py) умеет получать выбранную пару ImageCHD через небольшой final ZIP и необходимый prefix z01, с original CRC/size check. Это получение выбранного development case, не скачивание всего набора. **Следующие команды — план, они здесь не выполнялись; текущая доступность download endpoint не проверялась скачиванием данных.**

```powershell
# Из корня repository. Получить только выбранный case существующим helper.
.\.venv\Scripts\python.exe scripts\fetch_sample.py `
  --out data\public_development --cases ct_1001

# До inference сопоставить оба SHA с таблицей выше.
Get-FileHash data\public_development\ct_1001_image.nii.gz -Algorithm SHA256
Get-FileHash data\public_development\ct_1001_label.nii.gz -Algorithm SHA256

# Inference получает только CT. Никаких GT-path параметров здесь нет.
.\.venv\Scripts\python.exe -m heart3d.ml.infer_cardiac `
  --variant v1 --checkpoint C:\Users\Andre\heart-model-recovery\chd68_v1_best.pt `
  --ct data\public_development\ct_1001_image.nii.gz --case ct_1001 `
  --batch-size 2 --repeats 1 --out outputs\recovery_ct1001_v1

.\.venv\Scripts\python.exe -m heart3d.ml.infer_cardiac `
  --variant v2 --checkpoint C:\Users\Andre\heart-model-recovery\chd68_v2_best.pt `
  --ct data\public_development\ct_1001_image.nii.gz --case ct_1001 `
  --batch-size 2 --repeats 1 --out outputs\recovery_ct1001_v2

# build сохранит CT/prediction overlays в slices.png и поверхности VTP.
.\.venv\Scripts\python.exe -m heart3d build `
  --ct data\public_development\ct_1001_image.nii.gz `
  --mask outputs\recovery_ct1001_v1\cardiac_prediction_original.nii.gz `
  --out outputs\recovery_ct1001_v1_view
.\.venv\Scripts\python.exe -m heart3d view outputs\recovery_ct1001_v1_view
```

Сначала проверить shape/affine/units CT↔GT, immutable source hashes и overlays. Затем отдельно посчитать Dice по классам на original grid и сравнить CT/GT/prediction в одинаковом окне. Это development sanity check, не новый blind test и не основание настраивать модель по test. Нельзя считать выход годным только потому, что build/viewer его открыл. Для публикации точности в целевой группе 2–18 нужны подтверждённые ages и независимая клиническая/внешняя выборка.

## 9. Можно ли уже подавать исходный CT и получать семь структур?

**Технически локальный маршрут готов для SHA-verified публичного development CT NIfTI:** команда прочитает CT без GT и сохранит автоматически предсказанную маску классов LV/RV/LA/RA/MYO/AO/PA на исходной сетке. Обе trained модели уже проверены на GPU через тот же путь с синтетическим входом.

**На настоящем CT это пока не воспроизведено в этой сессии**, поскольку локальной проверочной пары нет. Следующее конкретное действие — получить одну указанную пару и выполнить команды раздела 8. Приём произвольного клинического DICOM и его перенос в CHD68 intensity domain не являются подтверждённой возможностью; current recovery command специально не принимает clinical inputs. Качество и наличие всех семи структур в каждом случае требуют отдельной проверки.
