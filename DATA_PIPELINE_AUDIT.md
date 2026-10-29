# Аудит ImageCHD68 и CHD68 v1/v2

Дата: 28.10.2026. Источники — реальные локальные файлы и код этого checkout.
Обучение, изменение весов и refit preprocessing не выполнялись.

## Что найдено

- Heart3D: `heart3d/`, существующие web/UI и ML модули сохранены.
- Реальные данные: `../laboratories/data/chd68/pairs/`, 136 NIfTI gzip файлов,
  68 изображений и 68 экспертных масок. DICOM в этом каталоге нет.
- Исходные multipart-архивы: `../laboratories/data/kaggle_parts/`.
- Receipt: `../laboratories/data/source_receipt.json`, Kaggle
  `xiaoweixumedicalai/chd68-segmentation-dataset-miccai19`, CC BY-NC 4.0,
  историческая проверка CRC. Новые SHA256 каждого изображения/маски — в batch JSON.
- Исторический source audit: `metadata/pediatric/public_chd_multiclass_audit_v1.json`.
  Некоторые одноимённые ImageCHD/CHD68 пары описаны как побайтовые дубликаты;
  они не добавляются повторно в набор.
- Cohort: `metadata/pediatric/cardiac_chd68_cohort_v1.json`.
- Frozen split: `configs/splits/cardiac_chd68_v1.json` — 48 train, 10 validation,
  10 test. Единица разбиения — release-case/group, независимость пациентов
  независимо не подтверждена. Development cases включены в train.
- v2 допускает только train/validation; test sealed. Лаборатория может описывать
  исходные данные всех разделов, но не подгоняет по ним параметры и не выдаёт
  исследовательские эксперименты за независимую оценку модели.

## Интенсивности и геометрия

`ct_1001` непосредственно прочитан: XYZ `(512,512,221)`, affine diagonal 1,
origin RAS `(1,1,1)`, orientation RAS, spatial unit `unknown`, NIfTI scaling
slope=1, intercept=0. Это **не доказательство HU или spacing=1 mm**.
При чтении nibabel учитывает slope/intercept. Независимых DICOM rescale metadata,
однозначно связанных с этими voxel volumes, в каталоге пар нет.

Исторический audit также явно помечает HU и физический масштаб как unverified.
Есть candidate source metadata, но linkage к release-case не подтверждён;
применять эти значения как физическую геометрию автоматически нельзя.

Новая проверка пар сравнивает shape, весь affine и spatial units; одних размеров
недостаточно. Проверяются qform/sform codes и противоречие активных форм.
Лаборатория строит отдельную ортогональную RAS сетку по world bounds, linear для
КТ и nearest для меток. Это геометрия исследования из заголовка, не восстановленная
клиническая геометрия. Неизвестные единицы сохраняются неизвестными.

## Реальные классы

`heart3d/ml/infer_cardiac.py:LABELS` и `cardiac_data.py:scores_from_confusion`:

| Значение | Класс |
|---|---|
| 0 | background |
| 1 | LV |
| 2 | RV |
| 3 | LA |
| 4 | RA |
| 5 | MYO |
| 6 | AO |
| 7 | PA |

Восемь output logits означают семь foreground структур и фон.
255 — ignore/padding target, не анатомический класс.
В исходных масках также найдены дополнительные значения вне 0..7 (исторический
audit перечисляет 8,11,12,13,14,15). Их анатомические названия в найденном
cardiac protocol не заданы. Это не новые выходные классы модели: подготовка и
`cardiac_train.evaluate` переводят все source labels>7 в ignore=255, исключая
из loss/metrics. Лаборатория сохраняет их в исходном анализе, показывает как
source extra и применяет историческое исключение только при cardiac metrics.

## Обучение и инференс

Подготовка train cache: `scripts/prepare_cardiac_chd68.py:prepare`.
Dataset: `heart3d/ml/cardiac_data.py:CardiacDataset`.
Обучение/validation: `heart3d/ml/cardiac_train.py`.
Восстановленный production inference: `heart3d/ml/infer_cardiac.py`.

| Параметр | v1 | v2 |
|---|---|---|
| Config | `configs/cardiac_chd68_runpod_v1.json` | `configs/cardiac_chd68_resolution384_v2.json` |
| Preprocessing | `configs/cardiac_chd68_preprocessing_v1.json` | `configs/cardiac_chd68_preprocessing_resolution384_v2.json` |
| Input BCHW | B×5×256×256 | B×5×384×384 |
| Channels | Z offsets -2,-1,0,+1,+2 | идентично |
| Clipping source | [0,2015] | идентично |
| Normalization | clip((x-0)/2015,0,1)*2-1 | идентично |
| XY | full FOV fit/pad, aspect по индексам | идентично, размер 384 |
| CT interpolation | linear, pixel centers | идентично |
| Mask interpolation | nearest, padding 255 | идентично |
| Orientation | source RAS gate, identity | идентично |
| Z / physical spacing | без resampling; физическая геометрия не используется | идентично |
| Out of bounds context | replicate edge, флаг | идентично |
| CT padding | -1 после нормализации | идентично |
| Crop | отсутствует; GT не управляет crop | идентично |

Фактический порядок: **исходная RAS КТ → XY resize/pad → clipping → normalization
→ сборка пяти каналов → BCHW → forward**. Не следует показывать clipping до
resize как настоящий pipeline: в общем случае эти операции не коммутируют.

Train script сначала пишет raw resized cache с fill=0, затем нормализует и
явно выставляет pad=-1. Inference `prepare_volume` использует fill=low, затем
нормализует. Для сохранённого low=0 это совпадает. Автоматический тест независимо
воспроизводит train-подготовку и сравнивает с production inference побитово,
включая прямоугольную форму и крайние Z с повторением контекста.

После forward вероятности восстанавливаются linear на исходную XY сетку,
**до argmax**, через `restore_probability_plane`. Source affine/qform/sform
сохраняются при экспорте NIfTI штатным `save_prediction`.

Bounds получены только из 48 train случаев: 65536 детерминированных original-grid
samples/случай, одинаковый вес случая, percentiles 1 и 99.5. Train mean/std
для Z-score не сохранены; их нельзя выдумывать или вычислять по test.

Augmentation train-only: rotation 5°, scale delta .05, intensity shift .03,
noise std .01, flips=false. При validation/inference augmentation выключена.
Sampling train: все foreground slices + не больше такого же числа background
slices/случай. Validation использует все срезы. Seed=20261009.

## Веса и метрики

Штатный loader: `cardiac_train.load_checkpoint`, `torch.load(weights_only=True)`,
strict state_dict и конфигурационные SHA. `infer_cardiac.load_recovered` проверяет
также полный historical checkpoint SHA и embedded config/preprocessing.

Реальные файлы прочитаны из `C:/Users/Andre/heart-model-recovery/`:

| Файл | Bytes | SHA256 |
|---|---:|---|
| chd68_v1_best.pt | 6094048 | c946ae1787583ff323e1620d82f6d069742c180c40754efb414a18d21b7145ee |
| chd68_v2_best.pt | 6006560 | fc24b90d6a7cf3a8f14130e44b82e78bcff7de7fa30d0cbb84c00474dd8760df |

Исторические метрики:
`metadata/pediatric/cardiac_chd68_results_v1.json`,
`metadata/pediatric/cardiac_chd68_resolution384_results_v2.json`,
`metadata/pediatric/cardiac_chd68_postprocessing_results_v1.json`.
В v2 paired validation comparison original-release-grid case-macro Dice:
baseline 0.7978329433, variant 0.7952308432, 10 случаев; это exploratory validation,
не независимый test. Выбор checkpoint происходил на prepared-grid validation.
Упомянутые в задании «около 0.73» не идентифицированы как конкретный показатель;
без provenance это число не интерпретируется.

Метрики `scores_from_confusion`: foreground macro только reference-present
классов, без награды за empty-empty; patient/case macro и voxel pooling различны.
Новые smoke-метрики относятся к **одному срезу ct_1001**, не всему случаю и не test.

## Реальная instrumentation и ограничения

`ct_lab/model_inspector.py` вызывает настоящую `prepare_volume` на пяти нужных
исходных Z planes, что точно эквивалентно full-volume preparation при отсутствии
Z resampling. Это проверено against full-volume production и train preparation.
После загрузки настоящих весов `forward_pre_hook` подтверждает побитовое совпадение
тензора перед forward. Реальный CPU smoke обеих версий сохранён в
`outputs/ct_preprocessing_lab/real_model_verification.json`.

Клиническая шкала и физические расстояния остаются неизвестными. Анизотропия
источника могла быть утрачена в release; это гипотеза, не измеренная величина.
Маски на неподтверждённой физической сетке анализируются в grid units³/².
Исходные файлы, веса и frozen protocol не изменяются.

## Завершённая проверка всей локальной копии

Все 68/68 image+mask пар успешно прочитаны; SHA256 каждой пары совпал с frozen
cohort. Все 68 имеют RAS, unknown spatial units, source dtype int16, scaling
slope=1/intercept=0. NaN/Inf и несогласованных пар нет. В 13 масках обнаружены
source labels вне 0..7; их обработка описана выше. Детальные измерения, аномалии
и новые exploratory прогоны ct_1001 обеих моделей — в
`IMAGECHD68_DATA_ANALYSIS.md` и `outputs/ct_preprocessing_lab/`.
