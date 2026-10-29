# FILTERING_FEATURE_EXPERIMENT_01

Дата: 28.10.2026. Исследовательские статистики только по реальным КТ; без обучения моделей на реальных данных и изменения CHD68 preprocessing.

## Протокол и область применимости

Основной интерактивный пример — ct_1001, source axial Z=110, train/development. Полная исходная плоскость → фильтр → признаки → ROI / sampled voxels. GT не входит в признаки.
Train groups: ct_1001, ct_1002, ct_1004, ct_1008, ct_1072.
Frozen validation groups: ct_1027, ct_1028, ct_1033, ct_1036, ct_1046, ct_1056, ct_1079, ct_1099, ct_1125, ct_1127.
Три source Z-позиции: round((nz−1) × 0.35/0.50/0.65). До 400 sampled voxels каждого класса на срез, deterministic seed=20261028. Планы зафиксированы в protocol.json до оценки validation. Все шесть конфигураций проверены без итеративного подбора по validation; test CT не открывались.
Validation отделён по release-case/group; независимая идентификация пациентов не подтверждена. Это ограниченная проверка выбранных срезов и классов, не whole-volume сегментация.
Веса, split, два CHD68 preprocessing JSON и source files проверены SHA256 до/после: unchanged. Источник интенсивностей — source intensity, не HU. Grid units не считаются миллиметрами.

## Точные преобразования

Baseline; Gaussian sigma=0.5/1/2 grid units, reflect; bilateral spatial sigma=1 grid unit, intensity sigma=100 source units, 10000 bins, reflect; CLAHE clip_limit=0.01, kernel=32 pixels, nbins=256, fixed [0,4095]→[0,1]→CLAHE→[0,4095]. Gaussian учитывает header spacing. Bilateral отклоняет анизотропную in-plane сетку вместо неверной трактовки scalar sigma.
Признаки: source/processed intensity; central-difference gradient / grid spacing; local mean и std в окне 5×5, reflect; LoG sigma=1 grid unit. Gradient Magnitude — отдельный канал, не замена КТ. Mask, координаты, class ID и distance-to-mask не подаются k-NN/PCA/UMAP.
Standardization: population mean/std только training samples. k-NN: k=15, weights=1/distance, exact cKDTree, zero-distance neighbors голосуют исключительно между собой. Никакого global refit модели CHD68.

## 1–2. Локальные колебания и границы LV/MYO

Ниже train-агрегаты: сначала медиана по доступным срезам случая, затем медиана по случаям (равный вес случая). Ratios сравнивают одни и те же позиции с baseline. Значение variance ratio ниже 1 означает меньшие локальные колебания; gradient ratio около 1 — сохранение среднего градиента на контактной GT-границе.

| Filter | Local variance ratio | Interface gradient ratio | Mean contrast ratio | Fisher ratio / baseline | Histogram overlap |
|---|---:|---:|---:|---:|---:|
| Baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0332 |
| Gaussian 0.5 | 0.9013 | 0.9732 | 0.9964 | 1.0049 | 0.0378 |
| Gaussian 1.0 | 0.6433 | 0.8922 | 0.9841 | 1.0104 | 0.0370 |
| Gaussian 2.0 | 0.3967 | 0.7129 | 0.9459 | 0.9631 | 0.0444 |
| Bilateral | 0.8612 | 0.9964 | 0.9830 | 1.0188 | 0.0713 |
| CLAHE | 14.4669 | 2.9217 | 1.8561 | 0.5575 | 0.2582 |

Максимальная доля algorithmic clipping в измеренных конфигурациях/срезах: 0.000000%. Display clipping меняет только отображение, не численные данные.
На этой train-подвыборке Gaussian sigma=2 сильнее всего снижает median local variance, но ослабляет интерфейсный gradient. Bilateral при выбранных spatial/intensity sigma лучше сохраняет median case gradient response, чем сильный Gaussian; слабый Gaussian 0.5 также близок к baseline. CLAHE резко увеличивает локальные колебания: это усиление контраста, а не шумоподавление.
Homogeneous proxy: GT-область эродируется на 3 pixels, затем сохраняется нижний квартиль baseline local std для каждого класса. Эти области фиксируются до фильтрации. Граница — однопиксельный контакт двух экспертных классов. ROI не переопределяется отдельно для фильтров.
Это не истинная дисперсия шума: texture, partial volume, размытие и масочная неточность влияют на показатель. При sigma=2 даже эрозия 3 pixels не гарантирует отсутствия влияния соседних тканей. CLAHE меняет transfer function, поэтому рост контраста/градиента сам по себе не доказывает сохранение полезной анатомии.

## 3–4. Разделимость LV/MYO и контроль RV/PA

Primary metric: mean case Macro F1 на 10 frozen validation случаях; ± — SD между случаями, не confidence interval. Confusion matrix в results.json/validation_charts.html агрегируется по sampled voxels и имеет другую единицу усреднения. Строки — GT, столбцы — predicted.

| Pair | Filter | Intensity only F1 ± SD | Five-feature F1 ± SD |
|---|---|---:|---:|
| LV / MYO | Baseline | 0.9054 ± 0.1378 | 0.8916 ± 0.1371 |
| LV / MYO | Gaussian 0.5 | 0.9006 ± 0.1361 | 0.8925 ± 0.1375 |
| LV / MYO | Gaussian 1.0 | 0.9027 ± 0.1368 | 0.8910 ± 0.1383 |
| LV / MYO | Gaussian 2.0 | 0.8996 ± 0.1351 | 0.8897 ± 0.1383 |
| LV / MYO | Bilateral | 0.8913 ± 0.1347 | 0.8753 ± 0.1471 |
| LV / MYO | CLAHE | 0.6659 ± 0.1342 | 0.6738 ± 0.1430 |
| RV / PA | Baseline | 0.4590 ± 0.0624 | 0.6003 ± 0.1146 |
| RV / PA | Gaussian 0.5 | 0.4833 ± 0.0515 | 0.6044 ± 0.1127 |
| RV / PA | Gaussian 1.0 | 0.4859 ± 0.0634 | 0.6156 ± 0.1171 |
| RV / PA | Gaussian 2.0 | 0.4929 ± 0.0558 | 0.6497 ± 0.1275 |
| RV / PA | Bilateral | 0.4681 ± 0.0611 | 0.5798 ± 0.1100 |
| RV / PA | CLAHE | 0.5907 ± 0.0702 | 0.6348 ± 0.1050 |

LV/MYO: в этой проверке ни одна обработка не превысила baseline intensity-only k-NN по mean validation F1. Добавление текстурных признаков также не обеспечило общего выигрыша. CLAHE заметно ухудшил разделимость LV/MYO несмотря на визуальное усиление контраста.
RV/PA: интенсивность сама по себе слаба; texture features дают больший F1. Среди заранее заданных вариантов Gaussian sigma=2 + пять признаков имеет наибольший описательный mean F1. CLAHE помогает RV/PA относительно intensity-only baseline, но это не универсальное улучшение: для LV/MYO направление противоположно.
Это сравнение одного фиксированного набора параметров, не доказательство оптимальности sigma=2. Разброс по случаям и выбор только трёх Z-позиций ограничивают вывод. Не выбирать новый параметр и не переоценивать его на тех же validation как независимый эксперимент.

## Development: оптимистичная проверка

Отдельные результаты ct_1001 80/20 voxel holdout сохранены с role="ct_1001 voxel holdout; optimistic". Train scaler fit только по 80%; соседние воксели пространственно зависимы, поэтому такой holdout не независим и не заменяет group validation.

## 5. Возможная потеря информации

Gaussian снижает локальные колебания вместе с gradient response на экспертных границах; при sigma=2 эффект сильнее. Повышение Fisher ratio в части областей не гарантирует переноса на другие случаи. CLAHE увеличивает локальную изменчивость и меняет соотношения source intensities; высокий display contrast может скрывать ухудшение classifier F1.
Histogram overlap считается по всем GT-вокселям выбранной плоскости/ROI, 128 фиксированных bins [0,4095]. Это intensity-only характеристика, не метрика многомерной разделимости. Для контрольной пары контактная граница может отсутствовать; n/a сохранено, а не заменено нулём.

## PCA/UMAP и связанная визуализация

В лаборатории доступны две проекции до/после, одинаковые sampled voxel IDs и seed. Scaler и reducer fit на 80% sampled voxels train/development ct_1001, transform — на оставшихся 20%; labels передаются только цветам. Проекции независимы: абсолютные расстояния/поворот разных UMAP не сопоставимы. Визуальные кластеры не доказывают улучшение сегментации.
linked_ct1001.html — интерактивный отчёт с выбором фильтра и осей признаков, подсветкой source координат при выборе точек, парой PCA/UMAP before/after. PNG/SVG используют одну фиксированную шкалу [0,4095] для всех фильтров; карта изменений — [0,500]. NPY/NPZ хранят точные sampled features и voxel identities.

## 6. Что стоит исследовать при последующем обучении

Предварительно заданная абляция слабого Gaussian (0.5/1) против baseline: компромисс между локальной вариацией и границами. Для RV/PA отдельно исследовать градиент/локальную текстуру как дополнительные каналы. Для сильного Gaussian проверить потерю тонких структур, прежде чем переносить его в модель. CLAHE не считать универсальным кандидатом: проверить case-dependent intensity shifts и пределы его применения.
Эти предложения — гипотезы для нового протокола, а не выбор параметров по test. Исходный CHD68 preprocessing и веса остаются неизменными.

## 7. Чего пока нельзя заключить

Нельзя утверждать увеличение Dice, уменьшение ошибок сегментатора, клиническое улучшение или перенос на полный 3D объём. k-NN решает бинарную классификацию GT-selected voxels; будущий автоматический сегментатор не получает GT для ROI, масочного sampling или входных признаков. Нужны отдельные заранее спланированные segmentation experiments и подтверждённая шкала/геометрия.

## Артефакты и воспроизводимость

`outputs/filtering_feature_01/`: protocol.json, results.json, measurements.csv, scores.csv, equal_case_measurements.json, per-case sampled features/IDs, projection manifests and embeddings, linked_ct1001.html, validation_charts.html, fixed-scale PNG/SVG, UI verification.
Запуск: `python -m ct_lab.filtering_run`; `python -m ct_lab.filtering_artifacts --umap`; `python -m ct_lab.filtering_report`. Исходные medical files не перезаписываются, test volumes не открываются. Synthetic fixtures применяются только в автоматических тестах математики.

Итоговые проверки: 30 автоматических тестов (filtering/features, CT Lab, cardiac inference, без cardiac_training); browser lasso → все source координаты на обоих КТ; автономный HTML filter/PCA/UMAP; UI guard не загружает validation/test для интерактивного fit. Результаты: tests.xml, browser_verification.json, offline_browser_verification.json, ui_verification.json, umap_ui_verification.json.

Отклонение при проверке: в первоначальный расширенный регрессионный прогон был ошибочно включён существующий test_cardiac_training с коротким обучением на синтетических fixtures. При запрете обучения его не следовало запускать. Из итогового набора он исключён; реальные КТ в neural training не использовались, существующие CHD68 checkpoint SHA256 не изменились. Синтетические test outputs не входят в исследовательские таблицы и отчётные метрики.

Фактическое наличие UMAP в последнем сохранённом artifact run: True.
