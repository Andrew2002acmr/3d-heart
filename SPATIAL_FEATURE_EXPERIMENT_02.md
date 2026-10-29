# SPATIAL_FEATURE_EXPERIMENT_02

Дата: 28.10.2026. Реальные ImageCHD68 CT; без обучения нейросетей и без изменения production preprocessing.

## Протокол и воспроизводимость

**Это продолжение разработки на validation, уже использованной на этапе 1; не независимая финальная оценка.** Sealed test CT не открывались.
Сохранены те же 5 train/development groups и 10 validation groups, три Z-позиции 0.35/0.50/0.65 и все sampled IDs/labels/folds из этапа 1. Для каждого release-case feature sets A–H имеют строго одинаковое число наблюдений. Пересечения train/validation release-groups нет.
Train: ct_1001, ct_1002, ct_1004, ct_1008, ct_1072. Validation: ct_1027, ct_1028, ct_1033, ct_1036, ct_1046, ct_1056, ct_1079, ct_1099, ct_1125, ct_1127.
Протокол сохранён до validation extraction/scoring в `outputs/spatial_feature_02/protocol.json`; все 8 основных и 9 фильтрационных конфигураций предзаданы. k=15, distance weighting, exact cKDTree; mean/std fit только train. Никакого global refit CHD68.
NIfTI affine проверен на native axis-aligned RAS; все плоскости получены из одного XYZ volume. Sampled IDs — C-order logical source indices с привязкой к case, не byte offsets. Oblique данные не транспонируются молча: текущий fixed-ID протокол их отклоняет. Spacing не подтверждён как мм, интенсивности не считаются HU.

## Матрица признаков

| Set | Columns | Input features |
|---|---:|---|
| A Intensity | 1 | Intensity |
| B 2D | 5 | Intensity, XY Gradient, XY Mean, XY Std, XY LoG |
| C Z±1 | 13 | Intensity, XY Gradient, XY Mean, XY Std, XY LoG, Z I -1, Z I +1, Z diff -1, Z diff +1, Z1 Mean, Z1 Std, Z1 Gradient, Z1 Range |
| D Z±2 | 17 | Intensity, XY Gradient, XY Mean, XY Std, XY LoG, Z I -2, Z I -1, Z I +1, Z I +2, Z diff -2, Z diff -1, Z diff +1, Z diff +2, Z2 Mean, Z2 Std, Z2 Gradient, Z2 Range |
| E Z±3 | 21 | Intensity, XY Gradient, XY Mean, XY Std, XY LoG, Z I -3, Z I -2, Z I -1, Z I +1, Z I +2, Z I +3, Z diff -3, Z diff -2, Z diff -1, Z diff +1, Z diff +2, Z diff +3, Z3 Mean, Z3 Std, Z3 Gradient, Z3 Range |
| F Tri-planar | 13 | Intensity, XY Gradient, XY Mean, XY Std, XY LoG, XZ Mean, XZ Std, XZ Gradient, XZ LoG, YZ Mean, YZ Std, YZ Gradient, YZ LoG |
| G 3D | 8 | Intensity, 3D Mean, 3D Std, 3D Gradient, 3D LoG, 3D Gaussian 0.5, 3D Gaussian 1, 3D Gaussian 2 |
| H Tri-planar+Z | 29 | Intensity, XY Gradient, XY Mean, XY Std, XY LoG, XZ Mean, XZ Std, XZ Gradient, XZ LoG, YZ Mean, YZ Std, YZ Gradient, YZ LoG, Z I -3, Z I -2, Z I -1, Z I +1, Z I +2, Z I +3, Z diff -3, Z diff -2, Z diff -1, Z diff +1, Z diff +2, Z diff +3, Z3 Mean, Z3 Std, Z3 Gradient, Z3 Range |

2D texture: local mean/std 5×5, central gradient, LoG σ=1 grid unit. Tri-planar: те же четыре величины в XY/XZ/YZ. 3D texture: mean/std в 5³, 3D gradient/LoG; G дополнительно включает Gaussian 3D σ=0.5/1/2. Поэтому выигрыш G нельзя приписать только замене плоскости на 3D: нужны отдельные заранее запланированные ablations Gaussian channels.
Z context каждого радиуса: соседние intensities и differences (neighbor−center), mean/std/range по окну, central Z gradient. Центральная intensity не дублируется в classifier. Raw XYZ profiles экспортируются/показываются отдельно; F не получает скрытые дополнительные Z-profile channels. H явно добавляет Z±3.
На границах profiles/context используют deterministic edge replication, texture — half-sample reflect. Градиенты texture соответствуют np.gradient (one-sided на границе), central context gradient использует replicated ±1. Флаги выхода за Z сохранены отдельно в context_flags_*.npz.

## Основные результаты на frozen validation

Главная метрика — mean case Macro F1; ± — SD по 10 случаям, не confidence interval. F1 сохраняет правило этапа 1: два заранее заданных класса, zero_division=0, включая empty-empty=0. В выбранных трёх срезах некоторых случаев один класс отсутствует; это не означает его отсутствия во всём сердце. Balanced Accuracy усредняет reference-present recalls. Fisher и silhouette усредняются только по случаям, где оба класса доступны; silhouette — до 128 samples/class/case, Euclidean train-standardized space. Fisher — squared standardized mean distance / sum within-class variance traces. Они диагностические, не Dice.

| Pair | Features | Case Macro F1 ± SD | Balanced accuracy | Silhouette | Fisher |
|---|---|---:|---:|---:|---:|
| LV/MYO | A Intensity | 0.9054 ± 0.1378 | 0.9540 | 0.6962 | 6.8034 |
| LV/MYO | B 2D | 0.8916 ± 0.1371 | 0.9364 | 0.4227 | 1.2486 |
| LV/MYO | C Z±1 | 0.8898 ± 0.1363 | 0.9342 | 0.3641 | 0.8351 |
| LV/MYO | D Z±2 | 0.8893 ± 0.1375 | 0.9335 | 0.3662 | 0.8945 |
| LV/MYO | E Z±3 | 0.8895 ± 0.1372 | 0.9339 | 0.3631 | 0.9191 |
| LV/MYO | F Tri-planar | 0.8904 ± 0.1364 | 0.9347 | 0.3247 | 0.6819 |
| LV/MYO | G 3D | 0.8940 ± 0.1378 | 0.9401 | 0.4883 | 1.8235 |
| LV/MYO | H Tri-planar+Z | 0.8899 ± 0.1368 | 0.9344 | 0.3413 | 0.7875 |
| RV/PA | A Intensity | 0.4590 ± 0.0624 | 0.5183 | 0.0729 | 0.1623 |
| RV/PA | B 2D | 0.6003 ± 0.1146 | 0.6380 | 0.1433 | 0.3524 |
| RV/PA | C Z±1 | 0.5982 ± 0.1140 | 0.6308 | 0.1137 | 0.2067 |
| RV/PA | D Z±2 | 0.5967 ± 0.1048 | 0.6291 | 0.1159 | 0.2014 |
| RV/PA | E Z±3 | 0.6052 ± 0.1005 | 0.6378 | 0.1213 | 0.2071 |
| RV/PA | F Tri-planar | 0.6030 ± 0.1097 | 0.6379 | 0.1516 | 0.3697 |
| RV/PA | G 3D | 0.6269 ± 0.1153 | 0.6661 | 0.1570 | 0.3820 |
| RV/PA | H Tri-planar+Z | 0.6018 ± 0.1004 | 0.6334 | 0.1286 | 0.2501 |

Per-class таблица ниже использует только reference-present случаи, поэтому её столбцы не обязаны усредняться в основной fixed-label Macro F1.

| Pair | Class | GT-present cases | B case F1 | G case F1 | B recall | G recall |
|---|---|---:|---:|---:|---:|---:|
| LV/MYO | LV | 9 | 0.9172 | 0.9210 | 0.8967 | 0.9067 |
| LV/MYO | MYO | 10 | 0.9577 | 0.9592 | 0.9672 | 0.9652 |
| RV/PA | RV | 9 | 0.5787 | 0.6264 | 0.5727 | 0.6384 |
| RV/PA | PA | 10 | 0.6798 | 0.6901 | 0.6949 | 0.6903 |

## 1–4. Соседние срезы, три плоскости и 3D

LV/MYO: исходная intensity A остаётся сильнее всех испытанных пространственных k-NN вариантов по среднему F1. Относительно B добавление Z/tri-planar почти не меняет результат и в среднем слегка снижает F1; G немного выше B, но ниже A. Это не основание добавлять все пространственные каналы для LV/MYO.
RV/PA: B с текстурой заметно сильнее A. Z±1/±2 не улучшили B, Z±3 дал небольшой прирост среднего, но меньше межслучайного разброса. Нельзя назвать ±3 универсально оптимальным. Tri-planar F и H дают небольшой средний эффект; полноценный G — самый высокий средний F1 среди основных A–H в данном протоколе.
Эффекты не монотонны по радиусу и не одинаковы у пациентов. Простой список соседних intensities/differences не заменяет 3D representation, а высокий размер feature space может ухудшать k-NN. Все варианты оцениваются на одних и тех же voxels, так что отличие не связано с изменением sample size.

| Pair | Variant vs B | Mean paired case ΔF1 | Cases improved / worsened / equal |
|---|---|---:|---|
| LV/MYO | C Z±1 | -0.00174 | 2 / 7 / 1 |
| LV/MYO | D Z±2 | -0.00230 | 2 / 8 / 0 |
| LV/MYO | E Z±3 | -0.00209 | 3 / 6 / 1 |
| LV/MYO | F Tri-planar | -0.00121 | 3 / 7 / 0 |
| LV/MYO | G 3D | +0.00246 | 6 / 4 / 0 |
| LV/MYO | H Tri-planar+Z | -0.00173 | 3 / 6 / 1 |
| RV/PA | C Z±1 | -0.00206 | 3 / 7 / 0 |
| RV/PA | D Z±2 | -0.00357 | 5 / 5 / 0 |
| RV/PA | E Z±3 | +0.00489 | 5 / 5 / 0 |
| RV/PA | F Tri-planar | +0.00271 | 5 / 5 / 0 |
| RV/PA | G 3D | +0.02662 | 7 / 3 / 0 |
| RV/PA | H Tri-planar+Z | +0.00152 | 4 / 6 / 0 |

## 5. Фильтрация: отдельный контроль

Диагностическая подвыборка: фиксированные первые до 32 исходных sampled ID каждого класса на каждом выбранном Z; одинакова у всех девяти фильтрационных вариантов и всех diagnostic feature sets. Она меньше основной выборки: её F1 сравнивается только с её собственным Baseline, а не с основной таблицей A–H.
Gaussian σ=0.5/1/2 в XY посрезово и согласованно в XYZ; truncate=4. Bilateral spatial σ=1, intensity σ=100 source units, truncate=3 в обеих размерностях. Здесь bilateral — exact finite kernel; этап 1 использовал skimage bins=10000, поэтому его F1 не считается побайтово идентичным этой реализации. CLAHE не включён.
В diagnostic 3D-core — пять каналов (Intensity + 3D gradient/mean/std/LoG), без Gaussian auxiliary channels G основной матрицы.

| Pair | Filter | 2D texture F1 | 3D-core F1 |
|---|---|---:|---:|
| LV/MYO | Baseline | 0.8837 | 0.8941 |
| LV/MYO | Gaussian 0.5 2D | 0.8831 | 0.8941 |
| LV/MYO | Gaussian 0.5 3D | 0.8839 | 0.8949 |
| LV/MYO | Gaussian 1 2D | 0.8822 | 0.8934 |
| LV/MYO | Gaussian 1 3D | 0.8796 | 0.8917 |
| LV/MYO | Gaussian 2 2D | 0.8871 | 0.9040 |
| LV/MYO | Gaussian 2 3D | 0.8774 | 0.8889 |
| LV/MYO | Bilateral 2D | 0.8812 | 0.8946 |
| LV/MYO | Bilateral 3D | 0.8778 | 0.8914 |
| RV/PA | Baseline | 0.5801 | 0.6075 |
| RV/PA | Gaussian 0.5 2D | 0.5860 | 0.5971 |
| RV/PA | Gaussian 0.5 3D | 0.5828 | 0.6017 |
| RV/PA | Gaussian 1 2D | 0.6099 | 0.6039 |
| RV/PA | Gaussian 1 3D | 0.6113 | 0.6180 |
| RV/PA | Gaussian 2 2D | 0.6284 | 0.6300 |
| RV/PA | Gaussian 2 3D | 0.6404 | 0.6737 |
| RV/PA | Bilateral 2D | 0.5885 | 0.5970 |
| RV/PA | Bilateral 3D | 0.5923 | 0.5971 |

Для RV/PA Gaussian σ=2 3D + 3D-core показывает более высокий описательный F1 на этой reduced-проверке, чем её baseline. Для LV/MYO согласованная 3D-фильтрация не даёт универсального выигрыша, а сильное сглаживание может терять детали. Это secondary development finding; параметры по результатам не перестраивались и вывод требует отдельного подтверждения.

## 6. Где исправляются ошибки

Distance-to-GT вычислена после extraction, отдельно от feature bank: расстояние до центра ближайшего 6-connected boundary voxel своего класса, в header grid units. Это не точное расстояние до непрерывной анатомической поверхности и не мм. В k-NN/PCA/UMAP это поле не передаётся.
Ниже validation counts: corrected = B wrong → variant correct; new wrong = B correct → variant wrong. Counts суммируются по sampled voxels, это не case-macro F1. У границ и в центре имеются разные baseline error counts; сырые числа без знаменателя нельзя трактовать как предпочтительную область улучшения.

| Pair | Variant | Region | Baseline wrong | Corrected | New wrong | Mean case correction fraction |
|---|---|---|---:|---:|---:|---:|
| LV/MYO | E | boundary ≤1 | 604 | 114 | 132 | 0.2225 |
| LV/MYO | E | transition (1,3] | 88 | 24 | 25 | 0.3469 |
| LV/MYO | E | interior >3 | 137 | 40 | 42 | 0.4499 |
| LV/MYO | F | boundary ≤1 | 604 | 92 | 106 | 0.1613 |
| LV/MYO | F | transition (1,3] | 88 | 21 | 20 | 0.2854 |
| LV/MYO | F | interior >3 | 137 | 36 | 39 | 0.2925 |
| LV/MYO | G | boundary ≤1 | 604 | 148 | 107 | 0.2393 |
| LV/MYO | G | transition (1,3] | 88 | 29 | 15 | 0.4310 |
| LV/MYO | G | interior >3 | 137 | 39 | 50 | 0.4206 |
| RV/PA | E | boundary ≤1 | 1546 | 772 | 735 | 0.4923 |
| RV/PA | E | transition (1,3] | 1539 | 661 | 605 | 0.4175 |
| RV/PA | E | interior >3 | 3320 | 811 | 874 | 0.2347 |
| RV/PA | F | boundary ≤1 | 1546 | 674 | 668 | 0.4311 |
| RV/PA | F | transition (1,3] | 1539 | 584 | 611 | 0.3603 |
| RV/PA | F | interior >3 | 3320 | 735 | 741 | 0.2183 |
| RV/PA | G | boundary ≤1 | 1546 | 736 | 637 | 0.4810 |
| RV/PA | G | transition (1,3] | 1539 | 660 | 511 | 0.4145 |
| RV/PA | G | interior >3 | 3320 | 870 | 720 | 0.2584 |

Проверка показывает и исправления, и новые ошибки во внутренних областях, а не только на границе. Нет оснований объявлять эффект исключительно boundary denoising. Spatial Error Maps доступны для заранее выбранного ct_1001 (optimistic train voxel holdout) и первого validation case ct_1027; полные sampled predictions/errors сохранены по всем 15 случаям.

## 7–8. Полезность по классам и избыточность

Наибольшая полезность в проверенных парах — у сложной RV/PA; для LV/MYO более высокое исходное intensity separation оставляет меньше возможностей для выбранного k-NN. Это вывод о бинарных GT-selected samples, не о качестве multiclass сегментации.
В reference-present class-анализе основная прибавка G относительно B у RV: F1 и recall растут заметнее, чем у PA. У PA F1 меняется слабо, recall может слегка снизиться; нельзя приписать одинаковую пользу обоим классам.
Z differences линейно выражаются через соседнюю и центральную intensity; context gradient повторяется у разных радиусов, а mean/std/range могут быть сильно коррелированы. Добавление всех таких столбцов не равно добавлению независимой информации.

| Pair | Train feature 1 | Train feature 2 | Pearson r |
|---|---|---|---:|
| LV/MYO | Z2 Gradient | Z3 Gradient | 1.00000 |
| LV/MYO | Z1 Gradient | Z3 Gradient | 1.00000 |
| LV/MYO | Z1 Gradient | Z2 Gradient | 1.00000 |
| LV/MYO | Intensity | 3D Gaussian 0.5 | 0.99964 |
| LV/MYO | Z1 Std | Z1 Range | 0.99925 |
| LV/MYO | 3D Gaussian 0.5 | Z1 Mean | 0.99914 |
| LV/MYO | XZ Mean | 3D Mean | 0.99908 |
| LV/MYO | YZ Mean | 3D Mean | 0.99904 |
| LV/MYO | YZ Mean | Z2 Mean | 0.99892 |
| LV/MYO | XZ Mean | Z2 Mean | 0.99889 |
| LV/MYO | 3D Gaussian 1 | Z1 Mean | 0.99883 |
| LV/MYO | XZ Mean | YZ Mean | 0.99851 |
| RV/PA | Z2 Gradient | Z3 Gradient | 1.00000 |
| RV/PA | Z1 Gradient | Z3 Gradient | 1.00000 |
| RV/PA | Z1 Gradient | Z2 Gradient | 1.00000 |
| RV/PA | Z1 Std | Z1 Range | 0.99922 |
| RV/PA | Z2 Std | Z2 Range | 0.99865 |
| RV/PA | Intensity | 3D Gaussian 0.5 | 0.99827 |
| RV/PA | 3D Gaussian 0.5 | Z1 Mean | 0.99712 |
| RV/PA | Z3 Std | Z3 Range | 0.99647 |
| RV/PA | XZ Mean | 3D Mean | 0.99603 |
| RV/PA | YZ Mean | 3D Mean | 0.99504 |
| RV/PA | YZ Mean | Z2 Mean | 0.99486 |
| RV/PA | XZ Mean | Z2 Mean | 0.99386 |

## Проекции и интерфейс

Spatial Feature Explorer связывает Canvas viewer трёх плоскостей с одним source XYZ и affine; щелчок по любой плоскости обновляет crosshair, scatter selection подсвечивает source voxels. Сплошные маркеры — в текущем срезе, кольца — проекция выбранных voxels других срезов, а не утверждение, что они лежат в этой плоскости. Error overlay рисует только in-plane evaluated voxels.
Independent PCA/UMAP до/после fit только на train voxel fold ct_1001; IDs одинаковы, seed=20261028, UMAP neighbors=15/min_dist=0.1/epochs=200, n_jobs=1. Независимые basis нельзя сравнивать по абсолютным расстояниям.
Shared projection реализована на пересечении реально измеренных столбцов B∩context. В неизменённых общих столбцах before/after совпадают: это математически корректный negative control. Он не отображает пользу исключённых дополнительных осей; применять одну PCA к матрицам разной размерности без явного отображения нельзя. Нулевые/выдуманные context channels не подставляются.
PCA при одном столбце имеет нулевую вторую координату. Проекции — визуальная диагностика, не доказательство роста качества.

## Производительность и корректность

CPU-only, float32 retained features, временные float64 moments; координатные блоки по 64 points, filtered blocks по 16. Halo = feature kernel radius + filter radius. Вычисляются только локальные source patches, а не dense feature tensor на всём наборе. Память ограничена размером patch/chunk, результаты case-cached.
Тесты сравнивают sampled и плотные 2D/3D расчёты, разные chunk sizes/halos, global boundaries, Gaussian и exact bilateral, voxel IDs и world-coordinate соответствие, train-only fit, отсутствие GT features и неизменность файлов. Training-тесты нейросетей не включаются.
Предыдущие результаты: 205 файлов этапа 1 и аудита совпали по SHA256. Existing CHD68 weights, frozen split и preprocessing совпали по SHA256.

## 9. Рекомендации по следующему этапу магистерской

Сначала зафиксировать отдельный ablation-протокол: intensity baseline, 2D texture, 3D-core без auxiliary Gaussian и 3D-core с ними. Для RV/PA проверить согласованное 3D сглаживание и learned spatial representation; сравнить boundary/interior ошибки и тонкие структуры. Не выбирать радиус только по среднему этой reused validation.
Перед физически заданным контекстом подтвердить исходную геометрию/spacing. Отдельно исследовать уменьшение избыточности context channels и объём k-NN training samples. Обучение настоящего сегментатора — только отдельным следующим этапом после фиксации гипотез, без GT-distance/масочных признаков и без изменений sealed test для подбора.

## 10. Что не подтверждено

Не подтверждены рост Dice, польза 3D для всех классов/случаев, оптимальность конкретного sigma/radius, независимая клиническая обобщаемость и результаты whole-volume segmentation. Три Z-позиции и ограниченные samples не покрывают всё сердце. Validation уже участвовала в разработке, поэтому следующий подтверждающий эксперимент должен учитывать это ограничение.

## Артефакты и команды

`outputs/spatial_feature_02/`: frozen protocol, main/filter scores JSON/CSV, by_case.csv, separate train_development.json, sampled features/IDs/context flags, validation_predictions.npz, errors_*.npz, error_diagnostics.json, projection arrays/manifests, charts.html и интерактивные графики всех 10 запрошенных типов, PNG/SVG, runtime/provenance.
Команды из repository: `python -m ct_lab.spatial_run`; `python -m ct_lab.spatial_artifacts`; `python -m ct_lab.spatial_report`; `python -m ct_lab.spatial_launch` (local UI 8510). Повторные запуски используют frozen case caches; изменение протокола поверх завершённого эксперимента отклоняется.

## Финальная проверка

Пройдено 39 автоматических тестов без neural training. UI integration: ui_verification.json. Реальные browser clicks во всех трёх плоскостях и scatter lasso: browser_verification.json. Проверены 15 исходных CT/mask пар, 205 файлов этапа 1 и все production guards. Версии библиотек — environment.json; resource sampling — cpu_benchmark.json.
