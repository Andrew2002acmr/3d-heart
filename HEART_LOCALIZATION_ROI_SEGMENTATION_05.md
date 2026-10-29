# HEART_LOCALIZATION_ROI_SEGMENTATION_05

Дата: 2026-10-29. В существующем Heart3D выполнено новое ограниченное обучение локализатора и 2.5D ROI-сегментаторов. CHD68 v1/v2 только читались. Development уже использовалась ранее; это продолжение разработки, не независимая финальная оценка.

Основное сравнение на 10 development cases: Full256 Dice=0.6427, Predicted256=0.6380; Δ=-0.0047, case bootstrap CI95=[-0.0348, 0.0224]. При 384 Δ=-0.0559. Локализатор сохраняет в среднем 99.892% foreground в 55.1% FOV, но полностью все foreground voxels — только в 2/10 случаях. Основной вариант, выбранный предварительной train-only CV: B_full_256. Это ограниченный пилот с одним seed; высокая coverage не гарантирует выигрыш Dice.

## Ответы на 14 исследовательских вопросов

1. Нормализованные центры масс train различаются по X/Y/Z: min=[0.466, 0.287, 0.363], max=[0.67, 0.54, 0.566]. Размеры сосудистых bbox заметно варьируют; фиксированная ROI по train statistics почти равна всему FOV.

2. GT-bbox занимает в среднем 0.350 FOV, диапазон [0.070,0.977]. Предсказанная development ROI занимает в среднем 0.551; этого нельзя считать минимально необходимым объёмом или гарантией безопасности.

3. Потеря деталей измерена после прямого и обратного преобразования изображения и масок на трёх train КТ. ct_1001: Full FOV256 Dice=0.9662, gradient=0.923, Full FOV384 Dice=0.9845, gradient=0.960, Oracle ROI256 Dice=0.9662, gradient=0.923, Oracle ROI384 Dice=0.9845, gradient=0.960; ct_1002: Full FOV256 Dice=0.9707, gradient=0.944, Full FOV384 Dice=0.9844, gradient=0.971, Oracle ROI256 Dice=0.9736, gradient=0.965, Oracle ROI384 Dice=0.9902, gradient=0.979; ct_1004: Full FOV256 Dice=0.9639, gradient=0.910, Full FOV384 Dice=0.9785, gradient=0.952, Oracle ROI256 Dice=0.9639, gradient=0.910, Oracle ROI384 Dice=0.9785, gradient=0.952. Это ошибки resampling без нейросети; визуальное увеличение само по себе не доказывает пользу для сегментации.

4. Локализатор: development bbox IoU=0.5927, center distance=39.39 grid indices; основной показатель coverage приведён отдельно.

5. Foreground coverage development=0.998916. Это среднее по случаям; confidence coarse foreground не калиброван как вероятность такого покрытия.

6. Полное сохранение всех foreground voxels: 2/10. Полностью исчезнувшие из ROI классы: нет. Худшее покрытие класса: 0.971866 (ct_1033, AO). Fallback к fixed ROI: 0/10. Частичное обрезание нужно учитывать независимо от высокого среднего coverage.

7. Predicted ROI−Full FOV: native Dice Δ256=-0.0047, CI95=[-0.0348, 0.0224]; Δ384=-0.0559, CI95=[-0.0854, -0.0278]. При 256 результаты близки, при 384 Predicted ROI хуже Full FOV в данном бюджете. Это не устанавливает результат более длительного обучения.

8. Oracle256 Dice=0.6845, automatic D256=0.6380. Oracle−Full256: Δ=+0.0417, CI95=[0.0051, 0.0798]. Oracle явно использует GT и не является автоматическим результатом или математически гарантированной верхней границей обученной модели. Разницу Oracle/Predicted нельзя объяснить только потерей GT-вокселей: отличаются bbox size, контекст и training distribution.

9. В среднем по development число ложных 6-связных foreground-компонент: Full256=1342.0, Predicted256=732.4; их объём: 355182 → 315573 вокселей. Средний recall по присутствующим классам: 0.7001 → 0.6374. Меньше компонент само по себе не означает лучшую анатомию: одновременно нужно учитывать пропуски. Компоненты не удалялись.

10. Dice на исходной сетке: B256=0.6427, D256=0.6380, B384=0.5904, D384=0.5344. Исторические CHD68 v1/v2 — отдельный reference с иным training budget.

11. Изменения среднего class Dice D256−B256: LV=-0.0168, RV=-0.0429, LA=-0.0360, RA=+0.0147, MYO=+0.0301, AO=+0.0305, PA=-0.0126. При 384: LV=-0.1529, RV=-0.0966, LA=-0.0736, RA=-0.0383, MYO=-0.0110, AO=-0.0291, PA=+0.0100. Это описательные результаты одного seed, а не подтверждённое ранжирование пользы для классов.

12. Дополнительная 3D-core ветвь: Δ native Dice=-0.0154, CI95=[-0.0741, 0.0234]. Ветвь отключаемая, без Gaussian2 и старой MLP; число параметров увеличено на 896. Преимущество ветви в этом пилоте не подтверждено; одного seed недостаточно для устойчивого вывода.

13. Для клинических КТ не подтверждены HU/physical spacing mapping, scanner generalization и coverage guarantee; cohort содержит release-cases с непроверенной пациентной уникальностью. Sealed test не использован, development reused; независимого клинического подтверждения нет.

14. Сохранить: coverage-oriented localizer, train-only margin/CV, точное source-grid восстановление вероятностей, oracle labels, matched B/C/D budget, raw topology/error maps, CPU/GPU gates и epoch resume. Более длительное обучение/несколько seed и внешняя проверка — отдельный следующий этап, а не вывод из этого пилота.

## Протокол, CV и вычислительный бюджет

Frozen split: 48 train / 10 development / 10 sealed test. Test КТ не открывались. Исходные значения 0–4095 не интерпретируются как HU. Affine учитывается при преобразовании координат и поверхностей; расстояния указаны в единицах индексной сетки. Метки: 0 Background, 1 LV, 2 RV, 3 LA, 4 RA, 5 MYO, 6 AO, 7 PA; остальные → ignore=255, не background. Исходные файлы, веса и production preprocessing сохранены.

Исходная предрегистрация: protocol.json; последующее предварительное решение только по train: training_gate.json. CV по 6 эпох × 48 шагов предсказывала только background. После проверки targets, геометрии и обновлений optimizer бюджет симметрично продлён до 18 × 48 только внутри train. Начальные результаты сохранены. Основные модели: 20 эпох × 192 шага × batch 2; модель, loss, margin и матрица экспериментов не менялись по внешней оценке сегментации. Train-only selection: {"train_only_CV_means": {"B_full_256": 0.18792406135756937, "D_predicted_256": 0.14757766544202722}, "primary_pipeline": "B_full_256", "primary_resolution": 256, "scope": "selected before outer development comparisons; 384 and core branch remain predeclared diagnostic ablations, not outer-selected defaults"}

Локализатор: три группы CV внутри train; лучшая эпоха по binary Dice на holdout своей группы, итоговое число эпох на всех 48 train — медиана этих трёх эпох. Margin выбран по train OOF coverage. Порог 0.4, крупнейшая foreground-компонента; пустой результат или средняя confidence < 0.5 → fixed train ROI. Fixed ROI: 2-й и 98-й процентили нормализованных train bbox плюс 0.05 FOV. Development ROI не исправляются по GT. В CV сегментатора используется локализатор своего fold и заранее заданный консервативный margin 0.3 + 8 индексов; main использует выбранный OOF margin 0.1 + 8. Это CV для разработки с выбором checkpoint и параметров, не независимая nested оценка.

| Fold | Pipeline | Native holdout Dice | Cases |
|---|---|---|---|
| fold0 | B_full_256 | 0.1719 | 16 |
| fold0 | D_predicted_256 | 0.1727 | 16 |
| fold1 | B_full_256 | 0.2061 | 16 |
| fold1 | D_predicted_256 | 0.1283 | 16 |
| fold2 | B_full_256 | 0.1857 | 16 |
| fold2 | D_predicted_256 | 0.1417 | 16 |

Бюджет CV — 864 optimizer steps и 32 train cases на fold; основной запуск — 3840 steps и 48 train cases. Абсолютные CV/development Dice поэтому не сопоставляются как оценка generalization gap. Предварительное ранжирование при малом бюджете также не гарантирует того же ранжирования при длительном обучении.

Исторический clip 0–2015 заранее зафиксирован по 48 train в прежнем протоколе; на внешней development он не пересчитывается. Внутренняя CV наследует этот clip и поэтому также не является полностью независимой nested оценкой.

| Model | Parameters | Epochs | Steps/epoch | Batch | GPU peak MiB |
|---|---|---|---|---|---|
| B_full_256 | 489112 | 20 | 192 | 2 | 283.6738 |
| D_predicted_256 | 489112 | 20 | 192 | 2 | 283.6738 |
| C_oracle_256 | 489112 | 20 | 192 | 2 | 283.6738 |
| B_full_384 | 489112 | 20 | 192 | 2 | 560.0957 |
| D_predicted_384 | 489112 | 20 | 192 | 2 | 560.0957 |
| D_core_256 | 490008 | 20 | 192 | 2 | 310.1938 |

Архитектура HeartUNet25D: каналы 16/32/64/128, 489112 параметров; пять согласованных axial-срезов [-2, -1, 0, +1, +2] → 8 logits. Для B/C/D одного разрешения одинаковы параметры, loss, optimizer, augmentation, выбранные случаи и Z-позиции, число обновлений. При 384 увеличиваются FLOPs; равен бюджет обновлений и примеров, не время. Ветвь добавляет 4 native scalar maps и fusion на первом encoder-блоке, без Gaussian и прежних MLP weights. AdamW: lr=0.001, weight_decay=0.0001. Loss CE+Dice — float32; CUDA forward — AMP float16 с GradScaler, CPU — float32. Main best.pt содержит фиксированную последнюю эпоху; development не выбирает checkpoint. Исторические модели A имеют другой бюджет и FP32 inference; контролируемое сравнение проводится между новыми B/C/D.

3D-core использует прежний CT-only extractor: Mean/Std по окну 5³; Gradient — центральные разности, на source edges односторонние; LoG — сумма вторых производных Gaussian с sigma=1 в индексной сетке этого набора, truncate=4. Отдельный Gaussian2 preprocessing/channel не добавляется. Source reflect и halo 8 индексов согласуют блоки; evaluation считает по 8 центральных Z за раз. Ограниченные patch intermediates прежнего extractor имеют float64 для устойчивости variance, результат — float32; кэш входов сети — float16. Маски — uint8/ignore255. Кэш 3D-core проверяет хэш исходных значений и геометрию, записывается атомарно и защищён file lock.

Core рассчитывается на исходных source counts, без clip 2015: Mean / 2015 × 2 − 1, остальные каналы / 512. Центральные CT-каналы B/C/D используют общий исторический clip 0–2015. Поэтому branch-абляция проверяет весь дополнительный pipeline: 3D-сводки, сведения о высоких интенсивностях и 896 новых параметров одновременно. Её возможный выигрыш нельзя приписать только 3D-контексту; clip-matched и parameter-matched controls остаются следующим исследованием.

Каждая main модель получила 7680 обучающих примеров с циклическим равным отбором случаев. На случай: 12 равномерных Z-позиций внутри объединённого train GT bbox и до 4 фоновых позиций снаружи. Центры одинаковы во всех режимах; GT применяется только к отбору train. Сдвиг интенсивности равномерный ±0.03 нормализованной шкалы (±30.225 source counts), Gaussian noise std=0.01 (10.075 counts). Одна фиксированная реализация на случай общая всем конфигурациям; вероятность выбора clean/aug — 50/50. Native core пересчитан на той же изменённой CT. Далее выполняется общий пространственный warp каналов и nearest targets: ±5°, масштаб ±5%, сдвиг ±3%, без flips и elastic. После 2D warp карты core интерполируются совместно; это не новое точное физическое 3D-измерение.

Для согласованного training sampling глобальные Z-центры не отбрасываются даже вне predicted bbox по Z: применяется один XY crop и тот же набор центров B/C/D. При inference ограничивается также диапазон выходных центральных Z, а остальная source mask заполняется background. Это вариант 2.5D XY-ROI training с 3D ROI output gate, а не обучение только на строго вырезанных Z-центрах. Различие training/inference sampling раскрывается как ограничение протокола.

| Smoke device | Size | Batch | Mean s/step | Loss before | Loss after |
|---|---|---|---|---|---|
| cpu | 64 | 1 | 0.0711 | 3.0158 | 2.6899 |
| cuda | 256 | 2 | 0.0834 | 3.0530 | 2.4315 |
| cuda | 384 | 2 | 0.1463 | 3.0489 | 2.4297 |

Seed фиксирован; CUDA loss/interpolation предупреждают об отсутствии bitwise deterministic реализации. CPU seed/optimizer reproducibility протестирована. Main только один seed; train-only folds не заменяют повторные final runs.

## Геометрия, coverage и resampling

| Role | Localizer | Coverage mean | BBox IoU | Center distance index | ROI/FOV | All foreground fully covered |
|---|---|---|---|---|---|---|
| train | fixed | 1.0000 | 0.3520 | 35.3852 | 0.9922 | 47 |
| train | predicted | 0.9987 | 0.6125 | 35.2514 | 0.4777 | 7 |
| validation | fixed | 1.0000 | 0.3645 | 34.9988 | 0.9922 | 10 |
| validation | predicted | 0.9989 | 0.5927 | 39.3855 | 0.5512 | 2 |

| Class | Development mean coverage |
|---|---|
| LV | 1.0000 |
| RV | 1.0000 |
| LA | 0.9965 |
| RA | 1.0000 |
| MYO | 1.0000 |
| AO | 0.9961 |
| PA | 0.9960 |

| Train OOF margin | Mean coverage | Worst case coverage | Mean ROI/FOV |
|---|---|---|---|
| 0.1000 | 0.9999 | 0.9978 | 0.7844 |
| 0.2000 | 1.0000 | 1.0000 | 0.8762 |
| 0.3000 | 1.0000 | 1.0000 | 0.9319 |

Координаты, размеры, центры, расстояния до границ и class bbox: train_geometry.json/csv. Margin — доля bbox плюс 8 индексов, не мм. Bbox полуоткрытый [lo, hi). Coarse cube 64³ используется только для локализации с последующим обратным mapping по исходному shape; это нормализованная сетка, не физически изотропное изображение. Z-контекст сегментатора сохраняет исходную сетку. При inference центральные срезы ограничены predicted bbox по Z, но соседние ±2 берутся с halo из исходного объёма. На границе source повторяется крайний срез.

| Case | Mode | Size | XY scale | Mask-only roundtrip Dice | Native gradient ratio | Image MAE counts | Thin MYO recall |
|---|---|---|---|---|---|---|---|
| ct_1001 | Full FOV | 256 | [0.5000, 0.5000] | 0.9662 | 0.9234 | 7.7678 | 0.8688 |
| ct_1001 | Full FOV | 384 | [0.7500, 0.7500] | 0.9845 | 0.9604 | 3.9993 | 0.8985 |
| ct_1001 | Oracle ROI | 256 | [0.5000, 0.5000] | 0.9662 | 0.9234 | 7.7678 | 0.8688 |
| ct_1001 | Oracle ROI | 384 | [0.7500, 0.7500] | 0.9845 | 0.9604 | 3.9993 | 0.8985 |
| ct_1002 | Full FOV | 256 | [0.5000, 0.5000] | 0.9707 | 0.9441 | 5.2725 | 0.8478 |
| ct_1002 | Full FOV | 384 | [0.7500, 0.7500] | 0.9844 | 0.9715 | 2.7053 | 0.9329 |
| ct_1002 | Oracle ROI | 256 | [0.5689, 0.5680] | 0.9736 | 0.9650 | 4.6417 | 0.8779 |
| ct_1002 | Oracle ROI | 384 | [0.8533, 0.8521] | 0.9902 | 0.9791 | 2.7891 | 0.9347 |
| ct_1004 | Full FOV | 256 | [0.5000, 0.5000] | 0.9639 | 0.9097 | 10.1770 | 0.8556 |
| ct_1004 | Full FOV | 384 | [0.7500, 0.7500] | 0.9785 | 0.9521 | 5.2898 | 0.9041 |
| ct_1004 | Oracle ROI | 256 | [0.5000, 0.5000] | 0.9639 | 0.9097 | 10.1770 | 0.8556 |
| ct_1004 | Oracle ROI | 384 | [0.7500, 0.7500] | 0.9785 | 0.9521 | 5.2898 | 0.9041 |

Thin MYO: диагностические GT pixels с 2 × distance-to-background ≤ 4 индексов; это не клиническая толщина стенки. Ошибки resampling и изменения площадей сохранены отдельно. После inverse ignore остаётся ignore; valid → ignore учитывается как unclassified/FN в геометрическом Dice. Основные маски получают argmax после обратной интерполяции вероятностей. B против D одновременно меняет детализацию, анатомический контекст и ограничение FOV. Контроль full-model + ROI-gating отдельно не выполнялся, поэтому эффект нельзя приписать только визуальной детализации.

## Сегментация на исходной сетке

Все итоговые mean/SD/CI в таблицах ниже относятся ровно к 10 development cases. Train ct_1001 показан отдельной колонкой. Сырые evaluation summary содержат mean по всем запрошенным 11 случаям; это общий технический агрегат и не development-метрика. Для выводов используются case_ids и role в study_summary.json, segmentation_cases.csv и таблицы ниже.

| Method | Train ct1001 Dice | Development mean Dice | Case SD | Case CI95 |
|---|---|---|---|---|
| B_full_256 | 0.7016 | 0.6427 | 0.0561 | [0.6062, 0.6728] |
| D_predicted_256 | 0.7533 | 0.6380 | 0.0694 | [0.5954, 0.6789] |
| C_oracle_256 | 0.8214 | 0.6845 | 0.0857 | [0.6331, 0.7368] |
| B_full_384 | 0.6458 | 0.5904 | 0.0605 | [0.5525, 0.6249] |
| D_predicted_384 | 0.5975 | 0.5344 | 0.0594 | [0.4972, 0.5675] |
| D_core_256 | 0.7689 | 0.6226 | 0.1162 | [0.5471, 0.6873] |
| CHD68_v1 | 0.9589 | 0.7978 | 0.0655 | [0.7576, 0.8347] |
| CHD68_v2 | 0.9394 | 0.7952 | 0.0663 | [0.7549, 0.8340] |

| Method | LV | RV | LA | RA | MYO | AO | PA |
|---|---|---|---|---|---|---|---|
| B_full_256 | 0.6632 | 0.6494 | 0.6454 | 0.7552 | 0.7823 | 0.5992 | 0.4046 |
| D_predicted_256 | 0.6464 | 0.6066 | 0.6094 | 0.7699 | 0.8124 | 0.6297 | 0.3920 |
| C_oracle_256 | 0.7112 | 0.6882 | 0.6711 | 0.7537 | 0.7872 | 0.6883 | 0.4914 |
| B_full_384 | 0.6381 | 0.4967 | 0.5787 | 0.7127 | 0.7833 | 0.5721 | 0.3509 |
| D_predicted_384 | 0.4852 | 0.4001 | 0.5050 | 0.6744 | 0.7723 | 0.5430 | 0.3609 |
| D_core_256 | 0.6244 | 0.6374 | 0.5641 | 0.7485 | 0.7670 | 0.6163 | 0.4005 |
| CHD68_v1 | 0.8722 | 0.8185 | 0.7893 | 0.8464 | 0.8537 | 0.7992 | 0.6055 |
| CHD68_v2 | 0.8594 | 0.8090 | 0.7821 | 0.8365 | 0.8488 | 0.7968 | 0.6340 |

| Paired comparison | Mean ΔDice | Case CI95 |
|---|---|---|
| D_predicted_256 − B_full_256 | -0.0047 | [-0.0348, 0.0224] |
| D_predicted_384 − B_full_384 | -0.0559 | [-0.0854, -0.0278] |
| C_oracle_256 − D_predicted_256 | 0.0464 | [0.0263, 0.0695] |
| D_core_256 − D_predicted_256 | -0.0154 | [-0.0741, 0.0234] |
| B_full_384 − B_full_256 | -0.0524 | [-0.0771, -0.0306] |
| D_predicted_384 − D_predicted_256 | -0.1036 | [-0.1349, -0.0725] |
| C_oracle_256 − B_full_256 | 0.0417 | [0.0051, 0.0798] |

Case bootstrap: 2000 выборок из 10 development случаев; воксели и срезы не считаются независимыми наблюдениями. Case SD описывает эти десять случаев (ddof=0). CI описательные, без поправки на повторное использование development и множественные сравнения. Foreground-классы без GT исключены из macro; empty-empty не получает награду, а FP-объёмы сохраняются. Все методы оцениваются на одинаковых исходных CT/GT и valid 0–7 после восстановления исходной сетки. Oracle C использует GT bbox и всегда выделен отдельно.

## Фрагментация, HD95 и 3D

| Method | Case | FG macro Dice | False FG components | False component voxels | BG→FG voxels |
|---|---|---|---|---|---|
| B_full_256 | ct_1001 | 0.7016 | 1409 | 258090 | 855894 |
| B_full_256 | ct_1027 | 0.4963 | 2086 | 810325 | 866415 |
| B_full_256 | ct_1028 | 0.6491 | 874 | 110815 | 607259 |
| B_full_256 | ct_1033 | 0.6095 | 1437 | 143785 | 529231 |
| B_full_256 | ct_1036 | 0.6710 | 897 | 170895 | 669154 |
| B_full_256 | ct_1046 | 0.6914 | 1887 | 381311 | 704189 |
| B_full_256 | ct_1056 | 0.6277 | 2316 | 1028849 | 1381495 |
| B_full_256 | ct_1079 | 0.6910 | 1051 | 256405 | 480354 |
| B_full_256 | ct_1099 | 0.6565 | 1197 | 165995 | 381522 |
| B_full_256 | ct_1125 | 0.6969 | 411 | 61035 | 116661 |
| B_full_256 | ct_1127 | 0.6380 | 1264 | 422400 | 749347 |
| CHD68_v1 | ct_1001 | 0.9589 | 216 | 18914 | 162521 |
| CHD68_v1 | ct_1027 | 0.6498 | 1082 | 580990 | 390375 |
| CHD68_v1 | ct_1028 | 0.8347 | 127 | 24312 | 322598 |
| CHD68_v1 | ct_1033 | 0.8700 | 208 | 90197 | 121822 |
| CHD68_v1 | ct_1036 | 0.8797 | 140 | 17165 | 165550 |
| CHD68_v1 | ct_1046 | 0.8650 | 243 | 63981 | 380904 |
| CHD68_v1 | ct_1056 | 0.7735 | 826 | 765816 | 603274 |
| CHD68_v1 | ct_1079 | 0.7901 | 214 | 145763 | 155194 |
| CHD68_v1 | ct_1099 | 0.7833 | 292 | 100011 | 145995 |
| CHD68_v1 | ct_1125 | 0.7525 | 72 | 24881 | 48165 |
| CHD68_v1 | ct_1127 | 0.7798 | 199 | 43304 | 329574 |
| CHD68_v2 | ct_1001 | 0.9394 | 398 | 45819 | 229368 |
| CHD68_v2 | ct_1027 | 0.6513 | 1271 | 619645 | 290752 |
| CHD68_v2 | ct_1028 | 0.8294 | 139 | 10578 | 351550 |
| CHD68_v2 | ct_1033 | 0.8649 | 279 | 73733 | 126808 |
| CHD68_v2 | ct_1036 | 0.8845 | 248 | 62441 | 226963 |
| CHD68_v2 | ct_1046 | 0.8628 | 439 | 126866 | 401081 |
| CHD68_v2 | ct_1056 | 0.7370 | 1377 | 1116502 | 771969 |
| CHD68_v2 | ct_1079 | 0.8001 | 346 | 108915 | 151655 |
| CHD68_v2 | ct_1099 | 0.7720 | 346 | 215405 | 143035 |
| CHD68_v2 | ct_1125 | 0.7747 | 91 | 13944 | 57134 |
| CHD68_v2 | ct_1127 | 0.7757 | 256 | 117545 | 330702 |
| C_oracle_256 | ct_1001 | 0.8214 | 600 | 130312 | 107716 |
| C_oracle_256 | ct_1027 | 0.5152 | 1258 | 509549 | 188639 |
| C_oracle_256 | ct_1028 | 0.7253 | 420 | 77788 | 187511 |
| C_oracle_256 | ct_1033 | 0.6636 | 463 | 67480 | 102813 |
| C_oracle_256 | ct_1036 | 0.8334 | 463 | 91092 | 136284 |
| C_oracle_256 | ct_1046 | 0.8107 | 544 | 164321 | 199594 |
| C_oracle_256 | ct_1056 | 0.6543 | 1120 | 1227235 | 308286 |
| C_oracle_256 | ct_1079 | 0.6795 | 613 | 147870 | 64032 |
| C_oracle_256 | ct_1125 | 0.6295 | 298 | 112432 | 47647 |
| D_predicted_256 | ct_1001 | 0.7533 | 679 | 187717 | 192598 |
| D_predicted_256 | ct_1027 | 0.4861 | 1250 | 303627 | 417263 |
| D_predicted_256 | ct_1028 | 0.6122 | 346 | 82161 | 208232 |
| D_predicted_256 | ct_1033 | 0.6237 | 686 | 140270 | 150735 |
| D_predicted_256 | ct_1036 | 0.7235 | 540 | 93664 | 195449 |
| D_predicted_256 | ct_1046 | 0.7584 | 513 | 167002 | 241753 |
| D_predicted_256 | ct_1056 | 0.6311 | 1446 | 1428036 | 595752 |
| D_predicted_256 | ct_1079 | 0.6340 | 964 | 405405 | 85322 |
| B_full_384 | ct_1001 | 0.6458 | 1463 | 289006 | 521596 |
| B_full_384 | ct_1027 | 0.4522 | 1977 | 700498 | 494691 |
| B_full_384 | ct_1028 | 0.6268 | 671 | 157117 | 423910 |
| B_full_384 | ct_1036 | 0.6510 | 778 | 160209 | 320035 |
| B_full_384 | ct_1033 | 0.6133 | 990 | 314863 | 278750 |
| C_oracle_256 | ct_1099 | 0.6659 | 436 | 186035 | 105650 |
| C_oracle_256 | ct_1127 | 0.6670 | 478 | 316024 | 217921 |
| D_predicted_256 | ct_1125 | 0.6041 | 303 | 50426 | 40095 |
| D_predicted_256 | ct_1099 | 0.6358 | 664 | 233457 | 121467 |
| D_predicted_256 | ct_1127 | 0.6716 | 612 | 251686 | 297052 |
| B_full_384 | ct_1046 | 0.6161 | 1401 | 343437 | 417861 |
| B_full_384 | ct_1079 | 0.6145 | 1081 | 265049 | 188596 |
| B_full_384 | ct_1056 | 0.5445 | 2839 | 941258 | 678660 |
| B_full_384 | ct_1125 | 0.6559 | 398 | 55991 | 96256 |
| B_full_384 | ct_1099 | 0.5262 | 797 | 470806 | 197789 |
| B_full_384 | ct_1127 | 0.6032 | 935 | 322397 | 655080 |
| D_predicted_384 | ct_1001 | 0.5975 | 1685 | 420328 | 232814 |
| D_predicted_384 | ct_1028 | 0.4918 | 1106 | 89208 | 122603 |
| D_predicted_384 | ct_1033 | 0.5987 | 1520 | 234152 | 133454 |
| D_predicted_384 | ct_1027 | 0.3939 | 2702 | 1199116 | 409201 |
| D_predicted_384 | ct_1036 | 0.5682 | 1157 | 139045 | 177528 |
| D_predicted_384 | ct_1046 | 0.5593 | 1580 | 452132 | 269299 |
| D_predicted_384 | ct_1079 | 0.5143 | 2503 | 807971 | 98596 |
| D_predicted_384 | ct_1099 | 0.5076 | 1383 | 285156 | 128594 |
| D_predicted_384 | ct_1056 | 0.5407 | 3830 | 952919 | 663168 |
| D_predicted_384 | ct_1125 | 0.5571 | 587 | 113000 | 41499 |
| D_predicted_384 | ct_1127 | 0.6126 | 1598 | 293419 | 271304 |
| D_core_256 | ct_1001 | 0.7689 | 712 | 218301 | 333597 |
| D_core_256 | ct_1028 | 0.3502 | 551 | 121430 | 109182 |
| D_core_256 | ct_1033 | 0.6764 | 692 | 127006 | 110096 |
| D_core_256 | ct_1027 | 0.5103 | 1290 | 613943 | 366294 |
| D_core_256 | ct_1036 | 0.7433 | 512 | 180488 | 102412 |
| D_core_256 | ct_1046 | 0.7945 | 885 | 196561 | 243670 |
| D_core_256 | ct_1079 | 0.6222 | 946 | 246066 | 71509 |
| D_core_256 | ct_1056 | 0.6107 | 1891 | 883072 | 481776 |
| D_core_256 | ct_1099 | 0.6298 | 689 | 122408 | 107257 |
| D_core_256 | ct_1125 | 0.6430 | 376 | 67628 | 54907 |
| D_core_256 | ct_1127 | 0.6458 | 741 | 222020 | 418455 |

HD95, precision, recall, крупнейшая компонента и доля ложных компонентов по классам: topology_classes.csv и source scores. Все размеры, bbox и положения компонентов: topology.json и component_locations.csv. HD95 вычисляется точно через EDT/KD в единицах сетки; поверхности у source/ignore границ исключены из дистанционной оценки. При отсутствии одной поверхности HD95=None; Dice и recall отражают пропуск. Малые компоненты не удаляются, GT cleanup не применяется. Ложная компонента не пересекает GT своего класса, включая ошибки между структурами; это отличается от объёма BG → FG. Raw VTP Heart3D сохраняет все поверхности. Упрощение до 30 тысяч граней на класс применяется только для браузера. Основные метрики считаются по неизменённой маске в valid domain с 6-связностью. Raw VTP сохраняет также предсказания в ignored GT regions: экспертная маска не используется для очистки поверхности. Дополнительные данные surface helper с 26-связностью и полным raw domain приведены отдельно; их нельзя смешивать с primary counts.

Marching cubes замыкает структуру у границ маски. Heart3D metadata отмечает artificial_boundary_caps при касании исходной границы скана; обрезание ROI также может создавать искусственные окончания сосудов. Замкнутая поверхность не доказывает анатомическую полноту; её нужно рассматривать вместе с coverage и картами ошибок.

Для производительности используется максимум два независимых metric worker; при старте с доступной RAM < 4 GiB или отсутствии psutil — один. VTK экспорт последователен. KD-tree и EDT реализуют одну симметричную Euclidean HD95: на ct_1127 все семь классов совпали с допуском 1e−8 (HD95_exact_algorithms_7classes.json). KD применяется к разреженным поверхностям; EDT остаётся для crop > 5 миллионов вокселей и > 2 миллионов surface points. Меняется алгоритм вычисления, не определение метрики или маски.

## Интерфейс, ограничения и следующий этап

В существующий CT Lab добавлен раздел Heart Localization & ROI Segmentation: три исходные проекции, predicted/fixed/oracle bbox, связанные XYZ, прозрачность и выбор классов, сравнение crop, native predictions, карты FP/FN и ошибок классов, поверхности Heart3D и история обучения. Oracle на validation выключен по умолчанию; для него требуется явный диагностический переключатель. Автоматический inference не принимает маску. Результаты AppTest и браузерных проверок сохранены рядом с отчётом.

Ограничения: ограниченный бюджет обучения с нуля на RTX 2060 и один основной seed; исторические CHD68 обучались с другим бюджетом. Срезы train отбирались по GT. Внутренняя CV наследует ранее выбранный intensity clip. Coarse bbox может частично обрезать тонкие и удалённые структуры; высокое среднее coverage не гарантирует покрытия каждого случая. HU и физические единицы не подтверждены; внешние клинические данные и sealed test не оценивались. Следующий этап: большее число train-only запусков, несколько seed и отдельная внешняя оценка после фиксации методики. Test не открывался и не использовался для подбора.

## Графики и воспроизведение

- [train_heart_centers](outputs/heart_localization_roi_05/charts/train_heart_centers.html) — Центр масс valid foreground, не bbox center. Цвет — доля GT-bbox от полного FOV. Только48 train; координаты нормализованы по shape, не мм.

- [all_train_boxes_normalized](outputs/heart_localization_roi_05/charts/all_train_boxes_normalized.html) — Все train GT-bbox в общей нормализованной системе координат. Это исследовательская визуализация, не автоматические ROI.

- [train_dimensions](outputs/heart_localization_roi_05/charts/train_dimensions.html) — Исходные объёмы и bbox различаются по форме. Сравниваются количества вокселей вдоль осей, не физические длины.

- [bbox_FOV_fraction](outputs/heart_localization_roi_05/charts/bbox_FOV_fraction.html) — Минимальный axis-aligned bbox всех классов1–7. Внутри него остаётся фон; bbox volume не равен foreground volume.

- [coverage_vs_ROI_volume](outputs/heart_localization_roi_05/charts/coverage_vs_ROI_volume.html) — Coverage — главная метрика. Более крупная ROI может снижать bbox IoU и при этом лучше сохранять разметку; margin выбран только по train OOF.

- [train_OOF_margin](outputs/heart_localization_roi_05/charts/train_OOF_margin.html) — Margin выбран до development оценки. Критерий выбора основан на среднем coverage и минимальном среднем по классам; это не гарантия худшего случая.

- [localization_class_coverage](outputs/heart_localization_roi_05/charts/localization_class_coverage.html) — GT используется после автоматического предсказания для диагностики покрытия. Никакой коррекции validation ROI по этой таблице не выполнялось.

- [IoU_vs_coverage](outputs/heart_localization_roi_05/charts/IoU_vs_coverage.html) — IoU не заменяет coverage: включение контекстного margin меняет IoU даже при полном сохранении foreground.

- [resampling_mask_only_Dice](outputs/heart_localization_roi_05/charts/resampling_mask_only_Dice.html) — Ошибки только от преобразования масок, без нейросети. Oracle здесь диагностический train bbox +20%+8 indices; main C использует выбранный train-only margin.

- [native_gradient_preservation](outputs/heart_localization_roi_05/charts/native_gradient_preservation.html) — Градиенты сравниваются после обратного восстановления на одинаковой исходной области каждой строки; разные FOV содержат разные ткани, поэтому raw gradient levels между ROI/full напрямую не сравниваются.

- [train_only_group_CV](outputs/heart_localization_roi_05/charts/train_only_group_CV.html) — Три holdout groups внутри48train, source-grid Dice. CV budget extended symmetrically6→18 epochs after train-only background-only pilot; main jobs unchanged. Не nested unbiased evaluation: localizer best epoch использует train-fold holdout.

- [native_Dice_methods](outputs/heart_localization_roi_05/charts/native_Dice_methods.html) — Среднее по10reused development cases; error bars — case SD, не voxel confidence interval. Oracle отдельно помечен; CHD68 historical budget differs, causal matched comparison — new B/C/D.

- [Dice_by_case](outputs/heart_localization_roi_05/charts/Dice_by_case.html) — Показывает межслучайный разброс, который нельзя заменить числом срезов или sampled voxels.

- [Dice_by_class](outputs/heart_localization_roi_05/charts/Dice_by_class.html) — Среднее только по случаям, где данный GT класс присутствует; false positives для отсутствующих классов сохранены отдельно.

- [paired_Dice_differences](outputs/heart_localization_roi_05/charts/paired_Dice_differences.html) — Одни и те же10cases. Bootstrap CI вычисляется по cases, не по voxels; reused development и multiple comparisons ограничивают выводы.

- [XY_scale_vs_Dice_change](outputs/heart_localization_roi_05/charts/XY_scale_vs_Dice_change.html) — Описательная связь на десяти случаях. ROI одновременно меняет контекст и ограничивает область предсказания; график не доказывает причинную роль детализации и не используется для выбора параметров.

- [false_components_no_GT_overlap](outputs/heart_localization_roi_05/charts/false_components_no_GT_overlap.html) — 6-connectivity, source valid grid, без удаления малых компонент. False component не пересекает GT своего класса; это не обязательно только фоновый false positive. Меньшее число компонент не доказывает лучшую анатомию.

- [false_component_voxels](outputs/heart_localization_roi_05/charts/false_component_voxels.html) — 6-connectivity, source valid grid, без удаления малых компонент. False component не пересекает GT своего класса; это не обязательно только фоновый false positive. Меньшее число компонент не доказывает лучшую анатомию.

- [largest_component_voxels](outputs/heart_localization_roi_05/charts/largest_component_voxels.html) — 6-connectivity, source valid grid, без удаления малых компонент. False component не пересекает GT своего класса; это не обязательно только фоновый false positive. Меньшее число компонент не доказывает лучшую анатомию.

- [HD95_by_class](outputs/heart_localization_roi_05/charts/HD95_by_class.html) — Среднее по случаям с двумя непустыми поверхностями. Пропуски поверхности не превращаются в нулевой HD95: значения None сохранены в таблицах; оценивать вместе с recall/Dice.

- [training_loss](outputs/heart_localization_roi_05/charts/training_loss.html) — Это train augmented/resampled grid. Native development Dice считается отдельно после полного восстановления; best.pt у main — fixed last epoch, не outer-selected.

- [training_train_augmented_grid_foreground_Dice](outputs/heart_localization_roi_05/charts/training_train_augmented_grid_foreground_Dice.html) — Это train augmented/resampled grid. Native development Dice считается отдельно после полного восстановления; best.pt у main — fixed last epoch, не outer-selected.

![Resolution ct_1001](outputs/heart_localization_roi_05/resolution_ct_1001.png)

![Resolution ct_1002](outputs/heart_localization_roi_05/resolution_ct_1002.png)

![Resolution ct_1004](outputs/heart_localization_roi_05/resolution_ct_1004.png)

![Native comparison ct_1001](outputs/heart_localization_roi_05/native_comparison_ct_1001.png)

![Native comparison ct_1027](outputs/heart_localization_roi_05/native_comparison_ct_1027.png)

Команды из repository (PowerShell). Для независимого воспроизведения нужна отдельная копия с новым stage05 outputs; прежние результаты не удалять. Завершённые receipts пропускаются, last.pt продолжает обучение с границы эпохи. Перед browser запустить roi_launch в отдельном терминале. Для уже выделенного RTX 4090: ROI05_DATA_ROOT и ROI05_WEIGHTS_ROOT, затем bash scripts/run_roi05_runpod.sh; платные ресурсы этот скрипт не создаёт.
```powershell
.venv/Scripts/python.exe -m ct_lab.roi_analysis
.venv/Scripts/python.exe -m ct_lab.roi_smoke
.venv/Scripts/python.exe -m ct_lab.roi_localizer
.venv/Scripts/python.exe -m ct_lab.roi_train
.venv/Scripts/python.exe -m ct_lab.roi_baselines
.venv/Scripts/python.exe -m ct_lab.roi_topology
.venv/Scripts/python.exe -m ct_lab.roi_analysis
.venv/Scripts/python.exe -m ct_lab.roi_verify_ui
node ct_lab/roi_browser.cjs
.venv/Scripts/python.exe -m pytest -p ct_lab.roi_pytest tests/test_roi_lab.py tests/test_multiclass_lab.py tests/test_learnable_features.py tests/test_spatial_features.py tests/test_filtering_features.py tests/test_ct_lab.py tests/test_cardiac_inference.py -q
.venv/Scripts/python.exe -m ct_lab.roi_report
.venv/Scripts/python.exe -m ct_lab.roi_launch
```

Целостность: проверены 116 исходных файлов; 1689 прежних артефактов не изменены побайтно. SHA256 весов v1/v2, split и preprocessing совпали. Проверены 16 новых checkpoints и 88 native NIfTI/array масок. Автотесты: 69; проверки UI: 24 — все успешно. Браузерные результаты сохранены. Production training regression test не запускался.

| Library | Version |
|---|---|
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| torch | 2.14.1+cu130 |
| nibabel | 5.4.2 |
| plotly | 6.3.1 |
| streamlit | 1.50.0 |
| scikit-image | 0.26.0 |
| pyvista | 0.49.0 |
| filelock | 4.2.4 |
| python | 3.13.3 |
| GPU | NVIDIA GeForce RTX 2060 |
| CUDA | 13.0 |
