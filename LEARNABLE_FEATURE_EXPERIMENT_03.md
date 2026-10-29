# LEARNABLE_FEATURE_EXPERIMENT_03 — обучаемое представление признаков


Дата: 2026-10-28. Продолжение разработки ImageCHD68; validation ранее использовалась и теперь служит также early stopping. Независимой финальной оценки нет. Sealed test не открывался.


## 1. Результат и контролируемая абляция

Гипотеза подтверждена для диагностической бинарной классификации RV/PA на этой validation: нелинейное преобразование C/G улучшает case-macro F1 относительно k-NN на тех же данных. Для LV/MYO улучшение зависит от набора признаков. SupCon и Spatial не дают универсального дополнительного выигрыша. Это не сегментация и не оценка Dice.

Сначала A–G рассчитаны на точных IDs этапа 02: пять train, десять validation. Затем отдельно повторены на 48 train с семью Z-позициями. Разность между двумя протоколами нельзя приписать одной архитектуре: изменились train cohort и sampling.

| Cohort | Пара | Набор | F1 case mean | SD | Balanced accuracy |
|---|---|---|---|---|---|
| legacy5 | LV_MYO | A Intensity | 0.9054 | 0.1378 | 0.9540 |
| legacy5 | LV_MYO | B 2D | 0.8916 | 0.1371 | 0.9364 |
| legacy5 | LV_MYO | C 3D-core | 0.8939 | 0.1378 | 0.9395 |
| legacy5 | LV_MYO | D core+G0.5 | 0.8938 | 0.1378 | 0.9396 |
| legacy5 | LV_MYO | E core+G1 | 0.8932 | 0.1377 | 0.9389 |
| legacy5 | LV_MYO | F core+G2 | 0.8931 | 0.1380 | 0.9388 |
| legacy5 | LV_MYO | G core+all | 0.8940 | 0.1378 | 0.9401 |
| legacy5 | RV_PA | A Intensity | 0.4590 | 0.0624 | 0.5183 |
| legacy5 | RV_PA | B 2D | 0.6003 | 0.1146 | 0.6380 |
| legacy5 | RV_PA | C 3D-core | 0.6164 | 0.1104 | 0.6542 |
| legacy5 | RV_PA | D core+G0.5 | 0.6152 | 0.1106 | 0.6531 |
| legacy5 | RV_PA | E core+G1 | 0.6145 | 0.1105 | 0.6523 |
| legacy5 | RV_PA | F core+G2 | 0.6261 | 0.1151 | 0.6663 |
| legacy5 | RV_PA | G core+all | 0.6269 | 0.1153 | 0.6661 |
| full48 | LV_MYO | A Intensity | 0.9466 | 0.0202 | 0.9490 |
| full48 | LV_MYO | B 2D | 0.9489 | 0.0184 | 0.9512 |
| full48 | LV_MYO | C 3D-core | 0.9513 | 0.0167 | 0.9528 |
| full48 | LV_MYO | D core+G0.5 | 0.9511 | 0.0171 | 0.9529 |
| full48 | LV_MYO | E core+G1 | 0.9506 | 0.0170 | 0.9526 |
| full48 | LV_MYO | F core+G2 | 0.9522 | 0.0173 | 0.9538 |
| full48 | LV_MYO | G core+all | 0.9540 | 0.0181 | 0.9560 |
| full48 | RV_PA | A Intensity | 0.5038 | 0.0485 | 0.5246 |
| full48 | RV_PA | B 2D | 0.6799 | 0.0842 | 0.6961 |
| full48 | RV_PA | C 3D-core | 0.7173 | 0.0793 | 0.7340 |
| full48 | RV_PA | D core+G0.5 | 0.7124 | 0.0821 | 0.7288 |
| full48 | RV_PA | E core+G1 | 0.7151 | 0.0800 | 0.7321 |
| full48 | RV_PA | F core+G2 | 0.7389 | 0.0730 | 0.7565 |
| full48 | RV_PA | G core+all | 0.7390 | 0.0745 | 0.7559 |

A — intensity; B — intensity + XY texture; C — intensity + 3D Mean/Std/Gradient/LoG; D/E/F — C + Gaussian σ=0.5/1/2; G — C + все Gaussian. Mean/Std: 5×5×5 (XY: 5×5), центральный gradient, LoG σ=1; Gaussian truncate=4, reflect, float32, bounded patches с halo. σ в единицах сетки, физические мм не подтверждены. Исходная шкала не подтверждена как HU.


### Вклад Gaussian относительно C, парные различия по 10 случаям

| Cohort | Пара | Набор | ΔF1 mean | min | max | cases improved |
|---|---|---|---|---|---|---|
| legacy5 | LV_MYO | D core+G0.5 | -0.0000 | -0.0040 | 0.0048 | 2 |
| legacy5 | LV_MYO | E core+G1 | -0.0007 | -0.0031 | 0.0028 | 2 |
| legacy5 | LV_MYO | F core+G2 | -0.0008 | -0.0073 | 0.0068 | 3 |
| legacy5 | LV_MYO | G core+all | 0.0002 | -0.0025 | 0.0062 | 5 |
| legacy5 | RV_PA | D core+G0.5 | -0.0013 | -0.0035 | 0.0020 | 1 |
| legacy5 | RV_PA | E core+G1 | -0.0019 | -0.0075 | 0.0017 | 2 |
| legacy5 | RV_PA | F core+G2 | 0.0097 | -0.0094 | 0.0238 | 8 |
| legacy5 | RV_PA | G core+all | 0.0105 | -0.0105 | 0.0237 | 8 |
| full48 | LV_MYO | D core+G0.5 | -0.0002 | -0.0061 | 0.0071 | 4 |
| full48 | LV_MYO | E core+G1 | -0.0007 | -0.0061 | 0.0036 | 3 |
| full48 | LV_MYO | F core+G2 | 0.0009 | -0.0059 | 0.0107 | 6 |
| full48 | LV_MYO | G core+all | 0.0027 | -0.0092 | 0.0142 | 5 |
| full48 | RV_PA | D core+G0.5 | -0.0049 | -0.0184 | 0.0039 | 2 |
| full48 | RV_PA | E core+G1 | -0.0022 | -0.0125 | 0.0127 | 3 |
| full48 | RV_PA | F core+G2 | 0.0216 | 0.0002 | 0.0482 | 10 |
| full48 | RV_PA | G core+all | 0.0218 | 0.0011 | 0.0444 | 10 |

На legacy RV/PA C=0.6164, G=0.6269: выигрыш G нельзя целиком назвать эффектом 3D-контекста. Gaussian 2 даёт основной небольшой дополнительный вклад; отдельные σ=0.5/1 не улучшают результат. Конфигурации после просмотра validation не менялись.


## 2. Данные и отсутствие утечки

Frozen split: 48 train release-cases / 10 validation / 10 sealed test. Case/group пересечения исключены. Sampling: Z fractions [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], максимум 64/class/Z, максимум 896 anchors/pair/case; редкие/отсутствующие классы не дополняются искусственными точками. Число точек одинаково во всех конфигурациях внутри одного протокола; состав см. sample_counts.csv.

Source XYZ и C-order voxel IDs сохраняются. Метки используются только для sampling, targets и допустимости train-соседей. Вход encoder — исключительно CT-derived признаки. Scaler mean/population SD fit только на 48 train, отдельно C/G, одинаковый для всех losses и размеров embedding. Validation не участвует в градиентах, scaler, PCA/UMAP fit или обучающих соседствах. Её F1 выбирает эпоху, поэтому модель всё же адаптируется к validation через early stopping.

| Пара | Partition | Sampled voxels |
|---|---|---|
| LV_MYO | train | 23535 |
| LV_MYO | validation | 4736 |
| RV_PA | train | 28164 |
| RV_PA | validation | 5737 |


## 3. Архитектура и обучение

Linear(input,64) → ReLU → Linear(64,32) → ReLU → Linear(32,e) → Linear(e,2). e=8/16/32, C=5 и G=8 входов. Поддерживается head на 8 классов, но обучены только две бинарные пары. Каждая из 36 моделей имеет собственные checkpoint, scaler, case IDs, seed, history, probabilities и embeddings. Никакие CHD68 v1/v2 веса не использовались и не изменялись.

AdamW lr=0.001, weight decay=0.0001, batch=512, max epochs=30, patience=5, min_delta=0.0001; CPU threads=1. Общий seed=20261028; per-model seed=seed+1000×pair_index+100×feature_index+embedding_dim. Одинаковая инициализация и последовательность batches для трёх losses при одной архитектуре. Class-balanced sampling без повторов внутри batch; веса обратно пропорциональны count(case,class), case balance в ожидании, не точная квота. Суммарное время train/evaluation циклов 36 моделей: 237.3 s (без extraction/UMAP).

Параметров на модель: 2746–3778. Train F1 — resubstitution; k-NN train дополнительно оптимистичен из-за exact self-neighbor. Все параметры и C/G были зафиксированы в protocol.json до первых validation результатов. Основное отображение e16 выбрано заранее; все кандидаты приведены ниже.


## 4. Loss и кривые

CE; CE+0.1 SupCon; CE+0.1 SupCon+0.1 Spatial. SupCon: L2-normalized z, temperature=0.1, self исключён, все другие same-class anchors — positives, other-class — negatives. Anchors без positives пропускаются. Spatial: MSE между normalized embeddings anchor и одного реального distinct 6-connected train-соседа того же GT класса, границы класса не пересекаются; при отсутствии допустимого соседа term пропускается. GT не передаётся в model.forward. Validation соседства вообще не создаются.

[F1 / epoch](outputs/learnable_feature_03/history_f1.html), [Loss / epoch](outputs/learnable_feature_03/history_loss.html). Истории всех отдельных components сохранены в models/**/history.json. Значения total loss между CE и комбинированными функциями напрямую не сравнимы; SupCon имеет ненулевой уровень при множестве positives, даже когда классы разделены.


## 5. Сравнение всех кандидатов

| Пара | Inputs | Метод | e | Train F1 | Val F1 | Val SD | Val BA | Best epoch |
|---|---|---|---|---|---|---|---|---|
| LV_MYO | C 3D-core | k-NN | 5 | 1.0000 | 0.9513 | 0.0167 | 0.9528 |  |
| LV_MYO | G core+all | k-NN | 8 | 1.0000 | 0.9540 | 0.0181 | 0.9560 |  |
| RV_PA | C 3D-core | k-NN | 5 | 1.0000 | 0.7173 | 0.0793 | 0.7340 |  |
| RV_PA | G core+all | k-NN | 8 | 1.0000 | 0.7390 | 0.0745 | 0.7559 |  |
| LV_MYO | C 3D-core | CE | 8 | 0.9077 | 0.9547 | 0.0213 | 0.9568 | 1 |
| LV_MYO | C 3D-core | CE | 16 | 0.9063 | 0.9542 | 0.0220 | 0.9572 | 1 |
| LV_MYO | C 3D-core | CE | 32 | 0.9149 | 0.9541 | 0.0292 | 0.9590 | 9 |
| LV_MYO | C 3D-core | CE+SupCon | 8 | 0.9097 | 0.9532 | 0.0243 | 0.9565 | 4 |
| LV_MYO | C 3D-core | CE+SupCon | 16 | 0.9129 | 0.9542 | 0.0257 | 0.9579 | 7 |
| LV_MYO | C 3D-core | CE+SupCon | 32 | 0.9136 | 0.9537 | 0.0241 | 0.9571 | 5 |
| LV_MYO | C 3D-core | CE+SupCon+Spatial | 8 | 0.9094 | 0.9534 | 0.0241 | 0.9567 | 4 |
| LV_MYO | C 3D-core | CE+SupCon+Spatial | 16 | 0.9128 | 0.9541 | 0.0256 | 0.9579 | 7 |
| LV_MYO | C 3D-core | CE+SupCon+Spatial | 32 | 0.9134 | 0.9537 | 0.0242 | 0.9571 | 5 |
| LV_MYO | G core+all | CE | 8 | 0.9069 | 0.9556 | 0.0194 | 0.9577 | 1 |
| LV_MYO | G core+all | CE | 16 | 0.9099 | 0.9520 | 0.0242 | 0.9554 | 1 |
| LV_MYO | G core+all | CE | 32 | 0.9085 | 0.9549 | 0.0227 | 0.9576 | 1 |
| LV_MYO | G core+all | CE+SupCon | 8 | 0.9030 | 0.9565 | 0.0169 | 0.9572 | 1 |
| LV_MYO | G core+all | CE+SupCon | 16 | 0.9114 | 0.9516 | 0.0247 | 0.9550 | 1 |
| LV_MYO | G core+all | CE+SupCon | 32 | 0.9083 | 0.9551 | 0.0222 | 0.9575 | 1 |
| LV_MYO | G core+all | CE+SupCon+Spatial | 8 | 0.9033 | 0.9565 | 0.0169 | 0.9572 | 1 |
| LV_MYO | G core+all | CE+SupCon+Spatial | 16 | 0.9114 | 0.9520 | 0.0251 | 0.9554 | 1 |
| LV_MYO | G core+all | CE+SupCon+Spatial | 32 | 0.9082 | 0.9550 | 0.0228 | 0.9575 | 1 |
| RV_PA | C 3D-core | CE | 8 | 0.7362 | 0.7783 | 0.0688 | 0.8055 | 14 |
| RV_PA | C 3D-core | CE | 16 | 0.7394 | 0.7785 | 0.0694 | 0.8038 | 17 |
| RV_PA | C 3D-core | CE | 32 | 0.7335 | 0.7762 | 0.0701 | 0.8017 | 7 |
| RV_PA | C 3D-core | CE+SupCon | 8 | 0.7331 | 0.7760 | 0.0682 | 0.8020 | 10 |
| RV_PA | C 3D-core | CE+SupCon | 16 | 0.7381 | 0.7776 | 0.0673 | 0.8064 | 18 |
| RV_PA | C 3D-core | CE+SupCon | 32 | 0.7334 | 0.7757 | 0.0686 | 0.8015 | 8 |
| RV_PA | C 3D-core | CE+SupCon+Spatial | 8 | 0.7335 | 0.7765 | 0.0658 | 0.8027 | 10 |
| RV_PA | C 3D-core | CE+SupCon+Spatial | 16 | 0.7417 | 0.7794 | 0.0634 | 0.8029 | 17 |
| RV_PA | C 3D-core | CE+SupCon+Spatial | 32 | 0.7362 | 0.7752 | 0.0664 | 0.7975 | 7 |
| RV_PA | G core+all | CE | 8 | 0.7848 | 0.8207 | 0.0604 | 0.8438 | 17 |
| RV_PA | G core+all | CE | 16 | 0.7892 | 0.8203 | 0.0563 | 0.8417 | 15 |
| RV_PA | G core+all | CE | 32 | 0.7848 | 0.8191 | 0.0570 | 0.8427 | 12 |
| RV_PA | G core+all | CE+SupCon | 8 | 0.7874 | 0.8190 | 0.0522 | 0.8372 | 26 |
| RV_PA | G core+all | CE+SupCon | 16 | 0.7877 | 0.8245 | 0.0527 | 0.8455 | 20 |
| RV_PA | G core+all | CE+SupCon | 32 | 0.7898 | 0.8209 | 0.0518 | 0.8450 | 19 |
| RV_PA | G core+all | CE+SupCon+Spatial | 8 | 0.7826 | 0.8188 | 0.0622 | 0.8431 | 14 |
| RV_PA | G core+all | CE+SupCon+Spatial | 16 | 0.7849 | 0.8210 | 0.0568 | 0.8427 | 13 |
| RV_PA | G core+all | CE+SupCon+Spatial | 32 | 0.7843 | 0.8206 | 0.0566 | 0.8463 | 18 |

Основная метрика — mean двухклассового macro F1 по validation cases, с фиксированными двумя labels и zero_division=0 (как в предыдущем этапе). BA усредняет recall присутствующих targets. Pooled voxel metrics хранятся отдельно и не заменяют case averaging. Class-wise precision/recall/F1 и CM — results.json, metrics_by_case.csv. Воксели не являются независимыми пациентами.

Class-wise ablation метрики: ablation_classwise.csv; confusion matrices всех A–G (pooled и per-case): ablation_confusion_matrices.json. Ниже pooled CM и class-wise F1 для основного e16; порядок строк/столбцов соответствует названной паре. Pooled F1 здесь приведён только как дополнительная диагностика.

| Пара | Inputs | Метод e16 | F1 class 1 | F1 class 2 | Pooled CM |
|---|---|---|---|---|---|
| LV_MYO | C | k-NN | 0.9448 | 0.9596 | [[1891, 93], [128, 2624]] |
| LV_MYO | G | k-NN | 0.9482 | 0.9619 | [[1904, 80], [128, 2624]] |
| RV_PA | C | k-NN | 0.6840 | 0.7646 | [[1675, 757], [791, 2514]] |
| RV_PA | G | k-NN | 0.7138 | 0.7793 | [[1783, 649], [781, 2524]] |
| LV_MYO | C | CE | 0.9483 | 0.9615 | [[1917, 67], [142, 2610]] |
| LV_MYO | C | CE+SupCon | 0.9497 | 0.9623 | [[1927, 57], [147, 2605]] |
| LV_MYO | C | CE+SupCon+Spatial | 0.9495 | 0.9621 | [[1927, 57], [148, 2604]] |
| LV_MYO | G | CE | 0.9468 | 0.9601 | [[1922, 62], [154, 2598]] |
| LV_MYO | G | CE+SupCon | 0.9465 | 0.9599 | [[1921, 63], [154, 2598]] |
| LV_MYO | G | CE+SupCon+Spatial | 0.9470 | 0.9603 | [[1922, 62], [153, 2599]] |
| RV_PA | C | CE | 0.7814 | 0.7947 | [[2171, 261], [954, 2351]] |
| RV_PA | C | CE+SupCon | 0.7854 | 0.7903 | [[2227, 205], [1012, 2293]] |
| RV_PA | C | CE+SupCon+Spatial | 0.7789 | 0.7999 | [[2123, 309], [896, 2409]] |
| RV_PA | G | CE | 0.8179 | 0.8422 | [[2178, 254], [716, 2589]] |
| RV_PA | G | CE+SupCon | 0.8218 | 0.8472 | [[2176, 256], [688, 2617]] |
| RV_PA | G | CE+SupCon+Spatial | 0.8199 | 0.8422 | [[2196, 236], [729, 2576]] |


### Основное e16: парные ΔF1 относительно k-NN

[График по каждому случаю](outputs/learnable_feature_03/paired_case_f1.html).

- LV_MYO C CE: mean +0.0029; range [-0.0153, +0.0231], improved 6/10.

- LV_MYO C CE+SupCon: mean +0.0029; range [-0.0185, +0.0336], improved 6/10.

- LV_MYO C CE+SupCon+Spatial: mean +0.0028; range [-0.0185, +0.0336], improved 5/10.

- LV_MYO G CE: mean -0.0020; range [-0.0156, +0.0194], improved 4/10.

- LV_MYO G CE+SupCon: mean -0.0024; range [-0.0155, +0.0177], improved 4/10.

- LV_MYO G CE+SupCon+Spatial: mean -0.0020; range [-0.0186, +0.0177], improved 5/10.

- RV_PA C CE: mean +0.0612; range [-0.0257, +0.1901], improved 9/10.

- RV_PA C CE+SupCon: mean +0.0603; range [-0.0233, +0.1953], improved 9/10.

- RV_PA C CE+SupCon+Spatial: mean +0.0621; range [-0.0126, +0.1971], improved 9/10.

- RV_PA G CE: mean +0.0812; range [+0.0263, +0.2416], improved 10/10.

- RV_PA G CE+SupCon: mean +0.0854; range [+0.0263, +0.2397], improved 10/10.

- RV_PA G CE+SupCon+Spatial: mean +0.0820; range [+0.0148, +0.2342], improved 10/10.


## 6. Изменение embeddings и проекции

[Все интерактивные графики](outputs/learnable_feature_03/charts.html). PCA/UMAP до/после используют одни и те же 32/class/case IDs, отдельно показаны train и validation. Scaler/reducer fit только train; validation только transform. Метки не передаются UMAP. Bases независимы при разных размерностях representations, поэтому положения/масштабы на рисунках напрямую не сравниваются. Для количественного сравнения ниже использовано исходное raw embedding space, не 2D проекция.

| Пара | Inputs | Method e16 | Silhouette | Fisher | Intra dist | Inter dist | Inter/Intra |
|---|---|---|---|---|---|---|---|
| LV_MYO | C | k-NN | 0.3331 | 0.8665 | 2.4601 | 3.7115 | 1.5087 |
| LV_MYO | G | k-NN | 0.4138 | 1.5764 | 2.8756 | 5.0514 | 1.7567 |
| RV_PA | C | k-NN | 0.1376 | 0.3094 | 2.7910 | 3.2927 | 1.1797 |
| RV_PA | G | k-NN | 0.1089 | 0.2218 | 3.7048 | 4.2077 | 1.1358 |
| LV_MYO | C | CE | 0.6302 | 5.0518 | 1.8294 | 5.4328 | 2.9697 |
| LV_MYO | C | CE+SupCon | 0.4919 | 1.9586 | 3.5880 | 7.3045 | 2.0358 |
| LV_MYO | C | CE+SupCon+Spatial | 0.4980 | 2.0419 | 3.4992 | 7.2113 | 2.0608 |
| LV_MYO | G | CE | 0.6112 | 4.3629 | 2.2651 | 6.3356 | 2.7970 |
| LV_MYO | G | CE+SupCon | 0.5417 | 2.8476 | 1.9089 | 4.4788 | 2.3463 |
| LV_MYO | G | CE+SupCon+Spatial | 0.5412 | 2.8407 | 1.8999 | 4.4529 | 2.3438 |
| RV_PA | C | CE | 0.2859 | 0.9415 | 2.2108 | 3.4779 | 1.5732 |
| RV_PA | C | CE+SupCon | 0.2783 | 0.8361 | 3.7736 | 5.8282 | 1.5445 |
| RV_PA | C | CE+SupCon+Spatial | 0.2805 | 0.8568 | 3.8520 | 5.9767 | 1.5516 |
| RV_PA | G | CE | 0.3309 | 1.2357 | 1.7050 | 2.8926 | 1.6965 |
| RV_PA | G | CE+SupCon | 0.2942 | 0.8955 | 4.0609 | 6.4249 | 1.5821 |
| RV_PA | G | CE+SupCon+Spatial | 0.2841 | 0.8572 | 3.9041 | 6.0649 | 1.5535 |

Геометрия оценивается на фиксированной стратифицированной подвыборке ≤256/class. Абсолютные расстояния зависят от масштаба embedding; отношения/Silhouette/Fisher полезнее, но тоже зависят от sampling. Ни один из 36 embeddings не collapsed в общую точку: проверены variance raw и normalized train vectors; minimum mean unit SD=0.0336299. Это исключает полный collapse, но не доказывает высокую intrinsic dimension или отсутствие частичной избыточности.


## 7. Contrastive loss

При фиксированном e16 RV/PA G: CE=0.8203, CE+SupCon=0.8245, то есть дополнительный эффект лишь +0.0042; основной выигрыш относительно k-NN=0.7390 уже даёт CE. По другим inputs/парам дополнительный эффект меняется. Один seed на архитектуру не позволяет утверждать статистически устойчивое превосходство SupCon. Параметры не подбирались по красивым UMAP.

Для RV/PA G validation Silhouette в raw spaces: k-NN inputs=0.109, CE embedding=0.331, SupCon=0.294, Spatial=0.284; Fisher соответственно 0.222/1.236/0.895/0.857. Небольшой рост classifier F1 у SupCon не сопровождается улучшением этих геометрических метрик относительно CE. Классификационная голова и геометрия пространства оценивают разные свойства.


## 8. Spatial consistency

Использовано 51693 реальных допустимых train neighbor pairs (один на anchor, если существует). e16 RV/PA C: Spatial=0.7794 против CE=0.7785 и SupCon=0.7776; G: Spatial=0.8210 против SupCon=0.8245. Пространственная регуляризация даёт смешанный результат; устойчивое преимущество не установлено. Она влияет на обучение embeddings, а не выполняет postprocessing/сглаживание validation маски.


## 9. Устойчивость

Stress tests заранее фиксированы: независимый noise σ=0.05 training SD по всем каналам; intensity shift +0.1 SD; intensity×1.1 относительно train mean; выключение 3D LoG заменой train mean. Это небольшие стандартизованные сдвиги и один более сильный channel ablation; они позволяют одинаковую механическую нагрузку C/G и методам, но не калиброваны по реальным сканерам. Per-ID noise одинаков между моделями, не зависит от порядка строк. Возмущения независимых коррелированных каналов могут лежать вне распределения реальных CT-derived features.

| Пара | Inputs | Method e16 | Perturbation | Clean F1 | Perturbed F1 | Mean |Δp| | Raw emb RMSE | Unit emb RMSE |
|---|---|---|---|---|---|---|---|---|
| LV_MYO | C 3D-core | k-NN | Gaussian .05 SD | 0.9513 | 0.9509 | 0.0197 | 0.0501 | 0.0320 |
| LV_MYO | C 3D-core | k-NN | Intensity +.1 SD | 0.9513 | 0.9489 | 0.0173 | 0.0447 | 0.0290 |
| LV_MYO | C 3D-core | k-NN | Intensity scale 1.1 | 0.9513 | 0.9514 | 0.0080 | 0.0512 | 0.0170 |
| LV_MYO | C 3D-core | k-NN | Drop 3D LoG | 0.9513 | 0.9526 | 0.0457 | 0.4699 | 0.2190 |
| LV_MYO | C 3D-core | CE | Gaussian .05 SD | 0.9542 | 0.9551 | 0.0072 | 0.0234 | 0.0113 |
| LV_MYO | C 3D-core | CE | Intensity +.1 SD | 0.9542 | 0.9512 | 0.0094 | 0.0248 | 0.0120 |
| LV_MYO | C 3D-core | CE | Intensity scale 1.1 | 0.9542 | 0.9547 | 0.0042 | 0.0259 | 0.0030 |
| LV_MYO | C 3D-core | CE | Drop 3D LoG | 0.9542 | 0.9537 | 0.0219 | 0.1089 | 0.0425 |
| LV_MYO | C 3D-core | CE+SupCon | Gaussian .05 SD | 0.9542 | 0.9536 | 0.0072 | 0.0461 | 0.0037 |
| LV_MYO | C 3D-core | CE+SupCon | Intensity +.1 SD | 0.9542 | 0.9515 | 0.0111 | 0.0512 | 0.0044 |
| LV_MYO | C 3D-core | CE+SupCon | Intensity scale 1.1 | 0.9542 | 0.9545 | 0.0033 | 0.0654 | 0.0017 |
| LV_MYO | C 3D-core | CE+SupCon | Drop 3D LoG | 0.9542 | 0.9520 | 0.0100 | 0.1951 | 0.0085 |
| LV_MYO | C 3D-core | CE+SupCon+Spatial | Gaussian .05 SD | 0.9541 | 0.9538 | 0.0072 | 0.0456 | 0.0036 |
| LV_MYO | C 3D-core | CE+SupCon+Spatial | Intensity +.1 SD | 0.9541 | 0.9509 | 0.0111 | 0.0500 | 0.0043 |
| LV_MYO | C 3D-core | CE+SupCon+Spatial | Intensity scale 1.1 | 0.9541 | 0.9545 | 0.0033 | 0.0632 | 0.0017 |
| LV_MYO | C 3D-core | CE+SupCon+Spatial | Drop 3D LoG | 0.9541 | 0.9522 | 0.0100 | 0.1972 | 0.0081 |
| LV_MYO | G core+all | k-NN | Gaussian .05 SD | 0.9540 | 0.9521 | 0.0167 | 0.0500 | 0.0271 |
| LV_MYO | G core+all | k-NN | Intensity +.1 SD | 0.9540 | 0.9518 | 0.0104 | 0.0354 | 0.0198 |
| LV_MYO | G core+all | k-NN | Intensity scale 1.1 | 0.9540 | 0.9545 | 0.0049 | 0.0405 | 0.0111 |
| LV_MYO | G core+all | k-NN | Drop 3D LoG | 0.9540 | 0.9510 | 0.0435 | 0.3715 | 0.1474 |
| LV_MYO | G core+all | CE | Gaussian .05 SD | 0.9520 | 0.9520 | 0.0051 | 0.0198 | 0.0081 |
| LV_MYO | G core+all | CE | Intensity +.1 SD | 0.9520 | 0.9490 | 0.0080 | 0.0224 | 0.0105 |
| LV_MYO | G core+all | CE | Intensity scale 1.1 | 0.9520 | 0.9515 | 0.0026 | 0.0210 | 0.0017 |
| LV_MYO | G core+all | CE | Drop 3D LoG | 0.9520 | 0.9503 | 0.0083 | 0.0638 | 0.0235 |
| LV_MYO | G core+all | CE+SupCon | Gaussian .05 SD | 0.9516 | 0.9517 | 0.0053 | 0.0160 | 0.0047 |
| LV_MYO | G core+all | CE+SupCon | Intensity +.1 SD | 0.9516 | 0.9495 | 0.0083 | 0.0167 | 0.0057 |
| LV_MYO | G core+all | CE+SupCon | Intensity scale 1.1 | 0.9516 | 0.9517 | 0.0037 | 0.0182 | 0.0017 |
| LV_MYO | G core+all | CE+SupCon | Drop 3D LoG | 0.9516 | 0.9505 | 0.0076 | 0.0577 | 0.0153 |
| LV_MYO | G core+all | CE+SupCon+Spatial | Gaussian .05 SD | 0.9520 | 0.9514 | 0.0053 | 0.0160 | 0.0047 |
| LV_MYO | G core+all | CE+SupCon+Spatial | Intensity +.1 SD | 0.9520 | 0.9496 | 0.0083 | 0.0166 | 0.0057 |
| LV_MYO | G core+all | CE+SupCon+Spatial | Intensity scale 1.1 | 0.9520 | 0.9518 | 0.0037 | 0.0181 | 0.0017 |
| LV_MYO | G core+all | CE+SupCon+Spatial | Drop 3D LoG | 0.9520 | 0.9505 | 0.0073 | 0.0563 | 0.0150 |
| RV_PA | C 3D-core | k-NN | Gaussian .05 SD | 0.7173 | 0.7113 | 0.0635 | 0.0503 | 0.0340 |
| RV_PA | C 3D-core | k-NN | Intensity +.1 SD | 0.7173 | 0.7116 | 0.0575 | 0.0447 | 0.0306 |
| RV_PA | C 3D-core | k-NN | Intensity scale 1.1 | 0.7173 | 0.7068 | 0.0480 | 0.0611 | 0.0167 |
| RV_PA | C 3D-core | k-NN | Drop 3D LoG | 0.7173 | 0.6433 | 0.1413 | 0.4406 | 0.1959 |
| RV_PA | C 3D-core | CE | Gaussian .05 SD | 0.7785 | 0.7692 | 0.0324 | 0.0640 | 0.0479 |
| RV_PA | C 3D-core | CE | Intensity +.1 SD | 0.7785 | 0.7653 | 0.0408 | 0.0705 | 0.0499 |
| RV_PA | C 3D-core | CE | Intensity scale 1.1 | 0.7785 | 0.7651 | 0.0381 | 0.1073 | 0.0480 |
| RV_PA | C 3D-core | CE | Drop 3D LoG | 0.7785 | 0.5416 | 0.2402 | 0.5000 | 0.2057 |
| RV_PA | C 3D-core | CE+SupCon | Gaussian .05 SD | 0.7776 | 0.7634 | 0.0356 | 0.0664 | 0.0120 |
| RV_PA | C 3D-core | CE+SupCon | Intensity +.1 SD | 0.7776 | 0.7560 | 0.0488 | 0.0703 | 0.0137 |
| RV_PA | C 3D-core | CE+SupCon | Intensity scale 1.1 | 0.7776 | 0.7519 | 0.0426 | 0.1043 | 0.0121 |
| RV_PA | C 3D-core | CE+SupCon | Drop 3D LoG | 0.7776 | 0.5089 | 0.2590 | 0.7781 | 0.0580 |
| RV_PA | C 3D-core | CE+SupCon+Spatial | Gaussian .05 SD | 0.7794 | 0.7665 | 0.0332 | 0.0680 | 0.0112 |
| RV_PA | C 3D-core | CE+SupCon+Spatial | Intensity +.1 SD | 0.7794 | 0.7687 | 0.0423 | 0.0716 | 0.0121 |
| RV_PA | C 3D-core | CE+SupCon+Spatial | Intensity scale 1.1 | 0.7794 | 0.7647 | 0.0371 | 0.1110 | 0.0103 |
| RV_PA | C 3D-core | CE+SupCon+Spatial | Drop 3D LoG | 0.7794 | 0.5134 | 0.2375 | 0.7708 | 0.0531 |
| RV_PA | G core+all | k-NN | Gaussian .05 SD | 0.7390 | 0.7369 | 0.0582 | 0.0499 | 0.0291 |
| RV_PA | G core+all | k-NN | Intensity +.1 SD | 0.7390 | 0.7356 | 0.0411 | 0.0354 | 0.0212 |
| RV_PA | G core+all | k-NN | Intensity scale 1.1 | 0.7390 | 0.7386 | 0.0337 | 0.0483 | 0.0112 |
| RV_PA | G core+all | k-NN | Drop 3D LoG | 0.7390 | 0.7231 | 0.1104 | 0.3483 | 0.1305 |
| RV_PA | G core+all | CE | Gaussian .05 SD | 0.8203 | 0.7669 | 0.0866 | 0.1096 | 0.0902 |
| RV_PA | G core+all | CE | Intensity +.1 SD | 0.8203 | 0.8186 | 0.0101 | 0.0135 | 0.0108 |
| RV_PA | G core+all | CE | Intensity scale 1.1 | 0.8203 | 0.8182 | 0.0085 | 0.0228 | 0.0097 |
| RV_PA | G core+all | CE | Drop 3D LoG | 0.8203 | 0.7878 | 0.1614 | 0.3128 | 0.1353 |
| RV_PA | G core+all | CE+SupCon | Gaussian .05 SD | 0.8245 | 0.7733 | 0.0871 | 0.1167 | 0.0260 |
| RV_PA | G core+all | CE+SupCon | Intensity +.1 SD | 0.8245 | 0.8220 | 0.0123 | 0.0192 | 0.0035 |
| RV_PA | G core+all | CE+SupCon | Intensity scale 1.1 | 0.8245 | 0.8229 | 0.0090 | 0.0231 | 0.0027 |
| RV_PA | G core+all | CE+SupCon | Drop 3D LoG | 0.8245 | 0.7938 | 0.1481 | 0.6510 | 0.0694 |
| RV_PA | G core+all | CE+SupCon+Spatial | Gaussian .05 SD | 0.8210 | 0.7737 | 0.0787 | 0.1039 | 0.0225 |
| RV_PA | G core+all | CE+SupCon+Spatial | Intensity +.1 SD | 0.8210 | 0.8218 | 0.0063 | 0.0167 | 0.0020 |
| RV_PA | G core+all | CE+SupCon+Spatial | Intensity scale 1.1 | 0.8210 | 0.8209 | 0.0052 | 0.0243 | 0.0020 |
| RV_PA | G core+all | CE+SupCon+Spatial | Drop 3D LoG | 0.8210 | 0.7920 | 0.1423 | 0.6137 | 0.0619 |

RV/PA G Gaussian noise: k-NN 0.7390→0.7369, CE 0.8203→0.7669, SupCon 0.8245→0.7733, Spatial 0.8210→0.7737. MLP сохраняет более высокий абсолютный F1, но заметно сильнее теряет относительно clean. Утверждение об общей большей устойчивости learned embeddings не подтверждено. Полные парные Δ по cases сохранены в robustness.json; raw embedding RMSE между разными моделями не является общей шкалой.


## 10. Признаки и ошибки

Global permutation importance: fixed validation subset 32/class/case, перестановка канала внутри case между классами, без refit. Локальная ablation в интерфейсе заменяет выбранный канал train mean и показывает изменение вероятности текущего predicted class. Абсолютные веса Linear не считаются universal importance. Перестановка/отключение разрывают корреляции, поэтому важность не причинная и не равна уникальной информации.

| Пара | Inputs | Method | Top feature | F1 drop |
|---|---|---|---|---|
| LV_MYO | C | k-NN | 3D Mean | 0.2028 |
| LV_MYO | C | CE | 3D Mean | 0.2591 |
| LV_MYO | C | CE+SupCon | 3D Mean | 0.2407 |
| LV_MYO | C | CE+SupCon+Spatial | 3D Mean | 0.2404 |
| LV_MYO | G | k-NN | 3D Gaussian 1 | 0.0630 |
| LV_MYO | G | CE | Intensity | 0.0961 |
| LV_MYO | G | CE+SupCon | Intensity | 0.1058 |
| LV_MYO | G | CE+SupCon+Spatial | Intensity | 0.1058 |
| RV_PA | C | k-NN | 3D Std | 0.1244 |
| RV_PA | C | CE | 3D Std | 0.2562 |
| RV_PA | C | CE+SupCon | 3D Std | 0.2318 |
| RV_PA | C | CE+SupCon+Spatial | 3D Std | 0.2407 |
| RV_PA | G | k-NN | 3D Gaussian 2 | 0.1627 |
| RV_PA | G | CE | 3D Gaussian 2 | 0.4080 |
| RV_PA | G | CE+SupCon | 3D Gaussian 2 | 0.3995 |
| RV_PA | G | CE+SupCon+Spatial | 3D Gaussian 2 | 0.4024 |

Число исправленных/новых ошибок относительно k-NN для каждого case/class/метода e16 — error_transitions.csv. Это парное сравнение тех же IDs, включая регрессии. В интерфейсе point/lasso связывает source XYZ и три плоскости, показывает probability и признаки выбранного voxel; true class используется только для окраски/оценки. Selected voxel вне пары получает предупреждение об условности двух вероятностей.

| Пара | Inputs | Метод e16 | Исправлены | Новые ошибки | Оба ошиблись |
|---|---|---|---|---|---|
| LV_MYO | C | CE | 57 | 45 | 164 |
| LV_MYO | C | CE+SupCon | 62 | 45 | 159 |
| LV_MYO | C | CE+SupCon+Spatial | 63 | 47 | 158 |
| LV_MYO | G | CE | 47 | 55 | 161 |
| LV_MYO | G | CE+SupCon | 43 | 52 | 165 |
| LV_MYO | G | CE+SupCon+Spatial | 45 | 52 | 163 |
| RV_PA | C | CE | 661 | 328 | 887 |
| RV_PA | C | CE+SupCon | 671 | 340 | 877 |
| RV_PA | C | CE+SupCon+Spatial | 660 | 317 | 888 |
| RV_PA | G | CE | 747 | 287 | 683 |
| RV_PA | G | CE+SupCon | 735 | 249 | 695 |
| RV_PA | G | CE+SupCon+Spatial | 725 | 260 | 705 |


## 11. Ограничения и проверки

Предыдущие 462 файлов экспериментов/отчётов сохранены byte-for-byte. Хеши 116 файлов CT/масок 58 разрешённых cases проверены повторно. Frozen split, production preprocessing и веса CHD68 v1/v2 совпадают с исходным snapshot; receipts 36 checkpoints совпадают с SHA. Sealed test не открывался. Аудит: integrity.json, sample_counts.csv, embedding_audit.csv; версии: versions.json.

47 автоматических тестов passed: SupCon и градиенты, spatial neighbors, class mapping, scaler train-only, seed, source IDs, checkpoint strict roundtrip/provenance, coordinate/boundary/halo regression, no GT feature schema, frozen split/groups, inference regression. 17 проверок Streamlit AppTest охватывают все три размера, три losses, PCA/UMAP, forward layers, validation и gating test: ui_verification.json. Реальный Chrome проверил клики во всех трёх плоскостях, lasso 64 source IDs, выбор CE e16 и embedding inspector; crosshair и overlay count совпадают, JS errors отсутствуют: browser_verification.json / browser_learnable.png / browser_embedding.png. Производственный training regression test не запускался.

Ограничения: один seed на архитектуру, десять validation случаев, early stopping на повторно использованной validation, sampled foreground pair-only classification, GT-stratified sampling не отражает natural prevalence; class/case баланс в train; features 5×5×5 в индексной сетке; физические единицы неизвестны. Несколько размеров embedding — кандидаты разработки, максимум по ним не независимая оценка. Отсутствует whole-volume evaluation, multiclass background/foreground, external scanner validation и доказательство роста Dice/клинической применимости.


## 12. Следующий этап магистерской

Сначала повторить заранее фиксированные C/G CE и SupCon с несколькими seeds и group-wise cross-validation только внутри 48 train, оценить устойчивость на согласованных возмущениях CT с последующим повторным feature extraction. CE — обязательный сильный baseline; SupCon рассматривать как небольшой возможный increment; Spatial пока не считать улучшением. Проверить, сохраняется ли выигрыш на всех foreground + background при case-level протоколе и естественной распространённости классов.

Затем отдельным этапом сравнить настоящий сегментатор с intensity-only, intensity+3D-core и intensity+Gaussian/learned branch, сохраняя одинаковые архитектуру/augmentation/budget. Требуются whole-volume Dice/HD95 и локализация ошибок на границах; выводы нынешней voxel classification их не заменяют. Текущий этап не запускает обучение такого сегментатора и не меняет CHD68 v1/v2.


## Воспроизведение

Команды из repository (PowerShell); existing frozen artifacts переиспользуются, параметры protocol.json не меняются:
```powershell
.venv/Scripts/python.exe -m ct_lab.learnable_train
.venv/Scripts/python.exe -m ct_lab.learnable_diagnostics
.venv/Scripts/python.exe -m ct_lab.learnable_artifacts
.venv/Scripts/python.exe -m ct_lab.learnable_verify_ui
.venv/Scripts/python.exe -m pytest tests/test_learnable_features.py tests/test_spatial_features.py tests/test_filtering_features.py tests/test_ct_lab.py tests/test_cardiac_inference.py -q
.venv/Scripts/python.exe -m ct_lab.learnable_package
.venv/Scripts/python.exe -m ct_lab.learnable_launch
node ct_lab/learnable_browser.cjs
```

Для полного повторного обучения используйте отдельную копию проекта без outputs/learnable_feature_03; исходные данные и frozen split сохранить. Существующие receipts позволяют безопасно продолжать завершённый run, не перезаписывая обученные модели.

| Library | Version |
|---|---|
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| torch | 2.14.1+cu130 |
| nibabel | 5.4.2 |
| scikit-learn | 1.9.1 |
| streamlit | 1.50.0 |
| plotly | 6.3.1 |
| umap-learn | 0.5.12 |
| python | 3.13.3 |
| platform | Windows-10-10.0.19045-SP0 |
