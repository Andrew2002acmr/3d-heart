# MULTICLASS_FEATURE_EXPERIMENT_04

Начат 2026-10-28; завершён 2026-10-29. Восьмиклассовое representation learning на ImageCHD68, продолжение разработки. Validation многократно использовалась; независимой финальной/клинической оценки нет. Test закрыт. Полноценная сегментационная сеть не обучалась; CHD68 v1/v2 только inference.

## 1. Протокол и границы выводов

### Ответы на исследовательские вопросы

1. Восьмиклассовое преимущество: E CE clean 0.2296 против E k-NN 0.2216; combined CE 0.2332. Это natural development F1, а не прежняя binary задача.

2. Лучшие классы у заранее выбранной E CE combined: MYO=0.2839, AO=0.1697, RV=0.1460. Трудные: LA=0.0998, PA=0.0948, RA=0.0895.

3. Наиболее частые FG→FG путаницы first seed: RA→RV: 335, LA→AO: 295, RA→LV: 270. Background ошибки показаны отдельно в CM/FPR.

4. Gaussian2: D CE clean 0.2295 против C 0.2345, Δ=-0.0050. Канал не считается полезным автоматически.

5. CT augmentation/noise12: E clean-trained CE 0.2279, combined-trained CE 0.2300; clean F1 соответственно 0.2296/0.2332. Это измерение переноса на новый synthetic recipe, не реальный scanner domain.

6. SupCon поверх combined CE: clean Δ=+0.0030, noise12 Δ=+0.0044. Clean paired case CI95=[-0.0009, +0.0068] включает ноль: убедительное дополнительное преимущество не установлено.

7. Spatial E clean Δ к CE=+0.0009; paired case CI95=[-0.0019, +0.0036] включает ноль. Убедительное устойчивое улучшение не установлено.

8. E CE combined: SD(seed)=0.0014, SD(case seed-means)=0.0354, case CI95=[0.2120, 0.2554]. Интервал условен на reused development протоколе.

9. Dense prediction выполнен на полном ct_1001 и fixed ROI трёх случаев. Классовые Dice/HD95 и примеры ниже являются измеренными dense результатами, не следствием sampled F1.

10. Полное ct_1001 MLP: 105770 false foreground components без overlap с GT того же класса; BG→FG=11030261 voxels. Это реальный dense connected-component analysis, не sparse fragment count.

11. Полное ct_1001 mean Dice present FG: E_CE_combined_s20261029=0.1897, CHD68_v1=0.9589, CHD68_v2=0.9394. MLP существенно уступает CHD68 в этом train-case сравнении; отдельно приведены development ROI. Локальный encoder не подтвердил пригодность как самостоятельная основа whole-heart segmentation.

12. Для следующего сегментатора необходим global/multiscale anatomical context и более реалистичный background/prior protocol. Class-balanced MLP training не отражает natural priors; calibration/prior correction требует отдельного train-only эксперимента, в этой frozen матрице не подбиралась.

Frozen 48 train / 10 development validation / 10 sealed test. 0 Background, 1 LV, 2 RV, 3 LA, 4 RA, 5 MYO, 6 AO, 7 PA. Прочие source labels → ignore=255, исключаются из loss и метрик; ignore не background. Полная предрегистрация: outputs/multiclass_feature_04/protocol.json. 36 основных моделей, три seed: [20261029, 20261030, 20261031], плюс шесть обучений для 3-fold group-wise CV внутри train. Case/group не делятся между folds.

Все A–F и augmentation/loss варианты зафиксированы до validation. Основная метрика — mean fixed-eight-label macro F1 по cases (absent labels: zero_division=0); дополнительно present-class macro, foreground macro и BA по присутствующим targets. Macro F1 на natural samples и ROI/full-volume Dice — различные измерения, одно не выводится из другого.

## 2. Sampling и natural validation

Train: 16 Z-strata, 128 uniform points/class + до 32 boundary points/class. Background: 64 near (6-connected dilation radius 6 index steps) + 64 far, затем boundary extra. Экспертные masks разрешены только для train sampling/targets и same-class spatial loss. Natural train: 4096, validation: 8192 равномерных уникальных IDs/case по всему source volume; IDs выбираются по image shape до чтения GT. Foreground и ignore не исключаются при выборе. Маска затем только задаёт targets/valid для оценки. Это spatial point sample естественного распределения, дополненный отдельным dense ROI/full-volume экспериментом.

Source C-order voxel IDs/XYZ/seed/case сохранены. Natural class counts: natural_class_counts.csv. Scaler fit только clean train anchors текущего fold, никогда validation; одинаковые scaler и anchors между losses/seed для одного набора. GT, distance-to-GT и absolute coordinates не входят в признаки.

| Partition | Cases | Natural selected points | Valid 0–7 | Ignored |
|---|---|---|---|---|
| train | 48 | 196608 | 196599 | 9 |
| validation | 10 | 81920 | 81903 | 17 |

## 3. Архитектура, признаки и обучение

Input → Linear64 → ReLU → Linear32 → ReLU → Linear16 → Linear8. CE; CE+0.1 SupCon (L2-normalized, temperature=.1, self excluded); CE+0.1 Spatial (без SupCon, в отличие от этапа03). Spatial использует distinct 6-neighbor того же train класса, без пересечения GT границ, только как loss; отсутствующие пары пропускаются. Восемь классов поддерживаются существующим FeatureEncoder; веса CHD68 не являются инициализацией.

| Набор | Каналы |
|---|---|
| A Intensity | Intensity |
| B 2D | Intensity, XY Gradient, XY Mean, XY Std, XY LoG |
| C 3D-core | Intensity, 3D Mean, 3D Std, 3D Gradient, 3D LoG |
| D core+Gaussian2 | Intensity, 3D Mean, 3D Std, 3D Gradient, 3D LoG, 3D Gaussian 2 |
| E core+all | Intensity, 3D Mean, 3D Std, 3D Gradient, 3D LoG, 3D Gaussian 0.5, 3D Gaussian 1, 3D Gaussian 2 |
| F core+Z | Intensity, 3D Mean, 3D Std, 3D Gradient, 3D LoG, Z I -1, Z I +1, Z1 Std |

Uniform mean/std window 5; central gradients с one-sided global edges; LoG σ=1; Gaussian .5/1/2, truncate4, reflect. F добавляет только Z−1, Z+1 и Z local SD, не десятки дублирующих каналов. Source intensity не HU. Sigma/spacing — неизвестные header grid units, не мм. Новый point extractor повторно использует stage02 stencil/kernels; dense engine сверяется с ним на всех краях и стыках блоков.

AdamW lr=.001, wd=.0001, batch256, максимум20 эпох, patience4, min_delta=.0002, CPU threads1. Class-balanced batches, внутри класса inverse count(case,class) sampling без replacement внутри batch. Early stopping по natural development case F1; одинаковая инициализация при одинаковом input/seed. Train augmentation: 50% clean / 50% выбранная augmented version; valid translation сохраняет тот же анатомический target, вне FOV используется clean вместо испорченной пары. Основной embedding=16, API допускает8/32, в этом этапе dimension search не выполняется.

## 4. Сохраняется ли преимущество MLP? Все clean результаты

| Inputs | Метод | Train augmentation | F1 mean(seed) | SD(seed) | SD(case means) | Case bootstrap CI95 | BA mean |
|---|---|---|---|---|---|---|---|
| A Intensity | CE | clean | 0.1996 | 0.0049 | 0.0301 | [0.1819, 0.2198] | 0.3296 |
| B 2D | CE | clean | 0.2192 | 0.0049 | 0.0325 | [0.2003, 0.2410] | 0.3722 |
| C 3D-core | CE | clean | 0.2345 | 0.0070 | 0.0345 | [0.2139, 0.2569] | 0.3876 |
| D core+Gaussian2 | CE | clean | 0.2295 | 0.0085 | 0.0385 | [0.2074, 0.2552] | 0.3941 |
| E core+all | CE | clean | 0.2296 | 0.0060 | 0.0343 | [0.2097, 0.2520] | 0.3907 |
| F core+Z | CE | clean | 0.2284 | 0.0065 | 0.0323 | [0.2097, 0.2501] | 0.3932 |
| E core+all | CE+SupCon | clean | 0.2279 | 0.0074 | 0.0378 | [0.2061, 0.2533] | 0.3905 |
| E core+all | CE+Spatial | clean | 0.2306 | 0.0066 | 0.0349 | [0.2106, 0.2535] | 0.3946 |
| E core+all | CE | intensity | 0.2207 | 0.0029 | 0.0354 | [0.2002, 0.2441] | 0.3906 |
| E core+all | CE | noise | 0.2305 | 0.0007 | 0.0347 | [0.2099, 0.2524] | 0.3937 |
| E core+all | CE | combined | 0.2332 | 0.0014 | 0.0354 | [0.2120, 0.2554] | 0.3934 |
| E core+all | CE+SupCon | combined | 0.2362 | 0.0006 | 0.0364 | [0.2150, 0.2602] | 0.3970 |
| C 3D-core | k-NN | clean | 0.2208 | 0.0000 | 0.0208 | [0.2091, 0.2344] | 0.3595 |
| E core+all | k-NN | clean | 0.2216 | 0.0000 | 0.0231 | [0.2078, 0.2367] | 0.3619 |
| A Intensity | k-NN | clean | 0.1934 | 0.0000 | 0.0174 | [0.1832, 0.2043] | 0.3000 |
| B 2D | k-NN | clean | 0.2142 | 0.0000 | 0.0244 | [0.2004, 0.2304] | 0.3440 |
| D core+Gaussian2 | k-NN | clean | 0.2235 | 0.0000 | 0.0270 | [0.2079, 0.2413] | 0.3625 |
| F core+Z | k-NN | clean | 0.2249 | 0.0000 | 0.0200 | [0.2135, 0.2382] | 0.3704 |

Сравнение с k-NN делается на тех же train/validation IDs и channels. Числа этапа03 binary GT-stratified sampling нельзя напрямую сопоставлять с нынешними natural восьмиклассовыми F1.

## 5. Gaussian2, дополнительные Gaussian и Z-context

A–F CE clean сравниваются при одинаковых sampling, architecture, three seeds и protocol. Таблица выше и feature_sets.html показывают отдельно C→D (Gaussian2), C→E (all Gaussian) и C→F (Z). Классовые значения и CM каждого seed: results.json / metrics_by_class.csv. Сильные train correlations |r|>.95:

| Feature1 | Feature2 | r |
|---|---|---|
| Intensity | XY Mean | 0.9895 |
| Intensity | 3D Mean | 0.9766 |
| Intensity | 3D Gaussian 2 | 0.9530 |
| Intensity | 3D Gaussian 0.5 | 0.9994 |
| Intensity | 3D Gaussian 1 | 0.9925 |
| Intensity | Z I -1 | 0.9719 |
| Intensity | Z I +1 | 0.9715 |
| XY Gradient | XY Std | 0.9603 |
| XY Mean | 3D Mean | 0.9934 |
| XY Mean | 3D Gaussian 2 | 0.9809 |
| XY Mean | 3D Gaussian 0.5 | 0.9929 |
| XY Mean | 3D Gaussian 1 | 0.9975 |
| XY Mean | Z I -1 | 0.9673 |
| XY Mean | Z I +1 | 0.9668 |
| 3D Mean | 3D Gaussian 2 | 0.9947 |
| 3D Mean | 3D Gaussian 0.5 | 0.9827 |
| 3D Mean | 3D Gaussian 1 | 0.9953 |
| 3D Mean | Z I -1 | 0.9637 |
| 3D Mean | Z I +1 | 0.9639 |
| 3D Gaussian 2 | 3D Gaussian 0.5 | 0.9614 |
| 3D Gaussian 2 | 3D Gaussian 1 | 0.9812 |
| 3D Gaussian 0.5 | 3D Gaussian 1 | 0.9959 |
| 3D Gaussian 0.5 | Z I -1 | 0.9739 |
| 3D Gaussian 0.5 | Z I +1 | 0.9735 |
| 3D Gaussian 1 | Z I -1 | 0.9734 |
| 3D Gaussian 1 | Z I +1 | 0.9733 |

## 6. Какие классы лучше и какие путаются

| Модель | BG | LV | RV | LA | RA | MYO | AO | PA |
|---|---|---|---|---|---|---|---|---|
| E CE clean | 0.9175 | 0.1353 | 0.1489 | 0.0502 | 0.0953 | 0.2848 | 0.1725 | 0.1002 |
| E CE+SupCon clean | 0.9116 | 0.1457 | 0.1497 | 0.0717 | 0.0808 | 0.2621 | 0.1753 | 0.1063 |
| E CE+Spatial clean | 0.9097 | 0.1439 | 0.1572 | 0.0693 | 0.0918 | 0.2674 | 0.1708 | 0.1041 |
| E CE intensity | 0.8871 | 0.1354 | 0.1463 | 0.1005 | 0.0766 | 0.2169 | 0.1726 | 0.1000 |
| E CE noise | 0.9137 | 0.1292 | 0.1449 | 0.0882 | 0.0892 | 0.2776 | 0.1697 | 0.0966 |
| E CE combined | 0.9159 | 0.1328 | 0.1460 | 0.0998 | 0.0895 | 0.2839 | 0.1697 | 0.0948 |
| E CE+SupCon combined | 0.9192 | 0.1302 | 0.1533 | 0.0744 | 0.0928 | 0.2949 | 0.1772 | 0.1000 |

Per-class precision/recall/F1, one-vs-rest FPR и BG→class FPR — metrics_by_class.csv; 8×8 pooled и case CM — results.json. Ignore никогда не становится ошибкой background. Основной первый seed E CE combined CM:

| GT / pred | Background | LV | RV | LA | RA | MYO | AO | PA |
|---|---|---|---|---|---|---|---|---|
| Background | 64426 | 313 | 454 | 172 | 1751 | 6712 | 1167 | 815 |
| LV | 0 | 120 | 116 | 11 | 31 | 7 | 161 | 5 |
| RV | 0 | 213 | 150 | 30 | 63 | 7 | 214 | 12 |
| LA | 23 | 135 | 103 | 66 | 79 | 16 | 295 | 66 |
| RA | 32 | 270 | 335 | 25 | 158 | 109 | 198 | 26 |
| MYO | 338 | 33 | 109 | 0 | 35 | 1472 | 49 | 13 |
| AO | 9 | 82 | 139 | 12 | 39 | 6 | 247 | 25 |
| PA | 31 | 57 | 42 | 35 | 23 | 4 | 150 | 67 |

## 7. CT augmentations и устойчивость

Обоснование амплитуд: шум8/12 — около0.2/0.3% source range4095; shift24–40 ≈0.6–1%; scale2–4%, mild gamma; smoothing σ.35/.55; integer shifts один voxel. Эти воздействия мягкие, заранее фиксированы, но не измерены на реальных сканерах. Rotation не добавлялась: для проверки гипотезы достаточно exact translations без интерполяционного размытия targets. Все CT-derived features пересчитаны после каждой операции, independent feature noise не использован.

Train recipes:
```json
{
  "clean": {},
  "intensity": {
    "shift": 32.0,
    "scale": 1.03,
    "gamma": 1.04
  },
  "noise": {
    "noise": 8.0
  },
  "combined": {
    "noise": 8.0,
    "shift": -24.0,
    "scale": 0.98,
    "gamma": 0.98,
    "smooth": 0.35,
    "translation": [
      1,
      0,
      -1
    ]
  }
}
```
Heldout validation recipes (другие параметры и seeds):
```json
{
  "clean": {},
  "noise12": {
    "noise": 12.0
  },
  "shift-40": {
    "shift": -40.0
  },
  "scale.96": {
    "scale": 0.96
  },
  "contrast.94": {
    "gamma": 0.94
  },
  "smooth.55": {
    "smooth": 0.55
  },
  "translateY+Z": {
    "translation": [
      0,
      1,
      1
    ]
  },
  "heldout-combined": {
    "noise": 12.0,
    "shift": 40.0,
    "scale": 1.04,
    "gamma": 1.06,
    "smooth": 0.55,
    "translation": [
      -1,
      1,
      0
    ]
  }
}
```

Intensity scale вокруг fixed source1024 (не HU); gamma по clipped source/4095. Translation: source XYZ→XYZ+integer shift, nearest image/target при train, вне поля valid=False. При validation GT не задаёт положение: исходные IDs сохраняются и переносятся известным image transform; baseline для Δ ограничен теми же common-valid IDs. Natural denominator при intensity-only transformations неизменен. Значения noisy ниже — именно noise12, все семь unseen transformations в robustness.json.

| Model | Clean F1 | Noise12 F1 | ΔF1 | Noise seed SD |
|---|---|---|---|---|
| E CE clean | 0.2296 | 0.2279 | -0.0017 | 0.0058 |
| E CE+SupCon clean | 0.2279 | 0.2256 | -0.0024 | 0.0079 |
| E CE+Spatial clean | 0.2306 | 0.2280 | -0.0026 | 0.0064 |
| E CE intensity | 0.2207 | 0.2176 | -0.0031 | 0.0024 |
| E CE noise | 0.2305 | 0.2283 | -0.0022 | 0.0009 |
| E CE combined | 0.2332 | 0.2300 | -0.0032 | 0.0008 |
| E CE+SupCon combined | 0.2362 | 0.2345 | -0.0018 | 0.0013 |

Абсолютный noisy F1 и относительное падение — разные свойства. E CE combined имеет более высокий noise12 F1, чем clean-trained CE, но и более высокий clean F1; его падение относительно clean больше. Универсальное повышение устойчивости аугментацией не подтверждено. Независимый feature-space noise этапа03 и нынешний image-space noise имеют разные амплитуды/задачи и не сравниваются причинно напрямую.

Чувствительность классов (noise12 минус clean), seed mean:

| Model | Background | LV | RV | LA | RA | MYO | AO | PA |
|---|---|---|---|---|---|---|---|---|
| E CE clean | -0.0028 | 0.0074 | -0.0005 | -0.0017 | 0.0006 | -0.0090 | -0.0063 | 0.0000 |
| E CE+SupCon clean | -0.0047 | 0.0119 | -0.0000 | 0.0021 | -0.0034 | -0.0126 | -0.0091 | 0.0004 |
| E CE+Spatial clean | -0.0041 | 0.0082 | -0.0009 | 0.0004 | -0.0031 | -0.0107 | -0.0091 | 0.0006 |
| E CE intensity | -0.0062 | 0.0139 | -0.0006 | 0.0012 | -0.0030 | -0.0098 | -0.0063 | -0.0026 |
| E CE noise | -0.0044 | 0.0116 | 0.0031 | 0.0024 | -0.0014 | -0.0129 | -0.0038 | -0.0015 |
| E CE combined | -0.0053 | 0.0134 | 0.0008 | 0.0047 | -0.0024 | -0.0158 | -0.0078 | -0.0016 |
| E CE+SupCon combined | -0.0046 | 0.0186 | -0.0013 | 0.0057 | -0.0005 | -0.0153 | -0.0079 | -0.0008 |

## 8. SupCon и экспериментальная Spatial Consistency

Сравнивать CE/SupCon/Spatial необходимо при E и одинаковой augmentation. Clean E три losses и E combined CE/SupCon показаны выше. Полезность регуляризации оценивается также по шумовым метрикам и seed SD; отдельный удачный seed или красивый UMAP не считаются доказательством. Collapse проверен по raw и unit embedding variance; все обученные checkpoints прошли проверку.

## 9. Стабильность seed, случаев и group CV

Case CI: 2000 bootstrap resamples cases после усреднения scores между seed. Это описательная неопределённость десяти development случаев, не коррекция за reuse validation/model selection; seed SD показан отдельно (sample SD, ddof=1, n=3). Sampled voxels не считаются независимыми пациентами. Paired ΔF1 относительно k-NN на том же feature set:

| Inputs | Method | Aug | Mean paired Δ | Case CI95 | Case SD |
|---|---|---|---|---|---|
| A Intensity | CE | clean | 0.0062 | [-0.0045, 0.0177] | 0.0177 |
| B 2D | CE | clean | 0.0051 | [-0.0016, 0.0130] | 0.0115 |
| C 3D-core | CE | clean | 0.0137 | [0.0033, 0.0250] | 0.0178 |
| D core+Gaussian2 | CE | clean | 0.0060 | [-0.0043, 0.0185] | 0.0184 |
| E core+all | CE | clean | 0.0080 | [-0.0012, 0.0178] | 0.0157 |
| F core+Z | CE | clean | 0.0035 | [-0.0064, 0.0149] | 0.0173 |
| E core+all | CE+SupCon | clean | 0.0063 | [-0.0039, 0.0185] | 0.0183 |
| E core+all | CE+Spatial | clean | 0.0089 | [-0.0004, 0.0193] | 0.0164 |
| E core+all | CE | intensity | -0.0009 | [-0.0108, 0.0102] | 0.0173 |
| E core+all | CE | noise | 0.0089 | [-0.0005, 0.0184] | 0.0155 |
| E core+all | CE | combined | 0.0116 | [0.0025, 0.0214] | 0.0157 |
| E core+all | CE+SupCon | combined | 0.0146 | [0.0050, 0.0254] | 0.0166 |

Прямые парные ablations: усреднение трёх seed внутри каждого случая, затем bootstrap только случаев. Это описательные CI без поправки на множественные сравнения и reuse development validation; параметры по ним не менялись.

| Comparison | Mean paired ΔF1 | Case CI95 | Case SD |
|---|---|---|---|
| Gaussian2: D−C | -0.0050 | [-0.0093, 0.0006] | 0.0080 |
| All Gaussian: E−C | -0.0049 | [-0.0099, -0.0002] | 0.0078 |
| Z context: F−C | -0.0061 | [-0.0126, -0.0004] | 0.0099 |
| Combined augmentation: E CE combined−clean | 0.0036 | [-0.0014, 0.0088] | 0.0085 |
| SupCon combined−CE combined | 0.0030 | [-0.0009, 0.0068] | 0.0062 |
| SupCon clean−CE clean | -0.0017 | [-0.0048, 0.0019] | 0.0054 |
| Spatial clean−CE clean | 0.0009 | [-0.0019, 0.0036] | 0.0044 |

3-fold CV: все voxels/slices release-group остаются вместе. Scaler и weights только на остальных train folds. Fold holdout служит early stopping, поэтому это development CV, не nested unbiased evaluation. Внешние 10 validation cases не входят ни в один CV fit.

| Fold | Inputs | Method | Aug | Train natural F1 | Holdout F1 | Cases |
|---|---|---|---|---|---|---|
| 0 | C 3D-core | CE | clean | 0.2208 | 0.2158 | 16 |
| 0 | E core+all | CE | combined | 0.2416 | 0.2353 | 16 |
| 1 | C 3D-core | CE | clean | 0.2349 | 0.2372 | 16 |
| 1 | E core+all | CE | combined | 0.2466 | 0.2467 | 16 |
| 2 | C 3D-core | CE | clean | 0.2335 | 0.2200 | 16 |
| 2 | E core+all | CE | combined | 0.2364 | 0.2220 | 16 |

## 10. Dense prediction, фон и ложные островки

Fixed ROI center fraction [.65,.45,.5], size[128,128,96], clipped только по CT shape. Никакой GT-localized crop. Три разрешённых случая: ct_1001 train, ct_1027/ct_1028 development validation. Сравниваются A CE clean, C CE clean, E CE combined first seed, CHD68 v1/v2. Полное ct_1001 дополнительно рассчитано для E CE combined и обеих CHD68. Только inference, без morphology/GT cleanup. Полные маски и 8-class probability/entropy maps сохранены на native grid.

Dense block64×64×16 с exact radius8 halo; float32 features/softmax, float16 disk probability, uint8 final mask из float32 argmax; float16 может создавать округлённые ties, финальная маска authoritative. Нет полного dense feature tensor датасета. CPU-only поддерживается; CHD68 inference использует RTX2060 при наличии, CPU fallback явный. Component counts — по реальной dense prediction, не sparse point graph.

| Case | Scope | Model | Dense macro F1 | Mean Dice present FG | False FG voxels on BG | False FG components no GT overlap |
|---|---|---|---|---|---|---|
| ct_1001 | predeclared fixed ROI; boundary-truncated components | A_CE_clean_s20261029 | 0.3143 | 0.2368 | 64743 | 24371 |
| ct_1001 | predeclared fixed ROI; boundary-truncated components | C_CE_clean_s20261029 | 0.3161 | 0.2355 | 46670 | 8375 |
| ct_1001 | predeclared fixed ROI; boundary-truncated components | E_CE_combined_s20261029 | 0.3345 | 0.2576 | 56045 | 5485 |
| ct_1001 | predeclared fixed ROI; boundary-truncated components | CHD68_v1 | 0.8820 | 0.8702 | 13595 | 19 |
| ct_1001 | predeclared fixed ROI; boundary-truncated components | CHD68_v2 | 0.8303 | 0.8141 | 20084 | 55 |
| ct_1001 | full native volume | E_CE_combined_s20261029 | 0.2756 | 0.1897 | 11030261 | 105770 |
| ct_1001 | full native volume | CHD68_v1 | 0.9636 | 0.9589 | 162521 | 216 |
| ct_1001 | full native volume | CHD68_v2 | 0.9465 | 0.9394 | 229368 | 398 |
| ct_1027 | predeclared fixed ROI; boundary-truncated components | A_CE_clean_s20261029 | 0.1708 | 0.1667 | 31450 | 24845 |
| ct_1027 | predeclared fixed ROI; boundary-truncated components | C_CE_clean_s20261029 | 0.2081 | 0.2029 | 26514 | 17545 |
| ct_1027 | predeclared fixed ROI; boundary-truncated components | E_CE_combined_s20261029 | 0.2047 | 0.1999 | 29185 | 13334 |
| ct_1027 | predeclared fixed ROI; boundary-truncated components | CHD68_v1 | 0.3420 | 0.3659 | 32537 | 199 |
| ct_1027 | predeclared fixed ROI; boundary-truncated components | CHD68_v2 | 0.3821 | 0.4233 | 15459 | 275 |
| ct_1028 | predeclared fixed ROI; boundary-truncated components | A_CE_clean_s20261029 | 0.2755 | 0.1939 | 272587 | 19633 |
| ct_1028 | predeclared fixed ROI; boundary-truncated components | C_CE_clean_s20261029 | 0.2654 | 0.1804 | 249367 | 8838 |
| ct_1028 | predeclared fixed ROI; boundary-truncated components | E_CE_combined_s20261029 | 0.2751 | 0.1906 | 242405 | 6748 |
| ct_1028 | predeclared fixed ROI; boundary-truncated components | CHD68_v1 | 0.6597 | 0.6192 | 105950 | 50 |
| ct_1028 | predeclared fixed ROI; boundary-truncated components | CHD68_v2 | 0.6690 | 0.6303 | 115410 | 45 |

False component = компонент данного predicted класса без overlap с GT того же класса; включает wrong-structure components и не тождественен только фоновому FP. BG→FG voxel count отдельно. Размеры ВСЕХ компонент: dense/**/metrics.json; background distance bins0–2/2–6/6–20/>20 index units вычислены только после inference по экспертным foreground структурам. Они никогда не являются features или ROI criterion.

Полный ct_1001: Dice / HD95(index) / 6-connected components:

| Model | Class | Dice | HD95 index | Components | FP voxels | False islands |
|---|---|---|---|---|---|---|
| E_CE_combined_s20261029 | Background | 0.8772 | 91.3947 | 17176 | 161627 | 1194 |
| E_CE_combined_s20261029 | LV | 0.2500 | 197.9933 | 11786 | 1583902 | 11515 |
| E_CE_combined_s20261029 | RV | 0.1041 | 257.7091 | 11832 | 867026 | 11111 |
| E_CE_combined_s20261029 | LA | 0.1705 | 72.7186 | 13472 | 331315 | 10166 |
| E_CE_combined_s20261029 | RA | 0.2267 | 201.7353 | 16705 | 1334489 | 14865 |
| E_CE_combined_s20261029 | MYO | 0.2286 | 273.0531 | 27626 | 7650830 | 27302 |
| E_CE_combined_s20261029 | AO | 0.1979 | 177.5500 | 15147 | 2723976 | 14733 |
| E_CE_combined_s20261029 | PA | 0.1502 | 81.5721 | 16495 | 727943 | 16078 |
| CHD68_v1 | Background | 0.9968 | 2.0000 | 57 | 160659 | 41 |
| CHD68_v1 | LV | 0.9655 | 2.0000 | 13 | 18338 | 10 |
| CHD68_v1 | RV | 0.9681 | 2.2361 | 5 | 22054 | 4 |
| CHD68_v1 | LA | 0.9613 | 3.6056 | 134 | 42396 | 87 |
| CHD68_v1 | RA | 0.9724 | 2.4495 | 5 | 30699 | 3 |
| CHD68_v1 | MYO | 0.9437 | 2.2361 | 21 | 108342 | 15 |
| CHD68_v1 | AO | 0.9608 | 2.0000 | 16 | 19153 | 6 |
| CHD68_v1 | PA | 0.9404 | 4.4721 | 140 | 50350 | 91 |
| CHD68_v2 | Background | 0.9959 | 4.2426 | 213 | 191015 | 114 |
| CHD68_v2 | LV | 0.9448 | 4.2426 | 17 | 13601 | 12 |
| CHD68_v2 | RV | 0.9501 | 4.8990 | 18 | 20524 | 16 |
| CHD68_v2 | LA | 0.9465 | 12.7279 | 185 | 49487 | 109 |
| CHD68_v2 | RA | 0.9596 | 5.0990 | 26 | 82079 | 24 |
| CHD68_v2 | MYO | 0.9019 | 5.0990 | 54 | 204321 | 48 |
| CHD68_v2 | AO | 0.9595 | 6.4031 | 22 | 24990 | 21 |
| CHD68_v2 | PA | 0.9137 | 12.2066 | 234 | 61964 | 168 |

## 11. Сопоставимость CHD68 v1/v2

Обе модели загружены существующим load_recovered с исторической проверкой SHA/config/preprocessing, используются их frozen clipping/context/resize/normalization. Вероятности восстановлены на исходную XY сетку до argmax. Для всех методов одинаковые исходные CT, cardiac_target 0–7/ignore255, source-grid ROI и evaluation functions. Dense Dice измерен независимо от sampled F1; HD95 только в индексных единицах. Поверхности у ignore и crop edges censored; ROI компоненты усечены, поэтому полное ct_1001 приведено отдельно. Это reused train/development сравнение, не независимая оценка.

## 12. Embeddings, importance и интерфейс

Multiclass Feature Lab добавлен в существующий CT Preprocessing Lab; Learnable Feature Explorer также переключается в восьмиклассовый режим. Выбор модели/seed/features/CT, все классы и background toggle, linked point/lasso с Three-plane Viewer, clean/noisy PCA/UMAP, CT perturbation preview, class probabilities, dense prediction/errors/entropy, 3D components и history. Служебные детали скрыты в expanders.

Проекции: фиксированные24/class/case, одинаковые source IDs clean/noisy; scaler/reducer fit только clean TRAIN; noisy train/validation только transform. Train preview noise8, heldout validation noise12. До/после encoder используют независимые bases; clean/noisy в одном representation — common basis. GT только для окраски/diagnostic subsampling, не reducer fit. Количественная geometry в raw embeddings: results.json/robustness.json, не PCA/UMAP coordinates. Importance — within-case permutation first-seed E на фиксированных natural validation1024/case; взаимодействия и корреляции ограничивают причинные выводы.

| E model | Val raw-space Silhouette seed mean | Val Fisher seed mean | Train unit embedding STD mean |
|---|---|---|---|
| E CE clean | -0.0674 | 1.7059 | 0.2146 |
| E CE+SupCon clean | -0.0260 | 1.1177 | 0.0788 |
| E CE+Spatial clean | -0.0577 | 1.5299 | 0.1311 |
| E CE intensity | -0.0612 | 1.6485 | 0.2150 |
| E CE noise | -0.0622 | 1.6424 | 0.2161 |
| E CE combined | -0.0617 | 1.6747 | 0.2152 |
| E CE+SupCon combined | -0.0244 | 1.0973 | 0.0796 |

## 13. Рекомендации и неподтверждённые выводы

Локальный MLP не имеет whole-heart context и shape prior; похожие по интенсивности кровь/сосуды/ткани вне сердца могут порождать FP, даже при хорошем binary representation. Для полноценного сегментатора нужны глобальный multiscale spatial context, background training, строгая ignore handling, image-level augmentation и проверка spatial coherence. CE остаётся обязательным baseline; Gaussian2 и SupCon следует переносить только если их измеренные paired эффекты устойчивы по seed/cases и dense метрикам. Spatial loss пока экспериментален.

Не подтверждены клиническая применимость, устойчивость к реальным scanner/domain shifts, независимое test качество и превосходство одной архитектуры вне рассмотренных cases. Natural samples конечны, rare classes могут отсутствовать; fixed-label macro штрафует такие случаи, present-class metric сохранена отдельно. CV early stopping не nested. Dense full-volume выполнен на одном train case, ROI на двух development cases; этих данных недостаточно для обобщения whole-volume quality на популяцию. Augmentation recipes синтетические и так обозначены. Segmentation network на этом этапе не обучалась.

## 14. Графики и воспроизведение

- [feature_sets](outputs/multiclass_feature_04/charts/feature_sets.html) — Одинаковые spatial IDs, естественные частоты. CE — среднее трёх seed; error bars — SD между seed, не CI пациентов.

- [F1_by_case](outputs/multiclass_feature_04/charts/F1_by_case.html) — Единица наблюдения — release-case. Разброс между случаями не заменяется количеством sampled voxels.

- [F1_by_class](outputs/multiclass_feature_04/charts/F1_by_class.html) — Class-wise F1 усреднён между тремя seed; background включён в задачу.

- [clean_vs_noisy](outputs/multiclass_feature_04/charts/clean_vs_noisy.html) — Преобразованы реальные CT, затем повторно извлечены признаки. Параметры validation отличны от train; не модель реальных сканеров. Для translations парные Δ в JSON используют common-valid IDs.

- [embedding_stability](outputs/multiclass_feature_04/charts/embedding_stability.html) — L2-normalized embeddings сравниваются на тех же source IDs; меньше — стабильнее, но это не метрика Dice.

- [class_noise_sensitivity](outputs/multiclass_feature_04/charts/class_noise_sensitivity.html) — Отрицательное значение показывает чувствительность класса к Gaussian noise на исходном изображении, σ=12 source counts.

- [confusion_8x8](outputs/multiclass_feature_04/charts/confusion_8x8.html) — Valid 0–7 voxels, ignore=255 исключён. Это pooled диагностическая таблица; основная метрика усредняется по cases.

- [importance](outputs/multiclass_feature_04/charts/importance.html) — Фиксированные natural validation IDs, перестановка внутри case. Нарушение корреляций ограничивает причинную интерпретацию; параметры после этого не менялись.

- [correlations](outputs/multiclass_feature_04/charts/correlations.html) — Корреляции только по train. Дополнительные Z-каналы ограничены тремя; feature ablations A–F проверены отдельно.

- [background_distance_errors](outputs/multiclass_feature_04/charts/background_distance_errors.html) — Distance до GT вычисляется только после inference для диагностики. Одинаковые dense ROI; расстояния не feature и не правило выбора ROI.

- [false_components](outputs/multiclass_feature_04/charts/false_components.html) — Компонент не пересекает GT того же класса; A/C CE clean изолируют дополнительный 3D context. E combined меняет и признаки, и augmentation. ROI компоненты усечены; полное ct_1001 измерено отдельно.

- [training_loss](outputs/multiclass_feature_04/charts/training_loss.html) — Early stopping на reused development validation; разные loss имеют разные масштабы. Истории всех 36 моделей сохранены.

- [training_validation_natural_case_f1](outputs/multiclass_feature_04/charts/training_validation_natural_case_f1.html) — Early stopping на reused development validation; разные loss имеют разные масштабы. Истории всех 36 моделей сохранены.

- [projection_PCA](outputs/multiclass_feature_04/charts/projection_PCA.html) — Реальные ImageCHD68 voxels, фиксированные IDs. Общая clean TRAIN basis внутри каждой колонки; разные representations имеют независимые bases. Train diagnostic σ=8, heldout validation σ=12. Вид кластеров не доказывает рост сегментационного качества.

- [projection_UMAP](outputs/multiclass_feature_04/charts/projection_UMAP.html) — Реальные ImageCHD68 voxels, фиксированные IDs. Общая clean TRAIN basis внутри каждой колонки; разные representations имеют независимые bases. Train diagnostic σ=8, heldout validation σ=12. Вид кластеров не доказывает рост сегментационного качества.

![Dense comparison ct_1001](outputs/multiclass_feature_04/dense_ct_1001.png)

![Dense comparison ct_1027](outputs/multiclass_feature_04/dense_ct_1027.png)

![Dense comparison ct_1028](outputs/multiclass_feature_04/dense_ct_1028.png)

Команды из repository, PowerShell. Завершённые receipts/cache переиспользуются; для полного повторения — отдельная копия без outputs/multiclass_feature_04, прежние эксперименты не удалять.
```powershell
.venv/Scripts/python.exe -m ct_lab.multiclass_train
.venv/Scripts/python.exe -m ct_lab.multiclass_analysis
.venv/Scripts/python.exe -m ct_lab.multiclass_dense
.venv/Scripts/python.exe -m ct_lab.multiclass_projection
.venv/Scripts/python.exe -m ct_lab.multiclass_verify_ui
.venv/Scripts/python.exe -m pytest tests/test_multiclass_lab.py tests/test_learnable_features.py tests/test_spatial_features.py tests/test_filtering_features.py tests/test_ct_lab.py tests/test_cardiac_inference.py -q
node ct_lab/multiclass_browser.cjs
.venv/Scripts/python.exe -m ct_lab.multiclass_report
.venv/Scripts/python.exe -m ct_lab.multiclass_launch
```

Integrity: 116 исходных CT/mask файлов повторно проверены; 919 прежних артефактов byte-for-byte сохранены; frozen split, production preprocessing и CHD68 v1/v2 hashes не изменились. Все42 checkpoints имеют verified SHA/provenance, source IDs проверены, validation training-neighbors отсутствуют. Автотесты: tests.xml; UI: ui_verification.json / browser_verification.json. Производственный cardiac training regression test не запускался.

| Library | Version |
|---|---|
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| torch | 2.14.1+cu130 |
| nibabel | 5.4.2 |
| scikit-learn | 1.9.1 |
| umap-learn | 0.5.12 |
| plotly | 6.3.1 |
| streamlit | 1.50.0 |
| python | 3.13.3 |
