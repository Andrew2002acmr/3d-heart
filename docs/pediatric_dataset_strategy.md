# Стратегия pediatric datasets для реконструкции сердца

Дата аудита: **05.10.2026**. Целевая задача магистерской — конструирование трёхмерной модели сердца по CT с перспективой СППР для возраста 2–17 лет. Этот этап проверяет данные и переносимость реконструкции; обучение и диагностическая классификация не запускались.

## Решение по результатам аудита

Первый собственный ML baseline предлагается сделать на **Pediatric-CT-SEG для локализации и бинарной сегментации области сердца** после проверки DICOM/RTSTRUCT. Для нормативной геометрии использовать **Pediatric Cardiac Shape Atlas**, для сегментации патологических структур — **CHD68 и данные НИИ с подтверждённым возрастом**, для необычной геометрии — **HVSMR-2.0**. Классификацию норма/патология отложить до получения достоверных контрольных данных и клинических меток.

Три результата существенно меняют первоначальные предположения:

1. CHD68 преимущественно содержит пациентов младше 2 лет; индивидуальный возраст в полученном release не найден. Ни один CHD68 случай пока нельзя включить в основную возрастную когорту только по принадлежности к набору.
2. **63 пары CHD68 совпадают с ImageCHD по имени, размеру и ZIP CRC32**. Это сильное свидетельство общих файлов, хотя CRC не является криптографическим доказательством идентичности пациентов. Нельзя считать эти наборы независимыми train/test источниками. До разделения — сверка SHA-256 доступных файлов и patient linkage, объединение повторов в одну группу.
3. Три проверенных CHD68 пары имеют NIfTI spacing `[1,1,1]` и неопределённые units. Совпадение сеток image/mask позволяет реконструкцию в координатах файла, но **не подтверждает физический масштаб**. Не подставлять typical spacing из статьи.

## Доступность и роль источников

| Источник | Modality, возраст | Клинический статус | Доступ и разметка | Роль и ограничение |
|---|---|---|---|---|
| [Pediatric Cardiac Shape Atlas](https://github.com/cmrg-lab/Pediatric-Atlas), [Zenodo](https://zenodo.org/records/18980865), [CAP](https://capchd.ucsd.edu/#/pediatric) | CMR-derived geometry, 101 здоровый пациент, 2.3–19.3 года | Нормативная здоровая когорта по описанию авторов | Публичные MATLAB tools, агрегированный PCA и topology; индивидуальные CMR/clinical data/point clouds — регистрация и DUA | Healthy geometric reference; это не открытые 101 исходные CMR и не атлас исключительно 2–17 |
| [Normal RV/LV volumes](https://pmc.ncbi.nlm.nih.gov/articles/PMC6998283/) | CMR, 149 здоровых детей, 22 дня–12 лет | Confirmed healthy cohort по критериям исследования | Статья и нормативные уравнения открыты; исходные данные могут предоставляться автором по обоснованному запросу | Научная нормативная база RV/LV и BSA; изображения и индивидуальная подвыборка не получены |
| [Pediatric-CT-SEG / TCIA](https://www.cancerimagingarchive.net/collection/pediatric-ct-seg/), [статья авторов](https://pmc.ncbi.nlm.nih.gov/articles/PMC9090951/) | CT, 359, 5 дней–16 лет | `unknown`: клинические показания, диагнозы не предоставлены | Публичный NBIA API / Data Retriever; DICOM CT + RTSTRUCT, до 29 органов | Pediatric CT representation, Heart region, DICOM QA; не healthy cohort и не chamber segmentation |
| [CHD68: авторы и диагнозы](https://github.com/XiaoweiXu/Whole-heart-and-great-vessel-segmentation-of-chd_segmentation), [Kaggle](https://www.kaggle.com/datasets/xiaoweixumedicalai/chd68-segmentation-dataset-miccai19), [MICCAI 2019](https://doi.org/10.1007/978-3-030-32245-8_53) | CT, 68, 1 месяц–21 год, большинство младше 2 | CHD + явно обозначенные Normal; отсутствие болезни не используется как норма | Публичный Kaggle API проверен без credentials; split ZIP, NIfTI image/label, 7 структур | Pathological CT segmentation; возраст, физический масштаб и overlap требуют разрешения |
| [HVSMR-2.0 publication](https://www.nature.com/articles/s41597-024-03469-9), [Figshare](https://doi.org/10.6084/m9.figshare.c.7074755.v2) | CMR, 60, <1–52; опубликованная группа 5–18 содержит 25, но не равна 2–17 | Клиническая смесь CHD, операций, дилатаций и отдельных Normal | Публичные CSV и NIfTI orig/cropped/cropped_norm, 8 blood-pool структур и endpoints | Внешняя проверка геометрии сложных сердец; MRI intensities отдельно от CT |

На дату проверки Kaggle API позволил скачивание CHD68 без аккаунта. Это наблюдение конкретного public endpoint, а не гарантия всех способов доступа. CAP DUA не принимался, регистрация не выполнялась, запросы авторам и НИИ не отправлялись.

Текущие TCIA API и collection page указывают CC BY 4.0; официальный исторический digest содержит CC BY-NC 4.0. Расхождение сохранено в registry. Для дальнейшего использования фиксировать версию, действующую лицензию и лицензию из downloaded series. Figshare HVSMR — CC BY 4.0. Открытый Zenodo atlas release — Apache 2.0; это не заменяет DUA на индивидуальные данные. Лицензию CHD68 нужно сверить с действующей карточкой и файлами release, не переносить лицензию стороннего каталога автоматически.

## Что действительно получено

Данные и outputs находятся в `D:/codexProjects/3d-hearts/data/pediatric_audit/`, вне Git новой ветки, и игнорируются исходным репозиторием. В Git попадают только helper scripts, metadata, конфигурация и небольшие числовые summaries.

| Источник | Получено и проверено локально | Возраст 2–17 |
|---|---|---|
| Atlas | Zenodo code release ~49.4 MB и AtlasFile ~27.2 MB, оба MD5 проверены; EDESatlas.mat, mean surfaces, PCA visualization, template topology | Количество неизвестно: публичных individual ages нет. Исходные CMR не скачаны |
| Normative CMR study | Проверены статья, критерии и access statement | Неизвестно без individual data; не считать все 149 подходящими |
| Pediatric-CT-SEG | Manifest, digest, API inventory: 359 CT + 359 RTSTRUCT. Один CT slice и RTSTRUCT одного пациента скачаны и разобраны | **327** по опубликованному `PatientAge`: 143 / 105 / 79 в рабочих группах 2–5 / 6–11 / 12–17 |
| CHD68 | Inventory 68 пар, диагнозы 68 случаев из README, 3 CT/mask пары с original CRC + SHA-256; все 3 прогнаны через существующий build с preview | **Неизвестно**: индивидуальный возраст не найден. Это не «0 подходящих пациентов», а 0 подтверждённых включений |
| HVSMR | clinical.csv и technical.csv, проверенные MD5; клиническая информация 59 пациентов, technical 60. MRI volumes не скачаны | **36** по полю Age, группы 15 / 6 / 15; `pat59` исключён из возрастной подвыборки из-за отсутствующего возраста |

Все количества воспроизводятся в `metadata/pediatric/registry_summary.json`. Полный реестр — `registry.json`: 487 записей (359 + 68 + 60). Агрегированные нормативные источники отдельно описаны в `source_catalog.json`; из 101 или 149 пациентов не создаются вымышленные patient rows.

Возраст HVSMR дан целыми годами; DICOM AS — опубликованный возраст с единицей D/W/M/Y. Это индивидуальный возраст в точности источника, **не точный возраст до дня**. Отбор реализован буквально `age >= 2 and age <= 17`; целое значение 17 не доказывает возраст ровно 17.000. При получении более точного возраста пограничные случаи перепроверить. Рабочие интервалы для дробных значений: [2,6), [6,12), [12,17] внутри выбранного диапазона; это исследовательская стратификация, а не медицинский стандарт.

## Реестр и healthy/normal

Основные поля: `dataset`, `patient_id`, `age`, `age_raw`, `age_precision`, `age_group`, `eligible_by_reported_age`, `exact_age_available`, `modality`, `healthy_status`, `diagnosis`, `pathology_group`, `segmentation_available`, `structures`, `image_format`, `spacing_available`, `source`, `access_type`. Дополнительные evidence fields различают протокол набора и локальную проверку, partial/full download, диагноз и историю операций. Неизвестные числа и возраст — `null`, диагнозы — списки, без искусственного заполнения.

| healthy_status | Основание |
|---|---|
| `confirmed_healthy` | Явные критерии здоровой нормативной когорты или подтверждение клиницистом |
| `no_structural_cardiac_abnormality_reported` | Явная авторская Normal/structurally normal метка клинического случая; это не доказательство общего здоровья |
| `unknown` | Нет достаточной клинической информации; сюда относятся все Pediatric-CT-SEG |
| `pathological` | Явный порок, патологическое изменение или хирургическая коррекция в metadata |

Флаг Normal не перекрывает одновременно указанные патологию/операцию. Например, HVSMR `pat2` имеет Normal и Marfan: реестр сохраняет оба факта и не относит его к healthy controls. Отсутствие флага операции означает `not_reported`, а не доказанное отсутствие операции. `severity` — авторская Category, не автоматически медицинская шкала тяжести.

Для будущей бинарной модели нужны одновременно клиническая верификация, подтверждённый возраст и достаточное качество данных. `binary_classification_eligible` оценивает только доказательность clinical status; это **не готовый train split**. Консервативно structurally-normal clinical cases остаются вне confirmed healthy controls до уточнения критериев с клиницистом.

CHD68 README содержит 65 pathological и 3 явных Normal (`1040`, `1072`, `1101`). Статья 2019 сообщает 2 Normal; это расхождение версий, а не повод переписать локальную таблицу. Нормальные случаи CHD68 не являются автоматически нормативной здоровой возрастной группой.

В таблице CHD68 представлены ASD, AVSD, VSD, anomalous drainage (AD), ToF, pulmonary artery sling (PAS), PDA, coarctation (CA), common arterial trunk (CAT), pulmonary stenosis (PS), aortic arch anomalies (AAA), TGA, single ventricle (SV), pulmonary atresia (PuA). Это multi-label диагнозы, не 14 взаимоисключающих классов. Точные clinical definitions и updated labels должны входить в согласование будущего baseline.

## Совместимость разметок

| Структура | CHD68 / текущий ImageCHD namespace | HVSMR-2.0 | Atlas | Pediatric-CT-SEG |
|---|---|---|---|---|
| LV / RV / LA / RA | 1 / 2 / 3 / 4 | 1 / 2 / 3 / 4 | Бивентрикулярная геометрия; предсердий нет | Отдельных chamber labels нет |
| MYO | 5 | В whole-heart 8-label release отсутствует | Epicardium и endocardium как поверхности, не эквивалент voxel MYO | Нет |
| AO / PA | 6 / 7 | 5 / 6 | Valve patches; не полные сосуды | Нет |
| SVC / IVC | Включаются в RA по протоколу CHD68 | 7 / 8 отдельно | Нет эквивалентной отдельной разметки | Нет |
| Pulmonary veins | Включаются в LA по протоколу CHD68 | Сверять границы по published protocol/endpoints | Нет | Нет |
| Heart region | Union 7 labels не обязательно совпадает с OAR Heart | Union blood pool не включает MYO | Surface template не является whole-heart OAR | ROI `Heart`, отдельная органная область |

**Нельзя подать HVSMR mask в текущий CT pipeline без source-specific mapping:** MRI label 5 означает AO, а CT label 5 означает MYO; 6 и 7 тоже имеют разный смысл. Совпадение названий не гарантирует совпадения анатомических границ. HVSMR endpoints с required/optional vessel extents должны учитываться при оценке длин и surface distances. Перестановка меток не решает различие протоколов. Common atrium / single ventricle и изменённые связи требуют клинической трактовки; отсутствующий label не является диагнозом.

## CHD68: проверка существующей реконструкции

| Случай | Диагноз в README | CT shape | Header axes | Header spacing / units | Результат |
|---|---|---|---|---|---|
| `ct_1008` | AD | 512×512×206 | RAS | 1×1×1 / unknown | 7 meshes + slices + 3D preview |
| `ct_1072` | Normal | 512×512×382 | RAS | 1×1×1 / unknown | 7 meshes + slices + 3D preview |
| `ct_1102` | ASD, VSD, AD, PDA, PuA | 512×512×205 | RAS | 1×1×1 / unknown | 7 meshes + slices + 3D preview |

У всех проверенных масок labels 0–7, сетки image/mask совпадают, qform code 0, sform code 1. Orientation здесь — по NIfTI header; DICOM/landmark verification для этих случаев отсутствует. Сохраняются все connected components; обнаруженные non-manifold edges не исправляются автоматически. Например, у `ct_1072` MYO имеет 81 и PA 45 non-manifold edges. Это геометрический smoke-test, не клиническая валидация и не оценка ML.

В исходной статье typical spacing около 0.25×0.25×0.5 mm³. В скачанных файлах эти данные не подтверждены. Получить case-specific DICOM/physical geometry у авторов либо документированное соответствие image-info release; затем исправлять только в производной копии с provenance. До этого размеры и объёмы выражаются в координатах файла. Существующий preview сохраняет legacy caption ImageCHD; dataset identity определяется явным source manifest и именами CHD68 в audit summary.

Полный числовой результат — `chd68_smoke_summary.json`. Полный release не скачивался: helper прочитал final split part и ограниченный prefix z05; CT/mask в разных дисках проверены по original ZIP CRC. AppleDouble `._` entries не принимаются за исследования. В Git нет CT, NIfTI, ZIP, generated meshes и outputs.

## Pediatric Cardiac Shape Atlas: практическая возможность

Открытый `EDESatlas.mat` действительно загружается Python/SciPy. Struct содержит `mean` (34860), `coeff` (34860×100), `latent` (100), `score` (101×100), `explained`, `tsquared`. Mean объединяет ED и ES: **5810 соответствующих вершин на фазу**, topology — 11760 triangles. Восемь первых modes объясняют 79.6236% variance; orthonormality проверена численно. Mean ED/ES и modes 1, 8 при ±3 SD сформированы и визуально просмотрены. Результаты и ограничения — `atlas_summary.json`.

Template различает LV endocardium, RV septum/free wall, epicardium и valve patches. Это не whole-heart atlas всех камер/сосудов. Пример ED/ES model в upstream repository не заменяет исходные healthy CMR или clinical metadata. Индивидуальные ages, BSA/height и сведения для условной нормы локально не получены.

Для вектора **уже соответствующих и правильно выровненных** координат исследовательская формула: `z_k = ((x - mean) @ coeff_k) / sqrt(latent_k)`. Mode reconstruction: `mean + z_k * sqrt(latent_k) * coeff_k`. Она не разрешает следующие проблемы:

- Marching-cubes meshes имеют произвольное число/порядок вершин. Требуется fitting к гомологичному бивентрикулярному template, landmarks и оценка ошибки соответствия.
- Поставляемая PCA совместная ED+ES. Единственный CT scan не содержит обе фазы. Нельзя дублировать CT как ED и ES. Нужны paired gated phases либо отдельно разработанный и проверенный atlas одной фазы.
- Scale, RAS/LPS, reflection policy и alignment должны воспроизводить формирование атласа. В upstream alignment Procrustes transform вычисляет scaling, но применяется rotation/translation; слепая generic normalization изменит смысл modes.
- `projectOntoAtlas.m` в скачанном code release использует adult `UKBRVLV.h5` и `unalignedPts.mat`, которых нет. README описывает более широкий workflow, чем готовая поставка. Нельзя объявить проекцию нашего mesh работающей.
- Возрастной диапазон и преимущественно подростковая нормативная когорта требуют age/BSA/sex conditioning и проверки представленности младших групп. Агрегированный atlas не позволяет точно удалить пациентов старше 17 без individual metadata.

Будущая гипотеза СППР: сегментация → template fitting → проверка phase/scale/correspondence → shape scores и residual map → экспертная оценка. Residual после PCA и «аномалия» — разные величины: residual отражает также registration error, segmentation error, sampling и формы вне span. Shape mode Z-score не равен доказанной вероятности порока; regional displacement не является regional Z-score без локальной дисперсии и калибровки. Для single ventricle и иной topology fitting к нормальному template может быть неприменим; нужен отказ от оценки, а не принудительное совмещение. CT/MRI intensities на этом пути не объединяются.

## Pediatric-CT-SEG: полезность и DICOM gate

Возраст, scanner и series доступны в digest; полных диагнозов нет. Протоколы трёх CT scanners неоднородны, contrast и reconstruction нужно выделять отдельно. 62 исследования размечались при native thickness, 297 были reformatted до 2 mm; возможны artifacts на крайних slices. Статья авторов сообщает 356 Heart contours, но фактический ROI census текущего release ещё не выполнен. Не переносить опубликованный count на любой filtered subset.

Локально у `Pediatric-CT-SEG-E03568A6` проверены CT DICOM `PatientAge=004Y`, PixelSpacing 0.546875×0.546875, SliceThickness 2, IOP/IPP, slope 1 / intercept −1000; RTSTRUCT действительно содержит `Heart` и ссылается на выбранную CT series. Скачан один CT slice, **не полный стек**. Это подтверждение доступности и tags, не завершённый DICOM → NIfTI/RTSTRUCT rasterization pipeline. SliceThickness не подменяет spacing по positions.

Перед Stage A: получить несколько полных CT series; сортировать по projected IPP, проверить regular spacing/orientation, SOP/Series/FrameOfReference links, rasterize ROI на исходную сетку, сохранять преобразование LPS → RAS и CT HU scaling, сравнить contours/overlay в независимом viewer. Отдельно записать scan coverage, Heart completeness, contrast, scanner и edge artifacts. Без Heart ROI случай пригоден для representation learning, но не для supervised Heart segmentation. Пересечение age filter и verified complete Heart masks пока неизвестно.

## HVSMR: подвыборка и сложная анатомия

`metadata/pediatric/hvsmr_2_17.csv` содержит **36 пациентов** с age, severity, diagnoses, previous surgery, available structures по dataset protocol, stress tags и artifacts. Из них 23 severe, 5 moderate, 8 mild; 15 / 6 / 15 по рабочим возрастным группам. `available_structures` означает протокол release, не проверенное наличие каждой структуры каждого пациента. Числа flags могут пересекаться:

| Stress tag | Число в подвыборке |
|---|---:|
| SingleVentricle | 1 |
| DILV (отдельный родственный диагноз, не переименован автоматически) | 3 |
| DORV | 13 |
| DLoopTGA / LLoopTGA | 6 / 2 |
| CommonAtrium | 5 |
| PAAtresiaOrMPAStump (комбинированное исходное поле, не всегда чистая atresia) | 6 |
| Glenn / Fontan | 19 / 7 |
| Heterotaxy | 8 |
| SuperoinferiorVentricles | 2 |

Clinical CSV не содержит `pat59`, technical содержит. Запись сохранена с unknown age/status и не включена в subset. Не считать группу 5–18 из статьи точным числом пациентов 2–17. Среди возрастной подвыборки Normal отмечен у шести, но один также Marfan; остальные пять — reported structurally normal, а не автоматически здоровые добровольцы. MRI geometry spacing пока описана авторами, но локально NIfTI headers не проверены: imaging download отложен до выбора stress-test cases. orig/cropped/cropped_norm — варианты тех же пациентов, не дополнительные независимые случаи.

## Предлагаемый следующий ML-эксперимент

Ни один из следующих этапов автоматически не запускать.

**A — pediatric CT representation / первый baseline.** Из 327 age-eligible Pediatric-CT-SEG получить verified Heart subset. Первый эксперимент — компактная CT-only модель локализации и binary Heart segmentation, например 3D U-Net. До выбора архитектуры и объёма обучения: готовый DICOM adapter, пройдённый QA, patient split и ресурсный бюджет. Разделить пациентов до resampling/augmentation; stratify по age и scanner, зафиксировать seed. При достаточном числе случаев — 70/15/15 train/validation/test и дополнительная проверка scanner domain; при малой группе — patient-level cross-validation без слияния результатов с holdout. Нормализацию оценивать только по train. Метрики: Dice, recall локализации, HD95/ASSD в mm только при подтверждённой physical geometry; результаты по возрасту/scanner/contrast и failures. Маску области сердца не выдавать за seven-structure segmentation.

**B — cardiac structure segmentation.** CHD68 + подходящие CT НИИ: сначала age metadata, diagnosis, scale/orientation, label protocol, deduplication. Общий target namespace и source-specific maps. Начать с небольшой проверенной когорты и одного baseline с per-structure Dice/surface distances; патологические и хирургические варианты оценивать отдельно. ImageCHD/CHD68 repeats группировать до split, не использовать их как external validation друг друга. Если suitable CHD68 2–17 мало, основной baseline возраста 2–17 должен опираться на НИИ; infant-heavy CHD68 можно предложить как отдельное auxiliary pretraining с явным domain/age shift.

**C — normal/pathological reasoning.** Достоверные healthy pediatric controls + клинически подтверждённые патологические cases. MRI-derived geometry допустима; MRI+CT voxel intensities требуют отдельной domain adaptation methodology. Healthy CMR против CHD CT создаёт confounding modality; нельзя интерпретировать точность как распознавание болезни. Нужны modality-balanced или geometry-only cohorts, проверка age/BSA/scanner/phase confounders, calibration, uncertainty и раздельный внешней test. Shape atlas — исследовательский reference; не готовый ground truth диагнозов и не заменяет здоровую контрольную image cohort.

**D — complex anatomy.** Отобрать stress-tag cases из 36 HVSMR, скачать images/seg/endpoints, проверить headers и source-specific label mapping. Оценить topology preservation, сохранение малых компонент, vessel extent и возможность отказа при неподходящем template. MRI не добавлять в CT intensity training автоматически.

## Что запросить у НИИ и держателей данных

У НИИ: обезличенные CT DICOM и полные physical tags для пациентов 2–17; age at acquisition с точностью/единицей, sex, height/weight/BSA, диагнозы с клиническим подтверждением, operative history/stage, phase/gating, contrast timing, scanner/protocol, reconstruction kernel, repeat-study linkage. Просить confirmed healthy/normal criterion или `no structural abnormality reported` отдельно: отсутствие CHD в файле не достаточно. Дополнительно экспертные маски LV/RV/LA/RA/MYO/AO/PA с протоколом границ и coverage/QC, сведения о редкой/послеоперационной анатомии, допустимые условия исследования/публикации и независимую evaluation cohort. Не назначать детям CT только ради контрольной выборки; использовать допустимые клинические исследования и нормативную MRI-геометрию.

Через CAP запросить DUA и age/anthropometry/phase-linked normative models/CMR для полноценного 2–17 reference и younger-age conditioning. Авторам normative CMR study — возможность получить индивидуальные измерения и изображения/контуры с возрастом и BSA. Авторам CHD68 — per-case age, physical spacing/orientation/original DICOM и объяснение overlap/version differences. Авторам HVSMR — clinical entry `pat59` и трактовку Normal/Marfan и combined PAAtresiaOrMPAStump, при необходимости.

## Git и воспроизведение

Основной checkout на `feature/mesh-smoothing` имел пользовательские изменения README и untracked files. Они сохранены без stash, reset или переноса. Для выполнения требования чистой ветки создан отдельный managed worktree от **bcd821e**, наиболее актуального сохранённого состояния с geometry/viewer/smoothing проверками; local main был старее. В новом чистом checkout создана только `feature/pediatric-datasets`. Основной checkout остаётся на прежней ветке; merge в main не выполняется.

Базовые проверки CI и новые tests: `python -m pytest tests --ignore=tests/test_viewer.py --ignore=tests/test_interactive_gui.py -q` — 46 passed на момент аудита. Существующие NumPy/skimage deprecation warnings сохранены как зависимость окружения, не ошибки данных.

Пример команд в новом checkout (подставить установленный project Python и внешний data root):

```powershell
python scripts/fetch_pediatric_metadata.py --data D:/codexProjects/3d-hearts/data/pediatric_audit --atlas
python scripts/build_pediatric_registry.py --data D:/codexProjects/3d-hearts/data/pediatric_audit --out metadata/pediatric
python scripts/fetch_chd68_sample.py --data D:/codexProjects/3d-hearts/data/pediatric_audit/chd68 --count 3
python scripts/inspect_pediatric_atlas.py --atlas-root D:/codexProjects/3d-hearts/data/pediatric_audit/atlas --summary metadata/pediatric/atlas_summary.json --output D:/codexProjects/3d-hearts/data/pediatric_audit/atlas_outputs
```

Случаи CHD68: `python -m heart3d build --ct <external CT> --mask <external mask> --out <new external output directory> --preview`. Existing outputs не перезаписывать. DICOM helper требует optional `requirements-pediatric.txt`: `python scripts/fetch_tcia_sample.py --data <external data root>`. После build обновить commit-safe summaries: `python scripts/summarize_pediatric_checks.py --data <external data root> --metadata metadata/pediatric`. Сравнение overlap воспроизводится отдельным `audit_pediatric_overlap.py` по двум inventory. Реестр сначала rebuild, затем enrich summaries; age summaries при этом не меняются.

## Готовность этапа

Выполнены аудит источников, стратегия, единый evidence-preserving registry, 36 HVSMR cases, изучение открытого atlas и mean/PCA, оценка Pediatric CT, 3 CHD68 reconstruction smoke-tests, Git isolation и план ML. Не выполнялись обучение, DUA access, healthy classifier и CT/MRI intensity pooling.

**Полная готовность данных для обучения ещё не достигнута:** неизвестен suitable CHD68 2–17 count и physical scale его release; нет индивидуальных normative controls, полного TCIA DICOM converter/ROI census, MRI header QA и клинической calibration atlas. Нельзя считать эти ограничения закрытыми только по успешному построению 3D. Следующее короткое действие — несколько полных Pediatric-CT-SEG series с RTSTRUCT→mask QA и получение case-specific age/geometry CHD68, затем отдельное согласование конкретного baseline и бюджета обучения.
