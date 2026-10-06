# Pediatric Heart segmentation: подготовка baseline

Дата проверки: 2026-10-06. Ветка `feature/pediatric-heart-segmentation`, создана от
`feature/pediatric-datasets` (`66c5804`). Основной checkout с пользовательскими
изменениями не переключался. **Обучение не запускалось.** Здесь зафиксирован
подготовительный этап; результаты модели, test metrics и predictions отсутствуют.

**Подготовительный отчёт перед обучением.** Census завершён для **327/327**
пациентов. У всех есть определение Heart ROI, у 324 — непустые контуры;
322 проходят metadata/reference gate, 2 имеют неподтверждённые CT SOP references,
3 имеют пустые Heart contours. Полные CT + оригинальные RTSTRUCT получены и
независимо конвертированы для **9/9 пилотных исследований**; все девять просмотрены
на native mosaics и в NiiVue. Три исключены из-за неполного scan coverage,
ещё один требует coverage review, пять — проверки объёма OAR-аннотации.
Готовая training cohort и frozen split отсутствуют: **0 полностью одобренных
для обучения случаев**. Это результат проверки gates, а не отрицательная оценка
качества всех исходных контуров. Предложение следующего эксперимента: 60 пациентов
после полного QA, split 42/9/9, собственная 2.5D U-Net на CPU. Работа остановлена
перед выбором полной когорты и обучением согласно заданному порядку этапа.
Точка продолжения: [pediatric_heart_segmentation_resume.md](pediatric_heart_segmentation_resume.md).

## 1. Цель

Исследовать устойчивость собственной бинарной сегментации **исходного Heart ROI**
на детских CT разного возраста, scanner/protocol и coverage. Первый target —
Heart organ-at-risk (OAR), определённый экспертами источника. Это не разметка
камер, миокарда или сосудов и не clinical whole-heart diagnosis. Цепочка этапа:
полная DICOM CT series → HU volume с проверенной геометрией → RTSTRUCT Heart
→ маска на исходной сетке → техническая и визуальная QA → план эксперимента.

## 2. Источник данных

[TCIA Pediatric-CT-SEG](https://www.cancerimagingarchive.net/collection/pediatric-ct-seg/),
DOI [10.7937/TCIA.X0H0-1706](https://doi.org/10.7937/TCIA.X0H0-1706).
Возрастная выборка из предыдущего аудита: 327 пациентов с `2 <= reported age <= 17`:
143 — группа 2–5, 105 — 6–11, 79 — 12–17. Фактически сообщённые возраста —
**2–16 лет**, случаев 17 лет нет. Это рабочие группы; DICOM `NNNY`
указывает возраст с точностью до сообщённых лет, а не точную дату рождения.

В [публикации источника](https://pmc.ncbi.nlm.nih.gov/articles/PMC9090951/)
описаны CT по клиническим показаниям, отсутствие диагнозов и экспертные OAR-контуры.
Поэтому `healthy_status=unknown` для всех случаев. Отсутствие CHD-метки ничего
не говорит о нормальности сердца. Авторы сообщают о разных scan ranges,
контрастировании, шуме и артефактах крайних срезов после исходного reformating.
Опубликованное количество Heart contours не заменяет наш case-level census.

## 3. DICOM QA

`heart3d/dicom/ct.py` поддерживает classic single-frame CT Image Storage.
Enhanced CT, смешанные серии, нерегулярные сетки и gantry shear требуют отдельного
adapter; они не исправляются автоматически. Проверяются PatientID/PatientAge,
SeriesInstanceUID, FrameOfReferenceUID, SOP UID uniqueness, размеры, IOP/IPP,
PixelSpacing, RescaleSlope/Intercept и допустимый RescaleType.

Порядок срезов: `dot(IPP, cross(IOP[:3], IOP[3:]))`. Z step определяется медианой
соседних проекций; SliceThickness сохраняется только как nominal metadata.
Допуск регулярности — `max(0.02 mm, 1% step)`, допустимый residual сетки — 0.05 mm.
Это технические допуски данного adapter. Совпадение с live TCIA SOP inventory
проверяет полноту **публикации**, но не доказывает отсутствие срезов в исходной
неопубликованной acquisition. InstanceNumber не определяет порядок.

Внутренняя CT grid: массив `[column,row,slice]`, координаты **LPS**;
оси affine: `IOP[:3]*PixelSpacing[1]`, `IOP[3:]*PixelSpacing[0]`, `normal*z_step`.
Origin — IPP первого пространственно отсортированного среза, центр voxel 0.
HU = stored pixel × RescaleSlope + RescaleIntercept, отдельно для каждого среза.
Без подтверждённых tags конвертация останавливается.

NIfTI имеет **RAS** affine: `diag(-1,-1,1,1) @ affine_LPS`, units mm,
qform=sform. Массив не переворачивается при смене системы координат.
Такой перевод предотвращает прежнюю неопределённость анатомической ориентации
ImageCHD. Правило следует [DICOM Image Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html).

## 4. RTSTRUCT rasterization

`heart3d/dicom/rtstruct.py`: точное имя `Heart` без учёта регистра и крайних
пробелов, ровно один ROI. Проверяются ROI FrameOfReference, referenced CT series,
global SOP references и уникальный CT SOP reference каждого контура.
Отсутствующие ссылки не заменяются ближайшим срезом.

ContourData LPS переводится inverse CT affine в voxel coordinates. Срез выбирается
по SOP, а не округлением Z. Максимальный plane residual — 0.05 mm; выход за FOV,
некорректные точки и пустая маска требуют review. Полигон заполняется по центрам
voxel. `CLOSED_PLANAR` объединяются, `CLOSEDPLANAR_XOR` комбинируются XOR;
смешанные типы отвергаются. **Никакой межсрезовой interpolation первичного GT.**
Основание: [DICOM ROI Contour Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.8.6.html).

CT и GT имеют одинаковые shape, affine, spacing и origin. Исходные DICOM
не изменяются. Для пилота используется оригинальный полный RTSTRUCT.
Heart-only extracts census являются отдельными производными файлами с новыми
SOP/Series UID и source receipt; они не выдаются за полные RTSTRUCT.

## 5. Критерии включения и исключения

Включение: допустимый сообщённый возраст, одна полная CT series, HU/geometry gate,
валидный Heart и все ссылки, техническая visual alignment QA, достаточное coverage,
проверенный объём исходной аннотации для согласованного Heart OAR target.

Исключение/отложенный review: missing/empty Heart; unmatched SOP/frame/series;
нерегулярные или неполные grid; открытые/смешанные contours; маска вне FOV;
Heart на scan boundary или в пределах одного неразмеченного CT slice от неё;
внутренние пропуски контуров; неоднозначный annotation extent. Правило одного
среза — технический сигнал для review, не медицинский критерий полноты сердца.
Автоматического исправления GT нет. Техническая корректность rasterization,
полнота scan coverage и анатомическая полнота разметки — три разные проверки.

## 6. Dataset census и фактически полученные данные

Машинный реестр: [heart_segmentation_candidates.json](../metadata/pediatric/heart_segmentation_candidates.json).
Результаты полных studies: [heart_segmentation_pilot_qa.json](../metadata/pediatric/heart_segmentation_pilot_qa.json).
Visual review: [heart_segmentation_visual_reviews.json](../metadata/pediatric/heart_segmentation_visual_reviews.json).
Итоговые количества находятся в поле `summary` реестра.

Пилот: девять полных CT + оригинальных RTSTRUCT. Выбор — median published CT size
в каждой представленной age/scanner cell плюс ранее проверенный E03568A6,
без выбора по качеству target. Есть все три модели scanner и возрастные группы.

Census для всех 327 пациентов использует live SOP inventory, одну полную CT probe
и **полный Heart contour item**. HTTP prefix читается последовательно и ограничен
64 MiB; может включать другие предшествующие ROI и небольшой read-ahead.
Item delimiters/length проверяются, неполный item не принимается.
Полные CT headers/pixels и оригинальные RTSTRUCT проверены для девяти studies.
Дополнительно скачаны два полных оригинальных RTSTRUCT для проверки reference
failures: всего **11 original RTSTRUCT**, из них 9 с полным CT. Для остальных
318 пациентов full-stack spacing, rasterization, coverage и suitability остаются
`null`/pending, а не автоматически подтверждёнными. Это census metadata,
**не завершённая preparation всей возрастной когорты**.

| Проверка | Результат |
|---|---:|
| Возрастной metadata census | 327/327 |
| Определение Heart / непустые Heart contours | 327 / 324 |
| Metadata/reference gate | 322 pass, 2 review, 3 empty |
| Scanner: LightSpeed VCT / SOMATOM Definition AS+ / Revolution CT | 134 / 128 / 65 |
| ContrastBolusAgent указан / не указан | 301 / 26 |
| Полный CT + RT, HU/geometry/rasterization gate | 9/9 |
| Native mosaics + независимый NiiVue | 9/9 |
| Pilot: неполное coverage / coverage review / annotation scope review | 3 / 1 / 5 |
| Полностью одобрены для обучения | 0 |

ContrastBolusAgent — evidence введения контраста; пустой tag оставляет статус
unknown. Scanner metadata здесь означают модель аппарата, а не независимые центры.

CT всего возрастного inventory: 54,658,909,982 bytes; RT: 7,286,328,758 bytes.
Свободного места на D недостаточно для всех raw CT. Пилот, raw originals,
архивы, derived volumes, masks и screenshots находятся вне Git, в
`D:/codexProjects/3d-hearts/data/pediatric_ct_heart`.

## 7. Предложение cohort и split

Первый эксперимент: **60 полностью QA-approved пациентов** — по 20 в каждой
рабочей возрастной группе. **42 train / 9 validation / 9 test**, patient-level,
seed `20261006`. Точные age/scanner квоты заданы в
[pediatric_heart_preparation_v1.json](../configs/pediatric_heart_preparation_v1.json).
Дополнительно балансировать contrast evidence и published reformatting artifacts
в пределах доступных case metadata. Нельзя объявлять absent ContrastBolusAgent
доказательством non-contrast. Revolution CT не представлен в старшей age group;
это реальное ограничение age/scanner независимости.

60 — точный **предлагаемый**, не уже подтверждённый размер training cohort.
Необходимы full-stack/coverage/annotation extent QA кандидатов и замены в той же
stratum при исключении. После этой проверки следует создать и зафиксировать
`configs/splits/pediatric_ct_heart_v1.json`; сейчас frozen split отсутствует,
чтобы не выдавать metadata candidates за готовый dataset. Уже inspected cases
исключены из будущего test: они уже использованы для разработки loader и QA.
Квоты — цели; если full coverage не даст нужного количества в stratum, пересмотреть
их до фиксации split и документировать причину. Три пилотных Revolution studies
с ограниченным coverage не доказывают непригодность всех 65 случаев этого scanner.

## 8. Preprocessing

Пока отсутствуют fitted training statistics. Диапазон -150..250 HU в screenshots —
только окно просмотра, не training clipping. После frozen split анализировать
train HU distributions, padding, body FOV, anisotropy и размеры Heart; выбрать
clipping/normalization только по train, зафиксировать их один раз для val/test.

Гипотеза для CPU baseline: axial full-FOV fit/pad до 256×256, без GT-guided crop;
кандидат Z grid 2 mm и пять slices с physical offsets [-4,-2,0,2,4] mm.
Окончательные spacing/FOV проверить по train до запуска. Image interpolation —
linear, mask — nearest-neighbor. Сохранять inverse spatial transform и original
GT grid; оценивать восстановленную prediction на original geometry.

## 9. Предлагаемая архитектура

Собственная **2.5D U-Net** на PyTorch с нуля: 5 input slices → binary mask центра.
Число каналов 16/32/64/128, три max-pool; в каждом block две Conv2d 3×3 +
GroupNorm(8) + ReLU. Decoder: bilinear upsample, concatenate skip, такой же block.
Выход Conv2d 16→1, logits. Расчётный размер при bias-enabled convolutions:
488,993 parameters (1.87 MiB FP32 weights). Код модели пока не реализован.

Причина выбора: на текущем компьютере не подтверждён CUDA device. 3D U-Net
целесообразен отдельным экспериментом при наличии поддерживаемого GPU с достаточной
памятью; чужая готовая segmentation network и nnU-Net не подставляются.

## 10. Loss — план сравнения

Три сопоставимых запусках с одним split и бюджетом: BCEWithLogits, soft Dice,
BCE + Dice. Для p=sigmoid(logits), y∈{0,1}:

`BCE = -mean[y*log(p)+(1-y)*log(1-p)]`

`DiceLoss = 1-(2*sum(p*y)+epsilon)/(sum(p)+sum(y)+epsilon)`

`Combined = BCE + DiceLoss`.

Реализовать Dice внутри проекта; вычислять per sample, затем mean. Отдельно
определить обработку empty-target slices, чтобы отрицательные slices не исчезали
из эксперимента. Формулы здесь — предложение, loss code ещё не выполнялся.

## 11. Augmentation — план

Небольшие rotations/scaling синхронно для CT и mask, image linear/mask nearest;
малый intensity shift/noise. Одинаковое spatial преобразование всех пяти input
slices. Без flips. Параметры зафиксировать после train analysis; validation/test
не augment. Не обрезать систематически границу Heart.

## 12. Training setup — следующий этап

После data gate и отдельного согласования: собственный `heart3d.ml.train`, CPU,
batch 2 (fallback 1), workers 0, deterministic seed; 30 epochs — начальный
план, Adam, learning rate/scheduler по development validation. Все три loss
варианта сравнить при одинаковом training budget. Сохранять best-validation/last
checkpoints, history, config, environment, split hash. Test не использовать для
подбора clipping, learning rate, threshold или postprocessing.

Многочасовой запуск не сделан. До измерения скорости PyTorch на одном коротком
train-only benchmark нельзя надёжно обещать duration целого эксперимента.

## 13. Hardware и budget

Windows, AMD Ryzen 5 4500, 6 cores / 12 threads, 16 GB RAM. Win32 сообщает
AMD Radeon RX 580 2048SP и около 4 GB AdapterRAM; это не проверка фактической
VRAM или работоспособности PyTorch backend. `nvidia-smi` и CUDA GPU не найдены,
PyTorch в использованном environment отсутствует. GPU training не предполагается.

Полный pilot CT array — float32, mask — uint8; конкретные MiB и время конвертации
есть в pilot QA. Самый крупный выбранный stack содержит 665 slices:
665 MiB CT + 166.25 MiB mask. Независимые reader/labels/stencil требуют временных
копий. Практический preparation budget: один volume одновременно, 3–5 GB RAM
с запасом, отдельно до 8 lightweight network workers. Training 2.5D: ориентир
4–6 GB CPU RAM при batch 2 и дисковом/lazy cache; это оценка, не измеренное обучение.

60 raw CT могут потребовать около 10–15 GB до archive duplication и prepared
cache, точный объём надо суммировать по выбранным UID. При проверке свободно
**28.07 GB на D**; этого недостаточно для всей возрастной raw cohort.
Перед cohort download проверить место, не удалять существующие
данные автоматически. Для всей когорты raw CT + RT + derived/cache рекомендуется
отдельное хранилище с ≥150 GB свободного пространства; фактический бюджет зависит
от retention архивов и формата cache. Полные float32 volumes не cache в RAM.

## 14. Метрики — будущая оценка

Dice, IoU, precision, recall per patient на original CT grid; aggregate и
разбивка age/scanner/contrast evidence/spacing/coverage. HD95 и ASSD в mm только
для cases с подтверждённой geometry; явно определить surface sampling, directed
aggregation и empty-mask convention. Никаких выдуманных test scores сейчас нет.

## 15. Результаты подготовительного этапа

Синтетические проверки включают oblique/anisotropic landmarks, HU, RAS/LPS,
unordered slices, duplicates/missing inventory, irregular spacing, ROI mismatch,
XOR hole, off-plane/out-of-FOV contours, implicit/explicit stream и truncation.
**73 tests passed** (GUI tests не запускались; существующие NumPy/scikit-image
deprecation warnings сохранены). Независимое чтение девяти реальных CT:
SimpleITK ImageSeriesReader с собственным GDCM discovery/sorting; проверены affine,
dimensions и все HU pixels. Во всех девяти max HU difference = 0.
Независимая rasterization: vtkPolyDataToImageStencil;
различия допускаются только на voxel boundary, порог Dice ≥0.99. Фактический
Dice двух rasterizers на пилоте: **0.999994727–1.0**, вне границы различий нет.
Максимальный contour-to-slice residual: **0.005 mm**.
Это **agreement двух способов конвертации**, не ML segmentation metric.

## 16. Failure cases и незавершённые gates

BEC712BF, 4C1A38AE, 813E523C: Heart на последних CT slices 164/164, 184/184,
188/188 соответственно, визуально superior coverage недостаточно; исключены
из whole Heart cohort. Masks не достраивались.
E03568A6: Heart заканчивается за один CT slice (2 mm) до верхней границы.
Контакт с voxel face отсутствует, но полное coverage не подтверждено; требуется
review coverage и исходной аннотации.
Пять остальных пилотных cases имеют плоскую cranial границу исходного Heart ROI.
Она совпадает с source contours; нужно согласовать объём OAR target, а не
автоматически объявлять это полной анатомической разметкой или исправлять.
Reference failures подтверждены по **полным оригинальным RTSTRUCT**:
272B6C5D — 211 global references и **17 Heart references** отсутствуют в
published CT inventory; 34ECBB32 — 147 global и **8 Heart references**.
Heart item и global references совпадают с census extracts; это не ошибка
stream extraction. Причина несоответствия на стороне опубликованной пары требует
выяснения; не подменять SOP ближайшими slices. Отчёт:
[heart_segmentation_reference_review.json](../metadata/pediatric/heart_segmentation_reference_review.json).
Три пустых Heart: наличие ROI definition не означает наличие segmentation.

## 17. Ground Truth vs Prediction

Сейчас доступен независимый QA viewer **CT + GT Heart**, axial и multiplanar,
toggle overlay. `scripts/view_pediatric_ct_qa.py` слушает только loopback и
выдаёт только allowlisted prepared files. В `--reviews` можно передать сохранённые
review statuses. NiiVue — независимый NIfTI viewer, визуальная техническая проверка
не является clinical annotation sign-off. Нужен интернет для pinned NiiVue
0.69.0 CDN. Нативный viewer проекта не переделан. Prediction/Comparison режимы
добавлять после реального baseline; пока prediction отсутствует.

## 18. 3D comparison

GT vs predicted surfaces, volume, area, components, distances и topology warnings
относятся к следующему этапу после training/inference. Текущий multiclass mesh
pipeline имеет label 1=LV; binary Heart нельзя выдавать за LV. При интеграции
нужен явный Heart label mapping. Сейчас сравнительных surfaces нет.

## 19. Ограничения

OAR scope отличается от специализированного cardiac whole-heart protocol;
coverage и clinical status неоднородны; reported age имеет ограниченную точность;
одна институция, три scanner models, age/scanner confounding, unknown contrast
при пустых tags. Metadata-level pass не заменяет full volume QA. Независимая
техническая QA не является медицинской переоценкой экспертных контуров.

## 20. Воспроизведение и следующий этап

Python dependencies: `requirements-lock.txt` + pinned `requirements-pediatric.txt`.
Raw data path задаётся вне Git. Пример PowerShell из нового worktree:

```powershell
python -m pip install -r requirements-lock.txt -r requirements-pediatric.txt
$dataRoot = 'D:/codexProjects/3d-hearts/data/pediatric_ct_heart'
python scripts/fetch_tcia_inventory.py --out "$dataRoot/tcia_series_v1.json"
python scripts/fetch_pediatric_ct_pilot.py --registry metadata/pediatric/registry.json --series "$dataRoot/tcia_series_v1.json" --data $dataRoot
python scripts/prepare_pediatric_ct_pilot.py --data $dataRoot
python scripts/census_pediatric_ct_heart.py --registry metadata/pediatric/registry.json --series "$dataRoot/tcia_series_v1.json" --data $dataRoot --out "$dataRoot/census.json" --workers 8
python scripts/audit_pediatric_rt_references.py --data $dataRoot --out metadata/pediatric/heart_segmentation_reference_review.json
python scripts/export_pediatric_heart_census.py --data $dataRoot --reviews metadata/pediatric/heart_segmentation_visual_reviews.json --out metadata/pediatric/heart_segmentation_candidates.json
python scripts/view_pediatric_ct_qa.py --data $dataRoot --reviews metadata/pediatric/heart_segmentation_visual_reviews.json
python -m pytest tests --ignore=tests/test_viewer.py --ignore=tests/test_interactive_gui.py -q
```

Download cache содержит SHA-256 receipts; ZIP CRC и отсутствие перезаписи
отличающихся DICOM проверяются. Inventory snapshot не перезаписывается; при
обновлении source использовать новый versioned путь и сравнить case metadata.
Оригинальные данные не входят в Git. Для текущего запуска использован существующий
Python 3.14 environment и отдельная external installation pydicom 3.0.2.

Следующий конкретный шаг: review annotation extent и coverage, выбрать 60
кандидатов, провести полный gate для каждого, зафиксировать patient split и
train-only preprocessing, реализовать предложенную модель и короткий resource
benchmark. Затем отдельное согласование многочасового training. CT/MRI joint
training, диагнозы, multiclass segmentation и СППР на этом этапе отсутствуют.
