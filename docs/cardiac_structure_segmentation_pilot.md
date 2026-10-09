# Аудит многоклассовой разметки и клинический пилот

Дата: 09.10.2026. Ветка: feature/pediatric-heart-segmentation.
Приоритет: КТ → сегментация → ручная коррекция → проверяемая 3D-модель.
Astra Linux, VR и печать отложены. Frozen Heart baseline v1 не изменяется.

## Выполненный аудит CHD68

Полный публичный release скачан на E: из Kaggle, на который ссылаются авторы.
Архив: 5.416 GB; все 136 image/label entries проверены по исходным ZIP CRC32
и SHA-256. Существовавшие копии с D: сверены по SHA-256 до/после копирования на E:;
оригиналы сохранены. При извлечении архива совпавшие файлы не перезаписывались.

68/68 пар прошли проверку сетки, конечных значений и целочисленных меток.
Все семь cardiac labels присутствуют в каждой паре; у 13 случаев есть
дополнительные несердечные метки. Исходные маски сохраняются без изменения.
[Реестр](../metadata/pediatric/public_chd_multiclass_audit_v1.json).

Все 68 release имеют spacing 1×1×1 и неопределённые пространственные единицы.
Значения изображений в совокупности находятся в диапазоне 0–4095;
HU не подтверждены. Эти числовые координаты нельзя автоматически считать мм,
а intensities — переносить в HU preprocessing бинарного baseline.

63 пары пересекаются с ImageCHD по archive name/size/CRC.
Для трёх доступных ImageCHD пар — ct_1001, ct_1002, ct_1004 — теперь
подтверждена полная идентичность image и label по SHA-256.
Остальные 60 совпадений остаются доказательством уровня archive CRC,
а не независимой SHA-проверкой. Перед split все потенциальные повторы
консервативно объединяются; эти наборы не используются как независимые train/test.

Из таблицы авторов получены диагнозные коды всех 68 случаев:
65 имеют патологические коды, у трёх явно указано Normal.
Последние обозначены как no_structural_cardiac_abnormality_reported,
а не как подтверждённые healthy controls.
В этом этапе диагностический классификатор не разрабатывается.

Источники:
[CHD68 authors](https://github.com/XiaoweiXu/Whole-heart-and-great-vessel-segmentation-of-chd_segmentation),
[publication](https://www.nature.com/articles/s41598-023-34013-1),
[ImageCHD authors](https://github.com/XiaoweiXu/ImageCHD-A-3D-Computed-Tomography-Image-Dataset-for-Classification-of-Congenital-Heart-Disease).

## Найденная metadata и возрастная несогласованность

Таблица imagechd_dataset_image_info.xlsx содержит 110 строк с данными
для расчёта возраста, PixelSpacing, calculate_z_thick и scanner model.
Исходная таблица находится на E:, даты в Git не экспортируются.

Правило idx + 1000 сопоставляет множество индексов с множеством всех
110 ImageCHD case IDs. Это сильный технический признак соответствия,
но авторское подтверждение связи и неизменности сетки ещё не получено.
Для 63 CHD68 случаев найдены такие кандидаты metadata; для пяти — нет.
58 кандидатов имеют возраст 0–17 полных лет, пять — 29–47 лет.

Последние расходятся с диапазоном 1 месяц–21 год в статье CHD68.
Возможны различие опубликованной и выпущенной когорт, иная связь индексов
или особенности дат. Причина не установлена. В реестре age остаётся null,
age_verified=false; кандидаты вынесены в отдельное поле.
58 — не число подтверждённых пригодных детских пациентов.

Source PixelSpacing и calculate_z_thick пока не подставляются в NIfTI:
не проверено, сохранена ли исходная сетка при подготовке release.
Нужны подтверждение авторов/исходная геометрия и описание преобразований.
[Черновик запроса авторам](public_chd_metadata_request.md) подготовлен,
письмо не отправлялось.

Возраст сам по себе не является обязательным входом U-Net. Технический
эксперимент в voxel coordinates возможен с явно описанными ограничениями,
но не доказывает результат на подтверждённой когорте 0–17 или на новорождённых.

## Протокол структур и модель

[Предлагаемая конфигурация](../configs/cardiac_structure_annotation_v1.json):
фон + LV/RV/LA/RA/MYO/AO/PA. В CHD68 venae cavae входят в RA,
pulmonary veins — в LA; протокол нужно согласовать с локальной разметкой.
Отсутствие структуры не является автоматическим диагнозом.
Не заставлять сложную анатомию соответствовать нормальным четырём камерам.

Собственная 2.5D U-Net расширена до восьми выходов:
5 каналов, encoder 16→32→64→128, GroupNorm/ReLU, bilinear decoder.
489112 параметров. Бинарный default остаётся одним выходом, старый checkpoint
и tests совместимы. Это адаптация стандартной архитектуры, не новая архитектура.

Реализованы CE, собственная multiclass Soft Dice и CE+Dice.
Dice усредняется по foreground classes после суммирования по B,H,W.
Для класса c: (2 sum(p*g)+epsilon)/(sum(p)+sum(g)+epsilon).
ignore_index=255 исключается из обоих слагаемых; all-ignore даёт нулевой
дифференцируемый loss. Несердечные метки не переименовываются молча в камеры.

## Ограниченный learning sanity check

На одном development-примере ct_1008 выполнены 12 CPU batches с нуля.
Windows 11, Ryzen 5 4500 (6 cores/12 logical CPUs), RAM 17067737088 bytes,
Python 3.14.4, PyTorch 2.14.1+cpu, 4 torch threads; CUDA недоступна.
Проверка использует одну выбранную обучающую плоскость повторно:
контекст по индексам [-2,-1,0,1,2], thumbnail 128×128, train-subset percentiles.
Это не утверждённый preprocessing будущего baseline и не физические offsets.

Loss 3.4012 → 2.4178; gradients конечны, веса обновились.
Медиана 0.0784 s/batch, sampled peak RSS 653815808 bytes;
checkpoint 1977141 bytes. Монитор учитывает training phase, не весь этап
загрузки данных. В параллельном окружении шёл аудит/GUI; скорость этого
маленького повторяемого примера не экстраполируется на полный experiment.

Нет train/validation/test experiment, оценки Dice или внешней валидации.
Checkpoint имеет scope multiclass_sanity_only и не является обученной
многоклассовой baseline-моделью. Первый неудачный запуск из-за signed-int8
ignore label исправлен и покрыт тестом; его отдельный каталог сохранён.

## Реальный клинический pilot

Проверенная CT case_001/series_004, 512×512×251,
spacing 0.227152×0.227152×0.5 mm. Только локальная CPU inference
существующего best checkpoint Heart baseline v1, без нового обучения.
Все выбранные оригинальные DICOM SHA совпали до/после.

Выполнены фиксированный image-only preprocessing, обратное преобразование
prediction на original grid и существующий mask→surface pipeline.
Этап inference/mesh/integrity после staging занял 23.85 s.
Результат: 153405 voxels, 11 компонентов, предсказанный объём 3957.71 mm³.
Это объём предсказанной маски, не измерение истинного сердца.

Просмотр axial/coronal/sagittal показал выраженную недосегментацию
и отдельные лишние фрагменты. Маска не является готовой реконструкцией.
Нет эталонного local GT; Dice/HD95 accuracy не рассчитываются.
Различие возраста, контраста, протокола и интенсивностей требует анализа;
причина ошибки не сведена к одному фактору.

## Минимальный ручной редактор

Команда heart3d edit добавляет отдельное окно на общих Qt/PyVista компонентах:
- связанные axial/coronal/sagittal и 3D;
- кисть/ластик с физическим радиусом;
- undo;
- Черновик / Правка / Сравнение;
- выбор точки поверхности → original CT indices;
- отдельные версии маски, операции, комментарии и поверхности.

Геометрия подтверждается по conversion.json и SHA CT.
Редактирование на исходной сетке без resampling.
Оригинальные CT и prediction не перезаписываются; анатомическая разметка
не получает экспертный статус после технической проверки.
Binary Heart label 1 не интерпретируется как LV; namespaces разделены.

Actual Qt widget check проверил выбор координат, ластик, undo, сравнение,
сохранение/загрузку original grid и render 3D.
Тестовое изменение 15 voxels сохранено отдельно с явной пометкой
TECHNICAL UI PROBE ONLY. Оно не является анатомической коррекцией,
не используется как GT. Полная ручная разметка сердца пока не выполнена.
Снимки, версии, модели и приватные provenance остаются на E:.

## Воспроизводимый запуск

Из feature worktree с проектным окружением:
```powershell
$dataRoot = 'E:/3d-heart-data/pediatric_ct_heart'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = 'D:/codexProjects/3d-hearts/data/pediatric_audit/python_deps'

python -m scripts.fetch_chd68_full --root "$dataRoot/public_chd/chd68"
python -m scripts.audit_public_chd --chd68-root "$dataRoot/public_chd/chd68" --imagechd-root "$dataRoot/public_chd/imagechd" --output "$dataRoot/public_chd/audit_v1"
python -m scripts.audit_public_chd_metadata --imagechd-root "$dataRoot/public_chd/imagechd" --audit "$dataRoot/public_chd/audit_v1/audit.json" --output "$dataRoot/public_chd/audit_v1" --summary metadata/pediatric/public_chd_multiclass_audit_v1.json

python -m scripts.multiclass_sanity --ct "$dataRoot/public_chd/chd68/pairs/ct_1008_image.nii.gz" --mask "$dataRoot/public_chd/chd68/pairs/ct_1008_label.nii.gz" --output "$dataRoot/public_chd/multiclass_sanity_new_run" --batches 12

python -m heart3d edit --ct "$dataRoot/external_ct/audit_v1/prepared/case_001/series_004/ct.nii.gz" --mask "$dataRoot/external_ct/pilot_v1/prediction/heart_prediction_original.nii.gz" --geometry "$dataRoot/external_ct/audit_v1/prepared/case_001/series_004/conversion.json" --output "$dataRoot/external_ct/pilot_v1/annotations"
```

Pilot script требует явные --audit-root, --series, --config, --checkpoint,
--output; повторный run — новый каталог. Sanity script аналогично требует
явные image/mask/output и ограничен максимум 50 batches.
CLI не содержит жёсткого dataRoot в Python.

## Provenance реализации

Аудит/download: commit 6ba8e18; модель/loss/sanity: commit 5010732;
editor/clinical pilot: commit d378203.
Короткие runs выполнены до фиксации кода, на рабочем дереве от 37a5b0c.
Sanity checkpoint содержит прежний base revision; его нельзя трактовать
как запись чистого commit состояния. Соответствующие изменения кода теперь
сохранены указанными отдельными коммитами. Source/checkpoint SHA и параметры
runs находятся в обезличенных summaries и полных отчётах на E:.

## Проверки

Полный suite после изменений: **168 passed**, 322 прежних
NumPy/skimage DeprecationWarnings, 19.83 s. Режим Windows: PYTHONUTF8=1.
Проверены бинарная совместимость, восемь выходных каналов, multiclass Dice,
ignore labels, extraction через границу ZIP disks, parsing metadata без дат,
недопустимые/повторные metadata indices, физическая кисть, undo и original-grid
сохранение. Отдельный реальный Qt widget check проверил связанный 2D/3D просмотр.

## Диск, границы результата и следующий шаг

На E: свободно около 256.3 decimal GB; public CHD занимает 16.56 GB.
Резерв 80 GB сохранён. Данные D: не удалялись и не переносились.
Сторонние pretrained модели не загружались. RunPod не использовался;
клинические КТ не отправлялись во внешние сервисы.

Следующий шаг: подтвердить связь metadata/геометрии публичного release,
определить новый patient split и train-only preprocessing; затем короткий
benchmark полноценного dataloader. Development cases, включая ct_1008,
не использовать в blind test. Полное многоклассовое обучение не запускалось.

Текущие CT НИИ — development/qualitative pilot. Для количественного external test
нужны свежие независимые пациенты и проверенный reference standard.
Наша правка prediction не становится независимым GT.
[План получения данных 18 октября](nii_ct_collection_plan.md).
