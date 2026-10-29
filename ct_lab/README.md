# CT Preprocessing Lab

Эксперимент 01 добавляет раздел **Filtering & Feature Analysis**.
См. `FILTERING_FEATURE_EXPERIMENT_01.md` в корне проекта.

```powershell
& .venv\Scripts\python.exe -m ct_lab.filtering_run
& .venv\Scripts\python.exe -m ct_lab.filtering_artifacts --umap
& .venv\Scripts\python.exe -m ct_lab.filtering_report
```

Основной пример — ct_1001; переключение LV/MYO и RV/PA. Z/ROI/filter и
display/feature-axis controls собраны в раскрывающиеся панели. Исходное и
обработанное изображения используют одну явно заданную шкалу. Выделение scatter
передаёт source voxel IDs обратно на оба изображения. Absolute difference и
верхний 1% изменений показывают, где действует фильтр. Gradient Magnitude —
отдельный канал признаков.

PCA использует exact NumPy SVD; diagnostic k-NN — exact SciPy cKDTree, k=15,
distance weights. Для UMAP установить `requirements-filtering.txt` в отдельное
окружение; workspace-local launcher также подхватывает `../ct_feature_deps`.
Numba cache записывается в derived output. UMAP запускается checkbox по запросу,
после включения автоматически обновляется при изменении параметров. Первая
компиляция может занять время; проекции кэшируются.

Batch protocol фиксирует 5 development/train groups и все 10 frozen validation
groups до scoring; по 3 заранее заданных axial Z и до 400 voxels/class/slice.
Test CT не читаются. Сохранённые train-only mean/std относятся к диагностическим
features, а не к preprocessing CHD68. Классы mask/координаты используются только
для sampling, подсветки и оценки, не подаются классификатору как признаки.

Все outputs — `outputs/filtering_feature_01/`. `protocol.json` блокирует случайное
изменение протокола поверх существующего эксперимента. Raw CT/GT, checkpoints,
split и CHD68 preprocessing проверяются SHA256 на неизменность. Для полного
повторного извлечения использовать новый `--output` у filtering_run; отчёт и
artifact scripts по умолчанию читают основной output первого эксперимента.

`linked_ct1001.html` содержит выбор фильтра/осей, linked source coordinate
highlight и пары проекций. `validation_charts.html` — equal-case F1 и pooled
confusion matrices; `results.json` сохраняет также метрики каждого validation
случая и отдельно оптимистичный within-ct_1001 voxel holdout.

Отдельный Streamlit-интерфейс. Heart3D, исходные КТ, маски и веса не изменяются.
Python 3.11+, CPU. В рабочей папке используется существующий Python 3.13 из `.venv`.

Запуск из `C:\Users\Andre\Desktop\3D-heart\repository`:

```powershell
& .venv\Scripts\python.exe ct_lab\launch.py
```

Открыть http://127.0.0.1:8510. Каталог по умолчанию — соседний
`laboratories/data/chd68/pairs`, 68 реальных пар NIfTI. Другой каталог можно задать
в sidebar. Пары именуются `*_image.nii[.gz]` / `*_label.nii[.gz]`.
Непарное изображение можно анализировать без GT; несогласованная геометрия пары
отклоняется, а не исправляется молча. DICOM-серии этим модулем не импортируются:
фактически найденный ImageCHD68 — NIfTI.

Установка на другой машине: в отдельном окружении установить
`requirements-ct-lab.txt`. Здесь UI-зависимости устанавливаются в соседнюю папку
`ct_preprocessing_deps`; launch добавляет её после существующих scientific packages.
Существующее окружение Heart3D не обновляется.

Пакетный анализ:

```powershell
& .venv\Scripts\python.exe ct_lab\launch.py batch --data ..\laboratories\data\chd68\pairs --output outputs\ct_preprocessing_lab
```

JSON на случай, `dataset.json`, `patients.csv`, `classes.csv`, интерактивный
`report.html`. Анализ последователен, с возобновлением по path/size/mtime;
SHA256 исходных файлов сохраняется в JSON. Для повторной проверки содержимого
при неизменённых timestamp используйте новый output-каталог. `--components`
добавляет дорогую 2D/3D связность всех масок. Сбой одной пары попадает в отчёт,
остальные исследования продолжают обрабатываться. Сводная статистика — по случаям,
с равным весом каждого случая, без подгонки preprocessing по test.

Разделы:

- Dataset Overview: metadata, scale provenance, классные доли, batch результаты.
- CT Explorer: физическое RAS reslicing, три проекции, общая позиция, overlay,
  zoom/pan, hover. Для oblique affine используется interpolation, не transpose.
- HU Analysis: название раздела соответствует заданию; сами интенсивности явно
  подписаны как source scale. Весь объём, срез, ROI, сердце, классы, background;
  density/box/violin, межпациентное сравнение, совместные распределения.
- Preprocessing: clipping, normalization, 2D Gaussian/median/bilateral/NLM/CLAHE,
  карты границ, transfer function, before/after, resampling и ROI crop/pad.
- Spatial Context: ±3 среза, разности, осевой профиль, линия по двум заданным
  координатам с отображением на срезе и границами экспертной маски.
- Mask Anatomy: площади, объёмы в grid units, центроид, протяжённость,
  26-connected 3D и 8-connected 2D компоненты и их размеры.
- Model Input Inspector: настоящий `infer_cardiac.prepare_volume`, пять каналов,
  стадии в реальном порядке, сохранённые train bounds, статистики, NPY.
- Model Diagnostics: проверка исторического checkpoint SHA, CPU forward,
  `forward_pre_hook`, вероятности, entropy, slice Dice и 2D компоненты;
  отдельный запуск полного volume inference, case Dice, 3D/2D связность,
  сравнение площадей предсказаний и GT на исходной сетке.
- Experiments: сохранение конфигураций/результатов и HTML, сравнение в одной
  пространственной позиции, ZIP с JSON/CSV/NPY/HTML/PNG/SVG.

Ограничения: HU и миллиметры релиза не подтверждены; spacing=1 из заголовка не
восстанавливает физическую геометрию пациента. Глобальных train mean/std нет,
поэтому сохранённая train-нормализация доступна как fixed train window, без
выдуманного Z-score. Фильтры работают на одном срезе. Box/violin используют
детерминированную выборку до 5000 значений/класс, межпациентный график — около
100000/случай; гистограммы одного исследования и таблицы рассчитываются полностью.
Эксперименты не доказывают улучшение сегментации.
Min-Max, percentile и fixed-window нормализация здесь отображают результат в
[-1,1]. Дополнительные source labels вне 0..7 остаются в анализе, но исключаются
из cardiac metrics как ignore=255 согласно историческому pipeline.

Модель 2.5D, а не 3D. Полный volume inference запускается отдельно по запросу,
не при каждом изменении среза. Batch-анализ предсказаний и feature maps пока
не реализованы. Исторические результаты не подменяются slice metrics.

Тесты и real CPU smoke:

```powershell
& .venv\Scripts\python.exe -m pytest tests\test_ct_lab.py tests\test_cardiac_inference.py tests\test_cardiac_training.py -q -o cache_dir=outputs/ct_lab_test_cache --basetemp=outputs/ct_lab_test_tmp_new
& .venv\Scripts\python.exe -m ct_lab.verify_real
& .venv\Scripts\python.exe -m ct_lab.verify_real --full
& .venv\Scripts\python.exe -m ct_lab.verify_fragments
& .venv\Scripts\python.exe -m ct_lab.research
& .venv\Scripts\python.exe -m ct_lab.write_report
```

Для `--basetemp` выбирайте новую папку: pytest удаляет существующую базовую
временную папку. Real smoke использует только `ct_1001` (train/development),
записывает provenance и slice metrics в `outputs/ct_preprocessing_lab`.
`--full` добавляет полный объём для обеих моделей, отдельно 3D/2D connectivity
и case Dice. `verify_fragments` сохраняет derived NIfTI predictions и таблицы
пересечений каждого компонента с GT, проверяя совпадение full-volume Dice
с завершённым прогоном. Повторно использует уже сохранённые predictions.
