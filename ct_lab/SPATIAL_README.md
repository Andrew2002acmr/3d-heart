# Spatial Feature Experiment 02

Из корня `repository`:

```powershell
& .venv\Scripts\python.exe -m ct_lab.spatial_run
& .venv\Scripts\python.exe -m ct_lab.spatial_artifacts
& .venv\Scripts\python.exe -m ct_lab.spatial_report
& .venv\Scripts\python.exe -m ct_lab.spatial_package
& .venv\Scripts\python.exe -m ct_lab.spatial_launch
```

Последняя команда запускает прежнюю CT Lab на `http://127.0.0.1:8510` с вкладкой
**Spatial Feature Explorer** и отдельным numba cache этапа 2. Если лаборатория
уже работает, используйте существующий сервер или остановите его перед запуском.
Зависимости уже установлены для первого эксперимента; новые пакеты не требуются.
Предыдущие результаты/отчёты не перезаписываются.

`outputs/spatial_feature_02/protocol.json` фиксируется до validation scoring.
Источник sampled voxels — неизменённые `outputs/filtering_feature_01/samples_*.npz`:
те же 5 train groups, 10 validation groups, Z-fractions, все source IDs, class labels
и voxel-fold. A–H используют одинаковые наблюдения. Filter diagnostics используют
отдельную одинаковую reduced subset (до 32/class/source Z), со своим Baseline.
Сравнивать её F1 напрямую с основной таблицей нельзя.

Признаки извлекаются по выбранным точкам с корректным halo, float32 outputs,
chunk=64 (filtered=16), без dense dataset tensor. Texture: reflect; profiles и
Z-context: replicate edge. Статистики uniform window=5, LoG σ=1 header grid unit;
Gaussian 3D channels σ=0.5/1/2. Unknown spacing не называется миллиметрами.
Native source grid проверяется на axis-aligned RAS; oblique input отклоняется,
чтобы не подменять fixed source IDs молчаливым resampling.

G включает Intensity, четыре 3D texture features и три Gaussian channels.
F включает XY/XZ/YZ texture; raw XYZ profiles сохранены для исследования, но не
подмешиваются к F. C/D/E/H явно получают Z-context. Координаты, expert mask,
class IDs и distance-to-GT не являются classifier/projection inputs.

Three-plane viewer использует один source volume, affine и source XYZ. Canvas
click возвращает точную координату в каждом отображённом пикселе. Overlay маски
регулируется. Scatter lasso/point selection переносит source IDs на viewer:
сплошные markers — in-plane; rings — проекции выбранных точек с других срезов.
Error overlay рисует только evaluated in-plane voxels, а не весь объём.
Произвольная foreground-пара доступна как exploratory train-only режим.

Validation можно просматривать с fixed predictions, без fit PCA/UMAP/scaler.
Sealed test блокируется перед загрузкой CT. Validation уже использовалась в
эксперименте 1: это **продолжение разработки**, не независимая финальная оценка.

Independent PCA/UMAP fit на train voxel fold ct_1001; одинаковые IDs, seed и
настройки экспортируются. Shared-intersection projection — корректный контроль
на B∩context с одинаковыми before/after координатами. Новые размерности в этом
контроле исключены; он не доказывает отсутствие пользы новых каналов. Матрицы
разной размерности не подаются одной PCA без явного общего пространства.

`charts.html` — индекс всех 10 типов графиков. `results.json`, `scores.csv`,
`by_case.csv`, `per_class.csv`, `error_diagnostics.json` содержат измеренные
метрики. `errors_*.npz` и `validation_predictions.npz` содержат source IDs,
истину, predictions и диагностическую distance, хранящиеся отдельно от features.
`train_development.json` — отдельный оптимистичный voxel holdout по 5 train cases.

Проверки (без neural training):

```powershell
& .venv\Scripts\python.exe -m pytest tests/test_spatial_features.py tests/test_filtering_features.py tests/test_ct_lab.py tests/test_cardiac_inference.py -q -o cache_dir=outputs/spatial_test_cache --basetemp=outputs/spatial_test_tmp_new
& .venv\Scripts\python.exe -m ct_lab.spatial_verify_ui
```

Для pytest выбирайте новый basetemp: pytest удаляет существующую папку.
`spatial_browser.cjs` — browser interaction checks в отдельном Chrome через
существующий `web/node_modules/playwright`; не использует личный профиль браузера.
`spatial_benchmark` — реальное время CPU и sampled RSS на fixed ct_1001 subset;
первый bilateral вызов включает JIT compilation, поэтому это не чистое сравнение
скорости 2D/3D.
