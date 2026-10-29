# Heart Localization & ROI Segmentation — Stage05

Запуск существующего CT Lab из `repository`:

```powershell
.venv/Scripts/python.exe -m ct_lab.roi_launch
```

http://127.0.0.1:8510 → Heart Localization & ROI Segmentation. Восьмиклассовые исходные маски:0background,1LV,2RV,3LA,4RA,5MYO,6AO,7PA; прочие labels→ignore255. Source counts не HU, header units не подтверждены как миллиметры.

Последовательность CLI: `roi_analysis` → `roi_smoke` → `roi_localizer` → `roi_train`, затем `roi_baselines`, `roi_topology`, `roi_analysis` (обновлённые geometry-only gradient diagnostics), `roi_verify_ui`, browser/test suite, `roi_report`. Каждый запускается как `.venv/Scripts/python.exe -m ct_lab.<name>`. Baseline/topology можно запускать параллельно основному train в отдельных sessions: topology ждёт готовые predictions/scores, GPU baseline имеет небольшой ограниченный batch.

Основная матрица: новый Full FOV256/384, Predicted ROI256/384, Oracle256 (диагностический), Predicted256 с необязательной3D-core branch. Все новые main jobs:20epochs×192steps×batch2, same base489112params, CE+Dice, AdamW, same seed/case-Z sampling. Branch добавляет896params. Пять исходных соседей Z±2 не resample по Z; predicted ROI задаёт общий aligned XY crop и диапазон центральных Z, context halo ±2 берётся из исходного объёма.

Предварительный train-only CV начинался с6epochs×48steps. Background-only результат вызвал симметричное продление толькоCVдо18epochs, без outer segmentation metrics: см. `training_gate.json`, сохранённые pilot results. Основная матрица не менялась. Триfolds внутри48train; scaler refit на validation отсутствует. Исторический train-fitted source clip0–2015 наследуется заранее, поэтому CV не является полностью независимой nested evaluation. Margin выбран по train-onlyOOF, validation GT не исправляет bbox. Source-level intensity/noise variant фиксирован на case и общий всем конфигурациям; core пересчитан на том же CT, затем все scalar/image fields пространственно warp совместно, targets nearest.

Checkpoint: `best.pt` у main — fixed final epoch; `last.pt` содержит optimizer/GradScaler/history и позволяет продолжать после остановки на границе эпохи. Завершённые result.json пропускаются. Не переиспользовать cache после изменения протокола; для нового независимого run создать отдельную копию проекта/outputs. Исходные файлы и старые эксперименты не удалять.

Inference не получает экспертную маску. Oracle на validation включается только явно и всегда помечен диагностикой. В GUI можно применить готовую модель к любому разрешённому train/development CT; такие результаты сохраняются отдельно в `interactive/` и не входят в frozen comparison. Sealed test блокируется до source I/O. UI автоматически не обучает модели.

Native metrics: inverse probability interpolation до argmax, общие source grids/ignore для всех методов. Primary Dice — mean reference-present foreground classes, затем mean release-case. HD95 — exact index-space EDT/KD, не mm; поверхности у source/ignore boundary censored. Components —6-connectivity на valid domain, все сохранены. Raw Heart3D VTP строятся по исходным маскам, включают все предсказанные области; browser display decimation не используется в метриках. Heart3D metadata может дополнительно содержать26-connectivity; её не смешивать с6-connected primary counts.

Команды тестов:

```powershell
.venv/Scripts/python.exe -m pytest -p ct_lab.roi_pytest tests/test_roi_lab.py tests/test_multiclass_lab.py tests/test_learnable_features.py tests/test_spatial_features.py tests/test_filtering_features.py tests/test_ct_lab.py tests/test_cardiac_inference.py -q
.venv/Scripts/python.exe -m ct_lab.roi_verify_ui
node ct_lab/roi_browser.cjs
```

Plugin `roi_pytest` изолирует JIT cache stage05 после collection; прежние outputs остаются byte-for-byte неизменными. Production `test_cardiac_training.py` не требуется. Browser script использует existing Playwright из `web/node_modules`, новый изолированный временный Chrome profile.

Все полные данные/cache/NIfTI/VTK остаются в `outputs/heart_localization_roi_05/`. Компактный ZIP исключает большие input caches, raw `.npy`, VTP и optimizer last.pt; содержит main/CV/localizer checkpoints, результаты, display meshes, графики и код. RunPod запуск не выполнялся и платные ресурсы не создавались: RTX2060 прошла CPU/GPU gates. Для переноса на4090 сначала скопировать тот же проект, frozen split/data/weights и зависимости, повторить smoke и использовать те же CLI; source protocol/record hashes должны совпасть.

Перенос в уже выделенный RunPod: установить зависимости проекта и CT Lab, задать `ROI05_DATA_ROOT` (каталог NIfTI pairs) и `ROI05_WEIGHTS_ROOT` (неизменённые v1/v2 checkpoints), затем из корня repository выполнить `bash scripts/run_roi05_runpod.sh`. Скрипт проверяет наличие RTX 4090 и не создаёт платные ресурсы. Для независимого воспроизведения нужна новая копия проекта с новым каталогом stage05 outputs; прежние результаты сохраняются. `ct_lab/roi_execution_gate.json` воспроизводит выполненное продление предварительной CV до 18 эпох; исходная предрегистрация остаётся в protocol.json.

Перед `node ct_lab/roi_browser.cjs` запустить `roi_launch` в отдельном терминале. GUI проверяет SHA256 выбранных CT и GT, прежде чем использовать сохранённые study predictions: другой файл с тем же case ID отклоняется. Пример ct_1001 явно помечается как train; development не называется независимой финальной оценкой.

CPU-only выбирается автоматически при отсутствии CUDA. Для принудительного CPU-режима в PowerShell перед CLI задать `$env:CUDA_VISIBLE_DEVICES='-1'`. Входные кэши сети — float16, вычисление признаков возвращает float32, ограниченные фильтровые patch intermediates прежнего extractor — float64. 3D-core cache публикуется атомарно и проверяет source hash/shape/affine/bbox; FileLock предотвращает одновременную запись. Кэш не считается готовым только по наличию `.npy`.

Основная статистика сравнения — ровно десять development cases. `study_summary.json` содержит role/case_ids и отдельную train ct_1001 колонку. Поле mean в сырых evaluation summary агрегирует все запрошенные случаи, включая train example; его нельзя выдавать за development score. Исторические CHD68 имеют другой training budget/FP32 inference, а branch добавляет 896 параметров и сведения об исходных counts выше clip 2015. Эти сравнения требуют соответствующих оговорок.
