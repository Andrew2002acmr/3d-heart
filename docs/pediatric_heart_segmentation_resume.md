# Точка продолжения: pediatric Heart segmentation

Работа приостановлена по просьбе пользователя 2026-10-06. Обучения нет.

## Где находятся изменения и данные

- Ветка: `feature/pediatric-heart-segmentation`, основана на audit commit `66c5804`.
- Worktree: `C:/Users/Андрей/.codex/worktrees/pediatric-heart-segmentation/3d-hearts`.
- Raw/cache/derived data: `D:/codexProjects/3d-hearts/data/pediatric_ct_heart` — вне Git.
- Source inventory: `D:/codexProjects/3d-hearts/data/pediatric_audit/tcia_series.txt`.
- Python: `D:/codexProjects/3d-hearts/.venv/Scripts/python.exe`.
- Pydicom 3.0.2: external `data/pediatric_audit/python_deps`, подключается PYTHONPATH.
- Основной checkout остаётся с пользовательскими незакоммиченными изменениями;
  их не переключать, не удалять и не переносить в эту ветку.

## Зафиксированный результат

Собственный strict classic CT adapter: IPP sort, actual Z step, LPS/RAS affine,
HU gate, проверки series/frame/SOP/inventory/regularity. Heart rasterization
на original grid, union/XOR, проверка references и contour planes.
Синтетические и существующие тесты: **72 passed**; warnings — прежние
NumPy/scikit-image deprecations.

Скачаны **9 полных CT series + 9 оригинальных RTSTRUCT**, ZIP CRC и SHA receipts.
Из них **8** конвертированы в original-grid CT/GT NIfTI, независимо проверены
SimpleITK и VTK, сохранены axial/coronal/sagittal mosaics. Девятый E03568A6
полностью скачан, но ещё не конвертирован/проверен.

Mosaics восьми случаев просмотрены. NiiVue screenshots и landmarks проверены
для первых двух. BEC712BF, 4C1A38AE, 813E523C исключены из whole-Heart baseline
из-за superior scan truncation. Для остальных пяти сохранён review исходной
cranial границы OAR target; контуры не исправлялись и не объявлялись полноценной
cardiac whole-heart аннотацией.

Census остановлен после **171/327** пациентов: 166 metadata candidates,
2 требуют review unmatched RT SOP references, 3 без Heart ROI. Осталось 156.
Metadata pass не означает full-stack QA или suitability. Partial snapshot:
`metadata/pediatric/heart_segmentation_candidates.json`; полный рабочий cache
и checkpoint census находятся в external data folder. Все процессы census,
download и QA viewer остановлены; background training отсутствует.

План baseline: 60 полностью одобренных cases, patient split 42/9/9,
собственная 2.5D U-Net, 5 slices, 256×256, features16/32/64/128, ~489k parameters,
CPU batch2. Это предложение. Split не заморожен, preprocessing не fitted,
модель и training pipeline не реализованы. AMD Radeon не подтверждена как
поддерживаемый PyTorch accelerator. RAM16 GB; целая raw cohort не помещается на D.

## Продолжить без повторного полного скачивания

Сначала `git status`, проверить branch/worktree и сохранность external cache.
Из нового worktree, PowerShell:

```powershell
$env:PYTHONPATH = 'D:/codexProjects/3d-hearts/data/pediatric_audit/python_deps'
$pythonExe = 'D:/codexProjects/3d-hearts/.venv/Scripts/python.exe'
$dataRoot = 'D:/codexProjects/3d-hearts/data/pediatric_ct_heart'
$seriesFile = 'D:/codexProjects/3d-hearts/data/pediatric_audit/tcia_series.txt'
& $pythonExe scripts/prepare_pediatric_ct_pilot.py --data $dataRoot
& $pythonExe scripts/census_pediatric_ct_heart.py --registry metadata/pediatric/registry.json --series $seriesFile --data $dataRoot --out "$dataRoot/census.json" --workers 8
```

Preparation использует существующие full studies, пропускает восемь уже
конвертированных cases. `--refresh` повторно проверяет derived outputs и
обновляет mosaics (полезно для первых двух с длинными titles); raw не изменяет.
Census продолжит 156 отсутствующих cases и повторит два требует-review cases
из их cache. Полные downloads уже завершены, запуск downloader не требуется.

Далее: проверить E03568A6 и остальные cases в independent viewer, записать
visual review, проверить причины reference failures. Когда census полный:

```powershell
& $pythonExe scripts/export_pediatric_heart_census.py --data $dataRoot --reviews metadata/pediatric/heart_segmentation_visual_reviews.json --out metadata/pediatric/heart_segmentation_candidates.json
& $pythonExe scripts/view_pediatric_ct_qa.py --data $dataRoot
```

Viewer: `http://127.0.0.1:8766`, pinned NiiVue CDN. До финального отчёта дополнить
pilot review девятым case и уточнить annotations/coverage и реализуемые age/scanner
квоты. Для 60 future candidates требуются full-volume gates; не превращать
171 metadata records в training split. **После завершения подготовки остановиться
и показать отчёт; многочасовое обучение требует отдельного согласования.**
