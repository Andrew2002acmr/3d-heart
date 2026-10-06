# Точка продолжения: pediatric Heart segmentation

Дата: 2026-10-06. Пауза 171/327 завершена после команды пользователя продолжить.
Текущий этап остановлен перед обучением в соответствии с порядком задачи.
Полный отчёт: [pediatric_heart_segmentation_baseline.md](pediatric_heart_segmentation_baseline.md).

## Где находятся изменения и данные

- Ветка: `feature/pediatric-heart-segmentation`, основана на audit commit `66c5804`.
- Worktree: `C:/Users/Андрей/.codex/worktrees/pediatric-heart-segmentation/3d-hearts`.
- Активный raw/cache/derived root: `E:/3d-heart-data/pediatric_ct_heart` — вне Git.
- Inventory: `<data_root>/cache/tcia_series_v1.json` (существующий snapshot).
- Копия на D сохранена без изменений; SHA-256 всех 4,109 copied files совпали,
  674 download receipt hashes проверены. Активные будущие artifacts писать на E.
- Python: `D:/codexProjects/3d-hearts/.venv/Scripts/python.exe`.
- Pydicom 3.0.2: external `data/pediatric_audit/python_deps`, подключается PYTHONPATH.
- Основной checkout с пользовательскими изменениями не переключать и не очищать.

## Проверенные результаты

Strict classic CT adapter: IPP/normal sort, actual Z step, LPS/RAS affine,
HU tags, series/frame/SOP/inventory/regularity. Heart rasterization на original
grid: union/XOR, references и contour planes, coverage flags. **78 tests passed**;
GUI tests исключены, прежние NumPy/scikit-image warnings сохранены.

**9 полных CT + 9 original RTSTRUCT**: original-grid NIfTI, independent
GDCM/SimpleITK geometry/order/HU (difference 0), independent VTK rasterization.
Все девять native mosaics и NiiVue views просмотрены; screenshots вне Git.
Ещё **2 original RTSTRUCT** скачаны для reference audit, без полных CT.

Пилот: BEC712BF, 4C1A38AE, 813E523C исключены из full Heart target из-за
scan truncation. E03568A6 требует coverage review (один slice над ROI).
Остальные пять требуют проверки cranial OAR annotation scope. Source masks не
исправлялись. Alignment подтверждён технически, clinical annotation sign-off нет.

**327/327 metadata census**: 327 Heart definitions, 324 непустых contours,
322 reference candidates, 2 reference failures, 3 empty contours.
272B6C5D: отсутствуют 17 Heart SOP references; 34ECBB32: 8. Полные originals
подтвердили совпадение extracts и эти проблемы. References не переназначались.
Full-stack geometry для 318 остальных пациентов pending. Все clinical statuses
unknown. Фактический reported age 2–16, возрастной фильтр 2–17.

**0 полностью QA-approved training patients**, frozen split отсутствует.
Предложение: 60 после полного QA, по 20 возрастной группы, patient split 42/9/9,
seed 20261006. Девять inspected pilot cases исключить из test. Scanner quotas
условны: revise до frozen split при нехватке full-coverage cases.

Модель предложена, не реализована: собственная 2.5D U-Net, 5 slices, 256×256,
features 16/32/64/128, 488,993 parameters, CPU batch 2/workers 0. RAM 16 GB,
ожидаемый training budget 4–6 GB, duration неизвестен до короткого benchmark.
Radeon backend не проверен; CUDA нет; torch не установлен. Whole raw cohort
теперь хранится на E (~329.90 GB свободно после копии). Минимальный резерв 80 GB.
Суммарный дополнительный бюджет со сжатыми prepared volumes ограничен 228.89 GB;
полная несжатая materialization с архивами и рабочим cache нарушает резерв.
Пересчитывать перед массовым скачиванием. Автоматически ничего не удалять.

## Проверить сохранённые результаты без повторного скачивания

Из worktree, PowerShell:

```powershell
$env:PYTHONPATH = 'D:/codexProjects/3d-hearts/data/pediatric_audit/python_deps'
$pythonExe = 'D:/codexProjects/3d-hearts/.venv/Scripts/python.exe'
$dataRoot = 'E:/3d-heart-data/pediatric_ct_heart'
& $pythonExe scripts/export_pediatric_heart_census.py --data $dataRoot --reviews metadata/pediatric/heart_segmentation_visual_reviews.json --out metadata/pediatric/heart_segmentation_candidates.json
& $pythonExe scripts/view_pediatric_ct_qa.py --data $dataRoot --reviews metadata/pediatric/heart_segmentation_visual_reviews.json
```

Viewer: `http://127.0.0.1:8766`, pinned NiiVue CDN. Preparation `--refresh`
повторяет derived QA без изменения raw. Census cache полный; повторная полная
загрузка девяти пилотных studies не нужна. Historical pause state сохранён как
история, актуальное состояние — `heart_segmentation_preparation_state.json`.

## Следующий разрешаемый этап

1. Согласовать точный Heart OAR target по исходной аннотации и coverage criteria.
2. Выбрать условную когорту 60, суммировать storage по UID; полный CT+RT gate и
   visual QA каждого, замены в stratum или пересмотр квот при необходимости.
3. Зафиксировать patient split, исключая inspected pilot из test; fit preprocessing
   только на train, сохранить original-grid GT и inverse transforms.
4. Реализовать собственную 2.5D модель, Dice/BCE losses, train/evaluate/inference.
5. Короткий train-only benchmark даст RAM/speed и бюджет полного эксперимента.
6. Многочасовое обучение — только после отдельного согласования. Predictions,
   test metrics, GT/Prediction viewer и 3D comparison пока отсутствуют.
