# Pediatric Heart: точка продолжения после baseline v1

Дата обновления: 2026-10-09. **Один полный baseline завершён. Все критичные результаты
скачаны на E: и проверены по SHA-256. Pod можно остановить.**
Автоматическая остановка не выполнялась.
Работа остаётся в `feature/pediatric-heart-segmentation`.
Training commit: `7c93cb9f8b284a9a526e78c9c67432e26d15e794`.
Подробности: [RunPod report](pediatric_heart_runpod_v1.md),
[baseline](pediatric_heart_segmentation_baseline.md).

## Что завершено

Census: 327 пациентов возрастом 2–16 лет. Получены 117 полных CT/RTSTRUCT пар;
114 прошли технические проверки и визуальный review. Пригодны 62 source Heart OAR
случая; 50 исключены по coverage, 3 по geometry, по одному оставлены для coverage
и identity review. Статус approved означает техническую пригодность исходной
разметки, клинический статус всех случаев остаётся unknown.

Frozen cohort: 60 пациентов, по 20 в каждой рабочей возрастной группе.
Split: 42 train / 9 validation / 9 test; внутри группы 14/3/3, seed 20261006.
Development IDs исключены из test. Не менять split/cohort/preprocessing и review
reports после freeze. Public IDs и content hashes не доказывают отсутствие всех
повторных исследований одного человека.

Train-only preprocessing: HU [−1000,969], полный FOV 256², Z 2 mm,
5 physical contexts [−4,−2,0,2,4] mm. Собственная 2.5D U-Net:
16/32/64/128 channels, GN8, 488993 параметра, FP32, Adam0.001,
batch16, BCE + Dice, 30 epochs с нуля. Benchmark weights не использовались.

GPU benchmark: 0.3344 s/batch, 47.84 samples/s, peak allocated VRAM 1.916 GB.
Реальное обучение с validation/save: **41.10 min**, best checkpoint epoch18.
Однократный original-grid test, n=9: mean Dice0.9194, IoU0.8521,
precision0.9472, recall0.8977, HD95mean17.90 mm/median14.00 mm/max70.65 mm,
ASSDmean3.01 mm. Это исследовательский source-OAR baseline.

## Локальные файлы

Data root: `E:/3d-heart-data/pediatric_ct_heart`; резерв 80 decimal GB, свободно ~277 GB.
Worktree: `C:/Users/Андрей/.codex/worktrees/pediatric-heart-segmentation/3d-hearts`.
Основной D: checkout, пользовательские изменения и старые D: данные не трогать.
В Git Credential Manager сохранены две учётные записи. Финальный push успешно
выполнен с явным выбором владельца репозитория, без изменения глобальных настроек:
`git -c credential.username=Andrew2002acmr push origin feature/pediatric-heart-segmentation`.
Локальный Python: `D:/codexProjects/3d-hearts/.venv/Scripts/python.exe` (CPU torch2.14.1).
External pydicom: PYTHONPATH `D:/codexProjects/3d-hearts/data/pediatric_audit/python_deps`.
Pod использовал Python3.12.3 / torch2.8.0+cu128 / CUDA12.8 под persistent /workspace.

`runpod_results/gpu_full_v1_20261006_io/` содержит best/last, history, config,
environment, pip freeze, resource samples, 9 predictions, provenance и test metrics.
`runpod_results/gpu_full_v1_20261006_io_audit/` содержит preflight, logs и сведения
об остановленном медленном pilot. Archives/receipts находятся в `runpod_transfer/`.
Проверены все 31 result и 9 audit файлов; best/last успешно загружаются на CPU.
QA, meshes, plots, screenshots и DICOM inference smoke находятся под
`experiments/heart_baseline_v1/gpu_full_v1_20261006_io/`.
Original CT/RTSTRUCT и CT/GT остаются в `<dataRoot>/<patient>/` без перезаписи.

## Проверки и ограничения

Полный suite: 127 passed; после расширения failure review ещё один QA test passed.
DICOM inference проверен на engineering/train06722123 без GT/RTSTRUCT.
Viewer показывает 9 test cases в режимах GT/Prediction/Comparison. Реальный
F22EFF95 проверен в headless Edge: JS errors0, Comparison screenshot просмотрен.
Для трёх заранее выбранных test cases и одного post-hoc failure построены meshes;
никакие компоненты не удалялись и source annotation не исправлялась.

792705D7: off-heart islands, HD95 70.65 mm. 423F282F: islands в области стола CT.
CA967BD7 (2y): недосегментация, recall0.7569, volume error−23.54%.
F50AD62F: volume error+12.97%. Target OAR не гарантирует полной камерной/сосудистой
анатомии. Test n=9, два scanner models, подтверждённых non-contrast случаев нет;
о клинической или широкой domain robustness делать вывод рано.

## Посмотреть без обучения

```powershell
$dataRoot='E:/3d-heart-data/pediatric_ct_heart'
$resultsRoot="$dataRoot/runpod_results/gpu_full_v1_20261006_io"
$testPredictions="$resultsRoot/experiments/heart_baseline_v1/gpu_full_v1_20261006_io/test"
python -m heart3d.ml.bundle verify --root $resultsRoot --manifest "$resultsRoot/bundle_manifest.json"
python -m scripts.view_pediatric_ct_qa --data $dataRoot --predictions $testPredictions --reviews metadata/pediatric/heart_cohort_reviews_v1.json --port 8766
```

Viewer слушает только 127.0.0.1. Если уже запущен на8766, использовать текущий
server; второй на том же порту не стартовать. Обучение автоматически не повторять.

## Следующий согласуемый этап

Уточнить OAR scope с экспертом. Заранее спланировать исследования ложных фоновых
компонент и 3D continuity на train/validation. Уже просмотренный v1 test нельзя
использовать для подбора threshold/postprocessing и вновь называть blind set.
Для новых методов объявить v2 protocol и held-out plan до запуска.
У НИИ запросить cardiac CT с согласованными anatomical и clinical labels для
будущей многоклассовой задачи. MRI/Atlas/diagnosis/healthy classifiers и СППР
здесь не начинались. Не merge main, не force push/reset --hard/clean -fd.

## Обновление 2026-10-09: клинические CT и цель 0–17

Пользователь подтвердил 0–17 лет, новорождённых и младенцев проверять отдельно.
Frozen v1 (2–16/Heart OAR) не менять. Полуручной редактор — requirement,
Astra Linux/ClearCanvas версии неизвестны, VR/печать — будущие exports.

Input: <dataRoot>/external_ct/incoming. 5273DICOM (5241CT,24SEG,8SR),
2studies/50series. После работы все5273original SHA совпали.
Один distinct PatientID: нельзя автоматически утверждать два разных patients;
identity/de-identification не подтверждены.
13classic CT series header geometry passed;5review, включая multiphase.
Два native volumes decoded и независимо проверены SimpleITK.
22/24BINARY SEG imported;2FrameOfReference failures оставлены review.
Labels Heart/LV визуально не подтверждены как final anatomy.
4source masks прошли mask→mesh, без clinical/print-ready approval.

Private reports/volumes/masks/meshes/PNG: <dataRoot>/external_ct/audit_v1.
Реальный dataRoot указан выше; пути CLI. Данные не отправлялись RunPod/cloud.
GT не сертифицирован, новые training/inference runs не выполнялись.
Независимый Slicer GUI review SEG ещё предстоит.

Итоговый suite: 148 passed, 322 existing NumPy/skimage warnings, 27.93 s (UTF-8 mode).
Windows: PYTHONUTF8=1. Без него существующие ML read_text()/write_text()
и non-ASCII profile paths дали6FileNotFound failures. Это locale limitation;
клинические данные ради устранения ошибки не менялись.

Next: согласовать selected phase/структуры/purpose SEG, получить final
reference masks/meshes и проверить deployment на конкретной Astra.
[Import audit](external_ct_import_audit.md),
[clinical workflow](clinical_reconstruction_workflow.md).

## Уточнение следующего этапа: НИИ не готовит training masks

По сообщению пользователя получение ручной разметки от занятого хирурга
маловероятно. Не блокировать разработку на expert segmentation delivery.
Принят новый план: public annotated CT для собственной модели и research manual/
semi-automatic correction для local NII CT; optional точечный anatomical review
по готовым panels вместо разметки всей cohort.
[План](annotation_without_clinician_plan.md) — proposal, без нового training,
model download, RunPod transfer или predictions.
Frozen v1 неизменен. Новая local research annotation не expert GT.

## Разбор автореферата: исследовательское ядро

Проанализирована 40-минутная запись пользователя: локальное распознавание речи
и выбранные кадры автореферата, без загрузки записи во внешние сервисы.
Артефакты: `<dataRoot>/research_review/supervisor_20261007/`.
[Разбор и предложения](supervisor_review_and_project_direction.md) отделяет
замечания автора записи от предлагаемых формулировок и завершённого baseline.

Предложен акцент на проверяемом 3D, многоуровневом контроле ошибок,
связанном 2D/3D и ручной коррекции. Собственная U-Net — baseline, не новая архитектура.
СППР, диагностика и поиск аномалий — перспектива.
Без экспертной оценки клиническую пользу не заявлять; техническую проверку
и работу с публичной разметкой продолжать.
Название не менялось, Word-файлы основного checkout не редактировались.
Новых медицинских экспериментов, обучения или inference не запускалось; v1 неизменен.

## Обновление 2026-10-09: public multiclass audit и клинический pilot

Продолжение по подтверждённому пользователем плану. Полный CHD68 release:
68 CT/label pairs скачаны на E:, CRC/SHA подтверждены, все pair-grid gates прошли.
13 масок имеют дополнительные несердечные labels. 63 пересекаются с ImageCHD;
для трёх доступных пар полная идентичность проверена SHA. Не считать независимыми
пациентами только из-за разных названий dataset.

В ImageCHD XLSX найдены metadata candidates для 63 CHD68 случаев. У 58 возраст
кандидата 0–17, у пяти 29–47; это противоречит опубликованному максимуму 21 год.
Linkage не подтверждена авторами, age остаётся unverified. Все release headers
1×1×1/unknown units: source spacing пока не назначать release arrays.
Новой frozen pediatric cohort нет. Черновик письма авторам подготовлен, не отправлен.

Собственная U-Net получила optional 8-output layer; бинарный v1 совместим.
CE/собственный multiclass Dice/CE+Dice поддержаны. 12 CPU batches на одной
development-плоскости: loss 3.4012→2.4178, finite gradients/weight updates.
489112 parameters; median 0.0784 s/batch — только tiny in-memory sanity,
без научной accuracy, cohort/test run или оценки времени большого обучения.

Локальная inference бинарного Heart v1 на case_001/series_004 и mask→mesh:
23.85 s после staging, source DICOM unchanged. Визуально маска сильно неполна,
11 компонентов; GT отсутствует. Это development failure case, не external accuracy.
Реализован редактор original grid: три плоскости, кисть/ластик/undo, отдельные
версии, сравнение и поверхность→КТ. Реальный widget check прошёл; 15 изменённых
voxels — только UI-probe, не анатомическая коррекция. Полная local разметка впереди.

Все новые данные/weights/PNG на E:, резерв 80 GB сохранён. НИИ CT никуда
не отправлялись. RunPod/full training не использовались. Astra/VR/печать отложены.
Полный suite: 168 passed, 322 существующих warnings, 19.83 s (PYTHONUTF8=1).
[Итог этого этапа](cardiac_structure_segmentation_pilot.md),
[поставка НИИ](nii_ct_collection_plan.md),
[запрос авторам](public_chd_metadata_request.md).

## Подготовка multiclass training для будущего RTX4090 Pod

По уточнению пользователя подготовлена исполняемая GPU-конфигурация, вместо
ограничения эксперимента CPU sanity. Новые cohort/split **68 / 48-10-10** описывают
public CHD68 release с непроверенными возрастами/physical units, не approved
pediatric cohort. Frozen Heart v1 не меняется. Train-only clip [0,2015],
256²/full FOV, 5 index contexts; actual train epoch 12381 samples.
Реализованы lazy dataset, multiclass augmentation, CUDA benchmark, scratch trainer,
original-grid evaluation, best/last/history/environment/provenance и integrity checks.
Готовый минимальный training bundle 3.333 GB на E:, без raw CT/НИИ.
Pod ещё не запущен, transfer/GPU benchmark/full training не выполнялись.
[Инструкция и готовность](cardiac_chd68_runpod_readiness.md).

## Обновление после миграции RunPod — 2026-10-09

Endpoint 213.173.109.80:12476: SSH прежним RunPod key работает.
RTX4090/CUDA и persistent /workspace проверены; старый Heart v1 сохранён.
Для нового multiclass baseline отдельный checkout d89ea28 и публичный bundle
3.333 GB; archive/225 files/repository configs SHA-256 совпали.
Linux targeted tests: 7 passed. Короткий GPU benchmark: 0.3031 s/batch,
3.91 min/train epoch, 117.30 min/30 train epochs без validation/checkpoint I/O.
Sanity loss 3.3389→1.8332; peak RSS1.883 GB, CUDA reserve3.127 GB.
Это не segmentation quality; test не использован. Full training не запускалось.
Results backup (8 files) на E:/3d-heart-data/pediatric_ct_heart/remote_runs/
cardiac_chd68_v1_20261009 полностью проверен SHA-256. Pod можно остановить.
Для full launch сохранить execution commit d89ea28 и matching benchmark receipt;
после обновления HEAD требуется повторный benchmark. Frozen split не менять.
[Подробности и ограничения](cardiac_chd68_runpod_readiness.md).

## Full multiclass training запущено — 2026-10-09

Пользователь явно разрешил полное обучение и временную проверку каждые 5 минут.
Detached runner PID1487, trainer1526, run gpu_full_v1_20261009; execution commit
остаётся d89ea28, config/split/preprocessing неизменны. 30 epochs с нуля,
затем одна evaluation best validation checkpoint на frozen test10 и SHA export.
Статус /workspace/exports/cardiac_chd68_gpu_full_v1_20261009_status.json;
log с тем же префиксом .log. GPU-вычисления подтверждены: 67%, 3188 MiB.
Не запускать второй процесс и не обновлять remote execution HEAD во время run.

Heartbeat cardiac-chd68-v1: после завершения скачать, проверить SHA всех critical
results, подготовить сравнения и update docs/metadata/push; затем отключить
heartbeat. Local backup root на E: remote_runs/cardiac_chd68_v1_full_20261009.
Full train/test metrics ещё отсутствуют; Pod пока не останавливать.

## CHD68 multiclass baseline v1 завершён — 2026-10-09

Full training30epochs117.74min, execution d89ea28, best epoch23/val0.8021.
Одна test evaluation10cases на original release grid: mean macro Dice0.7897,
median0.8205; LV0.8298 RV0.7881 LA0.8384 RA0.7975 MYO0.8419 AO0.7474 PA0.6847.
Диапазонcases0.5794–0.8783; worst ct_1083 сильно путает PA/AO/камеры.
Это публичный pilot без подтверждения age/physical scale/patient independence,
не neonatal/clinical validation и не результат на НИИ CT.

Все39 critical results на E:/3d-heart-data/pediatric_ct_heart/remote_runs/
cardiac_chd68_v1_full_20261009/artifacts SHA-verified; best/last separately verified.
Images в visual_QA_v2: bestct_1032/upper-medianct_1026/worstct_1083,
orthogonal GT/pred/error, PA failure, learning curves и metrics chart.
Helper scripts/qa_cardiac_predictions.py; source CT/GT/pred SHA unchanged.
Heartbeat cardiac-chd68-v1 PAUSED, GPU process завершён. Pod можно остановить.
Новых trainings/tuning после просмотра test не было. Frozen protocol не менять.
[Результаты и следующий этап](cardiac_chd68_results_v1.md).


## Подготовка resolution384 v2 — 2026-10-09

По запросу пользователя подготовлен single-factor вариант: XY256 ->384,
прочие architecture/normalization/loss/optimizer/sampling/seed сохранены.
48train/10validation из original public CHD68, test sealed и отсутствует в bundle.
Cache10.925GB, train samples12381/epoch. CPU real-cache3updates loss3.3290->2.9850,
weights changed; 179tests passed, source/256/384 QA двух train cases просмотрена.
Portable bundle7.847GB, 217payload SHA-verified плюс gzip CRC; включает только
validation reference v1 для original-grid сравнения, не initialization.
На E сохранён запас более80GB. Pod выключен, подключения/GPU benchmark/full training
нового варианта не было; прирост Dice и время v2 пока неизвестны.
[Resolution384 протокол и RunPod runbook](cardiac_chd68_resolution384_v2.md).


## Разрешён новый full run, preflight — 2026-10-10

Пользователь включил Pod и разрешил обучение resolution384. Актуальный endpoint
213.173.109.80:15827, root, только прежний ключ. Старый port12476 закрывал SSH
до authentication; port15827 publickey authentication успешно. Для нового endpoint
host key сохранён, последующие connections strict=yes. Windows ssh-keyscan оказался
несовместим с предложенным sntrup761 KEX; обычный ssh curve25519 подключился успешно.

RTX4090/torch2.8.0+cu128/CUDA подтверждены; persistent /workspace mount подтверждён.
Прежние v1 repos/data сохранены. На volume по du занято около18.41GB.
Новый archive7.847GB + unpack11.733GB +10GBreserve дают минимум48.0GB
до дополнительного environment/results. Для сохранения старых данных нужен volume
минимум50GB, лучше60GB. df общего FUSE pool не показывает личную quota.
Вопрос о выделенном размере отправлен пользователю; mass transfer/GPU benchmark/
full training до подтверждения capacity не запускались. Данные не удалялись.

Готов scripts/runpod_cardiac_resolution_full.sh: pinned commit, exclusive lock,
refusal of reused prefix/output, train30scratch, v1/v2 original-grid validation,
paired metrics, SHA export. Test не оценивается. Comparator проверяет одинаковые
split/cohort/cases/GT voxel counts и original grid. 185 tests passed;17focused passed;
bash -n passed. Actual execution commit будет зафиксирован при benchmark/launch;
не обновлять checkout между ними. [Preflight/launch metadata](../metadata/pediatric/cardiac_chd68_resolution384_launch_v2.json).

## Resolution384 launch preparation — 2026-10-10

Volume увеличен пользователем до120GB. SCP передаёт verified7.847GB public train/validation bundle. Remote cleanfeature checkout `/workspace/3d-heart-resolution384-v2` pinned948c4bb70da7c9f494e7dbfe84bcf09c8446271b. Preparation PID9540 ждёт завершения передачи, затем SHA/unpack/verify/GPU benchmark; full30epochs разрешены после sanity. Временная heartbeat обновлена на новый port15827 и разрешена пользователем; testsealed. Prefix `/workspace/exports/cardiac_chd68_resolution384_v2_20261010`; читать `_preparation_status.json` до full runner и `_status.json` после него. Не дублировать работающие процессы.

Fullresolution384 runner started04:48:22UTC, PID9970; trainPID10077, state training. Actualbenchmark0.34384s/batch, estimated30train epochs2h13m excludingvalidation/save, technicalsanitypassed. 217payloadSHA verified; benchmark backed up onEwithSHA. No newtest evaluation. See resolution384 report and launchmetadata for exactrun/prefix/pin. Monitoractive, disableonlyafter verifiedresults/report.

## Resolution384 v2 завершён и сохранён — 2026-10-10

30epochs scratch, commit948c4bb, best epoch10; trainer9194.41s (2h33m14s).
Paired original-grid validation10: v1 macroDice0.797833 → v2 0.795231,
Δ−0.002602;4cases improved/6worsened. PA0.6055→0.6340, остальные6meanDice ниже.
V2 не заменяет v1. Ct_1099 PA0.2398; ct_1125 PA recall0.2285;
ct_1056 macro0.7735→0.7370. Test v2 не оценивался, split/config не менялись.

64/64results SHA проверены и safe unpack на E, best/last дополнительно verified.
Backup remote_runs/cardiac_chd68_resolution384_v2_20261010/artifacts;
QA одинаковых CT/GT/v1/v2 slices в visual_QA: ct_1036/ct_1079/ct_1027.
GPU-процесс завершён; Pod можно остановить. НИИ CT не отправлялись.
Age/patient independence/physical scale CHD68 не подтверждены;
это exploratory validation, не neonatal/clinical accuracy.
[Итог, ресурсы и ограничения](cardiac_chd68_resolution384_v2.md).
[Machine summary](../metadata/pediatric/cardiac_chd68_resolution384_results_v2.json).
Новый full training/test/tuning не запускать автоматически.
