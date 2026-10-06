# Точка продолжения: pediatric Heart baseline

Дата: 2026-10-06. CPU readiness завершён; затем подготовлен RunPod full pipeline.
**Full training разрешён после успешного GPU benchmark, но ещё не запускался.**
Актуальный cloud status: [pediatric_heart_runpod_v1.md](pediatric_heart_runpod_v1.md).
Блокер: сервер TCP endpoint отклоняет явно заданный public RunPod key. Пользователь
добавил key в account Credentials после boot; требуется restart/update authorized_keys.
Проверка CUDA/RTX4090/mount, transfer и benchmark ещё не выполнялись.
При restart уточнить новый endpoint. Не отключать host key checks и не использовать
другие приватные keys. Только локальный ввод passphrase, без передачи в chat.

Локально готовы caches **всех 60** (42/9/9), bundle201 files/2.35 GB + SHA manifest,
`runpod_transfer/frozen_v1_20261006.tar.gz`. Frozen config/split/cohort hashes прежние.
Full scratch trainer/best/last/evaluation/results export реализованы, **124 tests**.
New config: `configs/pediatric_heart_runpod_v1.json`; driver `scripts/run_pediatric_gpu.py`.
Benchmark50 measured +5 warmup +20 sanity → при pass один30epoch BCE+Dice fromscratch,
затем один test original-grid evaluation и SHA results bundle. Critical results
скачать на E и verify; только после этого подтверждать, что Pod можно остановить.

Ниже историческая запись предыдущего CPU readiness; текущие изменения указаны выше.
Readiness: [pediatric_heart_training_readiness.md](pediatric_heart_training_readiness.md).
Baseline: [pediatric_heart_segmentation_baseline.md](pediatric_heart_segmentation_baseline.md).

## Рабочее место

- Branch `feature/pediatric-heart-segmentation`, audit base66c5804.
- Worktree `C:/Users/Андрей/.codex/worktrees/pediatric-heart-segmentation/3d-hearts`.
- Data root `E:/3d-heart-data/pediatric_ct_heart`, ~281.48 GB free, reserve80 GB.
- Main D checkout/user changes/old D-data не трогать, ничего не удалять.
- Python `D:/codexProjects/3d-hearts/.venv/Scripts/python.exe`.
- External pydicom PYTHONPATH `D:/codexProjects/3d-hearts/data/pediatric_audit/python_deps`.
- torch2.14.1+cpu, psutil7.2.2 installed, requirements-ml.txt pinned.
- Existing inventory cache/tcia_series_v1.json, receipts/QA/fingerprints на E.

## Зафиксированный результат

327 metadata census, age2–16, healthy_status unknown. **117 complete CT/RT pairs**,
114 gates/visual reviews, 62 technical OAR approvals; 50 coverage_incomplete,
3 geometry_failure, 1 coverage_review, 1 identity_review. 234 archive hashes verified,
cache recovery завершён с preserved failures; OS root lock добавлен.
Exact HU audit114, duplicate groups0; unresolved similar identity pair не используется
как два patients. Original labels не исправлялись.

**Frozen cohort60** по20 age group, **42/9/9**, внутри age14/3/3, seed20261006.
configs/splits/pediatric_ct_heart_v1.json содержит cohort SHA. Development IDs вне
test; patients не пересекаются, frozen v1 не менять. Train scanner29 LightSpeed /
12 SOMATOM /1 Revolution; held-out6/3/0 каждый. Revolution/confirmed non-contrast
generalization здесь не проверяется.

Train-only42: HU[-1000,969]→[-1,1], full FOV256², Z2 mm, offsets[-4,-2,0,2,4] mm,
linearCT/nearestGT, original grid+inverse сохранены. Train mmapcache10280 slices:
1742 positive /8538 negative, balanced epoch3484 samples =1742 batches.
Three-age train context mosaic просмотрен. Held-out fitting/cache/evaluation не было.

Own2.5D U-Net16/32/64/128, GN8, 3down/up, 5→1, 488993 parameters, fromscratch.
BCE/softDice/combined supported, benchmark combined; CPU6 threads, batch2,
Adam0.001, seed20261006, no flips. **120 tests passed**.

Actual `cpu_b50_v1_20261006`:5 warmup +50 measured +20 fixed sanity =75 updates,
mean0.4150 sec/batch, 4.819 samples/s, peak sampled RSS624 MB, checkpoint5.95 MB.
Epoch~12.05 min; 30 training epochs~6.02 h **без validation/save overhead**.
Fixed2 train slices loss1.4544→1.0210→0.8787, finite gradients/weight updates, exit0.
Это sanity, не model-quality metric. Benchmark weights не final/pretrained baseline.
Summary metadata/pediatric/heart_cpu_benchmark_v1.json, actual code revision686fe95.
Поздняя provenance правка включает fixed IDs в all-used list; в данном run они
уже присутствовали, measurements/weights не изменены.

## Artifacts

- Small QA/cohort/reviews/split/train-stat/cache/benchmark summaries в Git.
- Native CT/GT: `<data_root>/<patient>/prepared/`.
- Train cache: `<data_root>/preprocessing/heart_v1/`.
- QA: `<data_root>/experiments/heart_baseline_v1/preprocessing_qa/`.
- History/config/environment/resources:
  `<data_root>/experiments/heart_baseline_v1/cpu_b50_v1_20261006/`.
- Technical checkpoint:
  `<data_root>/checkpoints/heart_baseline_v1/cpu_b50_v1_20261006/last_benchmark.pt`.
- Full training, val/test cache/evaluation, exported predictions/surfaces отсутствуют.

## Проверить без повторной загрузки

```powershell
$env:PYTHONPATH='D:/codexProjects/3d-hearts/data/pediatric_audit/python_deps'
$pythonExe='D:/codexProjects/3d-hearts/.venv/Scripts/python.exe'
$dataRoot='E:/3d-heart-data/pediatric_ct_heart'
& $pythonExe -m heart3d.ml.prepare --config configs/pediatric_heart_baseline_v1.json --data $dataRoot --partition train
& $pythonExe scripts/view_pediatric_ct_qa.py --data $dataRoot --reviews metadata/pediatric/heart_cohort_reviews_v1.json
# Только SHORT benchmark при желании: новое имя, без overwrite.
& $pythonExe -m heart3d.ml.benchmark --config configs/pediatric_heart_baseline_v1.json --data $dataRoot --batches 50 --run-name cpu_b50_new_unique
```

Не refresh approved reports/SHA, не пересоздавать v1 split/cohort. Не продолжать
full experiment из sanity checkpoint: initial seed/fromscratch. No main merges,
force push/reset/clean, D-data cleanup или auto loss sweeps.

## Следующий отдельно разрешаемый этап

CPU/RAM/storage и technical gates подходят для full **source Heart OAR** baseline.
После явного согласования реализовать full train/validation loop, best/last и
original-grid metrics. Один fixed experiment, test не используется для выбора
параметров. Затем untouched-test pass, failures age/scanner/spacing/coverage,
GT/Prediction viewer и binary Heart meshes. Clinical sign-off, multiclass/CHD
reasoning и СППР — отдельные задачи.
