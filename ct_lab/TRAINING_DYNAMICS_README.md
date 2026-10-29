# Этап06: локальное исследование и ручной RunPod

Пользователь ограничил локальные обучения одним полным baseline. Остальная матрица подготовлена для ручного запуска на уже существующем RTX4090. Скрипты не создают pod, не вызывают provider API и не запускают платную инфраструктуру.

## Артефакты и ограничения

- Основной отчёт: `TRAINING_DYNAMICS_GENERALIZATION_06.md` после завершения baseline. Пока обучение идёт, промежуточный отчёт находится в `outputs/training_dynamics_06/TRAINING_DYNAMICS_GENERALIZATION_06_IN_PROGRESS.md`.
- Протокол: `configs/experiments/training_dynamics_06_baseline.json`, frozen `outputs/training_dynamics_06/protocol.json`.
- Матрица: `configs/experiments/training_dynamics_06_runpod.json`,33 jobs/301860 optimizer steps. Локально выполняется только первый baseline из этой матрицы.
- Исторические веса/классы0–7/ignore255/frozen split не изменяются. Test CT и маски не передаются и не открываются. Development повторно используется; это не независимая финальная оценка.
- Native-grid Dice и prepared-grid Dice различаются. Main reported native score — mean case-macro foreground Dice на10 development cases; train-monitor8 и ct_1001 показываются отдельно.
- Spacing/HU не подтверждены; HD95 в index-grid units, без мм. Нет удаления компонент при основном сравнении.

## Подготовленный перенос

`python -m ct_lab.dynamics_bundle` создаёт `outputs/training_dynamics_06/runpod_manual_bundle.zip`. Архив содержит allowlisted исходники, конфигурации, cohort/split metadata, экспортированные historical source snapshots, небольшие контекстные JSON и **копии** неизменённых v1/v2 checkpoints. Manifest SHA256 находится внутри; ZIP CRC проверяется при создании. Raw CT/GT, sealed test, `.git`, credentials и private clinical данные не включаются.

В отдельную директорию данных передать **только116 файлов** из `data_transfer_manifest.json`:58 train/development cases × image/label. Имена и SHA перечислены явно. Не копировать всю директорию с68 cases. Доступ к этим public данным на pod организуется вручную.

Лучше использовать отдельную директорию/checkout, чтобы remote результаты не перезаписали локальные. После распаковки корень архива является рабочей директорией Python. Проверка архива и data SHA:

```bash
python -m ct_lab.dynamics_bundle --verify .
export DYNAMICS06_DATA_ROOT=/workspace/chd68_train_dev/pairs
export DYNAMICS06_WEIGHTS_ROOT="$PWD/inputs/checkpoints"
python -m ct_lab.dynamics_bundle --verify-data
```

При необходимости задать абсолютный `DYNAMICS06_OUTPUT_ROOT=/workspace/results/dynamics06_remote`. По умолчанию результаты остаются в `outputs/training_dynamics_06` текущего checkout. Не направлять remote OUTPUT_ROOT в директорию локального RTX2060 run.

## Среда и smoke

Использовать готовый CUDA PyTorch образ с RTX4090; версия Torch/CUDA фиксируется в environment каждого запуска. Историческая среда — Python3.12.3/Torch2.8+cu128; локальный Stage06 — Python3.13.3/Torch2.14.1+cu130. Совпадение pipeline не означает bitwise совпадения между версиями/железом.

```bash
python -c "import torch; print(torch.__version__,torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
python -m pip install -r requirements-dynamics06.txt
python -m ct_lab.dynamics_audit
python -m ct_lab.dynamics_data
python -m ct_lab.dynamics_smoke
python -m ct_lab.dynamics_suite --prepare-only
```

`dynamics_data` готовит58 исторических F32 cache volumes; никакого test cache. Предусмотреть место под исходные данные,~4GiB cache и checkpoints/native masks (порядка десятков GiB для всей33-job матрицы; фактическое потребление зависит от компрессии). CPU smoke64/B2 и GPU256/B16 должны пройти до full run; при нехватке ресурсов runner не меняет batch или precision автоматически. Локально измерено~.752s/update и1.91GiB GPU peak; оценивать время/стоимость4090 по его собственному smoke.

## Последовательный ручной запуск

```bash
# Полные baseline seeds; первый обязательно проходит fidelity gate.
export DYNAMICS06_GROUP=baseline
bash scripts/run_training_dynamics06_runpod.sh

# После проверки baseline curves/results, вручную выбрать следующую группу:
python -m ct_lab.dynamics_suite --group sampling
python -m ct_lab.dynamics_topology --all-runs
python -m ct_lab.dynamics_suite --group data
python -m ct_lab.dynamics_topology --all-runs
python -m ct_lab.dynamics_suite --group losses
python -m ct_lab.dynamics_topology --all-runs
```

`--group all` существует, но означает явный запуск всей матрицы. Наличие конфигурации не запускает её. Основной runner требует сначала3 полных baseline для reference prefixes остальных3-seed сравнений. Если инженерный gate первого baseline (native Dice в±.03 от v1, полный budget,finite/decreasing loss) не пройден, downstream jobs останавливаются; анализировать причину вручную, не подбирать по test.

Полные baseline:23220 steps/371430 examples,48 cases, seeds20261009/10/11. Native controls480/3840/7740/15480/23220 плюс best prepared selected. Дополнительные группы —7740 steps/fixed last; исторический reference на7740 переиспользуется из baseline prefix. N8/16/24/32 вложены в train48, composition seed20261029; это один seed состава и три seed инициализации. Preprocessing clip остаётся исторически fitted на48 train — ограничение low-N интерпретации. Loss class weights fitted только на pixels выбранного train subset.

`last.pt` — атомарный mutable resume state; runner продолжает с него. `step*.pt`/`selected.pt` — отдельные контрольные веса. Не заменять `last.pt` между разными jobs и не переименовывать папки. Для повторного запуска с начала использовать новую OUTPUT_ROOT. Если переносить частично завершённый локальный run, копировать согласованную папку **после остановки процесса**; checkpoints с разным SHA не смешивать. Для чистого GPU сравнения предпочесть новый run; CUDA warn-only не гарантирует bitwise resume на другом железе.

## Диагностики, интерфейс и проверка

```bash
python -m ct_lab.dynamics_sampling    # train-only simulation, без model training
python -m pytest -p ct_lab.dynamics_pytest tests/test_training_dynamics.py -q --basetemp=outputs/training_dynamics_06/manual_test_tmp -o cache_dir=outputs/training_dynamics_06/pytest_cache
python -m ct_lab.dynamics_report     # промежуточный отчёт, только completed measurements
python -m ct_lab.dynamics_launch     # http://127.0.0.1:8510
```

`Training Dynamics Lab` содержит7 panels. Checkpoint Explorer меняет predictions на том же CT/native Z, показывает GT,FP/FN/confusion; checkpoint weights не записываются из UI. При переносе remote результатов без cache UI может выполнить CPU inference отдельного Z из raw CT. Для train/validation stats и графиков переносить `case_statistics.json`,metrics/topology/summary/history/config/environment вместе с weights. В директории данных должны быть SHA-согласованные public пары.

`dynamics_report --final` допускается после полного первого baseline и topology completion. Этот локальный исследовательский отчёт сфокусирован на одном baseline; для последующих remote групп CSV всех completed runs и UI показывают фактические значения, но текстовые выводы о seed/loss/data нужно обновить по новым результатам. Не выдавать шаблонные утверждения о локальном бюджете за описание уже выполненной remote матрицы.

`python -m ct_lab.dynamics_finish --wait` только ждёт завершения **уже запущенных** baseline/topology workers, затем выполняет fidelity check, финальный отчёт, SHA integrity и ZIP результатов. Этот helper не запускает обучение. Он подходит для текущего длительного локального запуска; файлы `completion.json` или `completion_error.json` фиксируют фактический исход. `local_baseline_completed_results.zip` содержит immutable checkpoints, native predictions, метрики и графики без raw CT/GT/cache и mutable resume weights.

После всех работ локальная `dynamics_integrity --final` сверяет116 raw-файлов, prepared cache, original weights/config/split и полные SHA snapshots артефактов01–05. На отдельном fresh RunPod checkout snapshots будут относиться к его исходному состоянию; это не замена проверке оригинального Windows workspace. При этом тесты новых loss/samplers не обучают дополнительные большие модели.
