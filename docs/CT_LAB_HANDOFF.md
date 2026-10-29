# Продолжение Heart3D с другого ПК

Ветка: `feature/ct-lab-research-handoff`.

В Git переданы исходники CT Preprocessing Lab (этапы01–06), конфигурации экспериментов, исследовательские отчёты завершённых этапов, тесты и исходники редактора аннотаций. Датасеты, checkpoints, optimizer states, cache, `outputs/`, журналы, приватные клинические данные и SSH-ключи в коммит не входят.

## Получить код

```powershell
git clone --branch feature/ct-lab-research-handoff https://github.com/Andrew2002acmr/3d-heart.git
cd 3d-heart
```

Если репозиторий уже клонирован:

```powershell
git fetch origin
git switch --track origin/feature/ct-lab-research-handoff
```

Если локальная ветка с таким именем уже существует, достаточно `git switch feature/ct-lab-research-handoff`, затем `git pull --ff-only` при чистом рабочем дереве.

## Состояние этапа06

- Аудит historical CHD68 v1/v2 и сравнение с ROI05 реализованы. Frozen split:48 train /10 reused development /10 sealed test. Test не открывать.
- Реализованы исходный baseline, одинаковый evaluation adapter, checkpoints по optimizer steps, sampling/loss/subset-конфигурации и Training Dynamics Lab.
- Один локальный baseline **приостановлен по просьбе пользователя** на исходном ПК. Его checkpoint и результаты намеренно не переданы в Git. С нового ПК этот локальный запуск продолжить нельзя без отдельного переноса состояния; Git содержит только программу и конфигурации.
- Полный baseline ещё не завершён, итогового `TRAINING_DYNAMICS_GENERALIZATION_06.md` пока нет. Не объявлять воспроизведение завершённым по промежуточным метрикам.
- Дополнительные seeds, sampling/loss/data-volume эксперименты подготовлены для **ручного** RunPod. Код не создаёт платную инфраструктуру. Не запускать всю матрицу автоматически и не возобновлять локальную работу без команды пользователя.

## Уже подготовленный RunPod

На использованном Network Volume сохранена отдельная папка:

```text
/workspace/heart3d_training_dynamics06_20261029/
  pairs/                 # только58 public train/development пар =116 файлов
  project/               # проверенный код и отдельное .venv
    inputs/checkpoints/  # неизменённые historical v1/v2
    outputs/training_dynamics_06/
  incoming/              # manifest, логи и проверка готовности
```

Перед выключением pod независимая проверка подтвердила116 source SHA256 +116 cache SHA256,58 подготовленных случаев/0 test,14 unit tests, CPU/GPU smoke FP32 batch16, неизменность7 защищённых файлов. GPU — RTX4090, Python3.12.3, PyTorch2.8.0+cu128. Полные удалённые обучения не запускались. Pod мог быть остановлен пользователем; не считать недоступность SSH ошибкой исследования.

Для работы использовать тот же Network Volume. Актуальные адрес/порт подключения взять из RunPod **Connect** после запуска. SSH-ключ нового ПК добавить через **Credentials → SSH Public Keys**; приватные ключи не помещать в Git.

Открыть сохранённый каталог в терминале pod:

```bash
cd /workspace/heart3d_training_dynamics06_20261029/project
source .venv/bin/activate
export DYNAMICS06_DATA_ROOT=/workspace/heart3d_training_dynamics06_20261029/pairs
export DYNAMICS06_WEIGHTS_ROOT="$PWD/inputs/checkpoints"
export DYNAMICS06_OUTPUT_ROOT="$PWD/outputs/training_dynamics_06"

# Только проверить/показать план; модель не обучается.
python -m ct_lab.dynamics_suite --prepare-only
cat outputs/training_dynamics_06/runpod_readiness_verified.json
cat outputs/training_dynamics_06/runpod_environment_verified.json
```

Сохранённый `project/` развёрнут из проверенного архива и может не содержать `.git`. Не выполнять `git pull` в нём и не перезаписывать весь каталог поверх данных/cache. Для редактирования исходников использовать отдельный clone этой ветки; замороженные конфиги, manifest и результаты существующего запуска не смешивать между checkout. Особенно важны байты JSON: Git нормализует LF, а артефакты ранее создавались и на Windows. Аудит сохраняет неизменённые файлы без перезаписи, чтобы не менять SHA только из-за CRLF/LF.

Полные команды и budgets: [TRAINING_DYNAMICS_README.md](../ct_lab/TRAINING_DYNAMICS_README.md). Перед затратным запуском проверить выбранную группу, исторический бюджет23220 steps, исходные58-case inputs, fidelity gate и правила выбора checkpoint. Не устанавливать CPU PyTorch поверх готового CUDA PyTorch pod.

## Локальный интерфейс без переноса обучения

Для CPU-анализа на другом Windows ПК создать отдельное окружение с Python3.13:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt -r requirements-filtering.txt -r requirements-dynamics06.txt
.\.venv\Scripts\python.exe -m ct_lab.dynamics_launch
```

`requirements-ml.txt` устанавливает CPU PyTorch; это вариант для локального интерфейса и тестов, не для RunPod CUDA. Соседние старые папки `ct_preprocessing_deps`/`ct_feature_deps` не обязательны: библиотеки можно устанавливать в `.venv`. Для отображения КТ нужны отдельно полученные публичные NIfTI-пары и, для inference, нужные веса. Пустые графики без `outputs/` не являются нулевыми результатами: сохранённые измерения не входят в Git.

Диагностические браузерные скрипты используют Playwright из `web/node_modules`; зависимости устанавливаются `npm ci` в `web/`. Исходники React-редактора и его API находятся в `web/` и `heart3d/web/`; инструкции — в `docs/WEB_ANNOTATION_MVP_REPORT.md` и соответствующих документах редактора.

## Корректность исследования

Сохранять классы0–7, ignore255, frozen split и группу release-case; GT не подавать как вход модели. Scaler/weights/fit преобразований только по train. Prepared-grid и native-grid Dice не смешивать; train-monitor не включать в dev10. Development многократно использована и не является независимым test. Не интерпретировать неизвестные spacing/HU как подтверждённые физические единицы. Не менять historical v1/v2 или production preprocessing.

Проверки этапа06 с реальными артефактами требуют подготовленного окружения и выполняются по README на RunPod; свежий Git clone не содержит нужных данных и результатов. Последняя проверка на RunPod:14 тестов passed,0 failures/errors/skips. Локальное обучение и автоматический мониторинг оставлены на паузе.
