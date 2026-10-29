# Heart3D Annotation — отчёт о работающем MVP

Продолжение от 23.10.2026: состояние первоначального синтетического MVP ниже сохранено как история. Первый локальный клинический эксперимент и новые возможности редактора описаны отдельно в [FIRST_CLINICAL_CT_INFERENCE.md](FIRST_CLINICAL_CT_INFERENCE.md).

Дата: 22–23.10.2026. Репозиторий: `C:\Users\Andre\Desktop\3D-heart\repository`.

## Результат

Первый вертикальный сценарий реализован и проверен в Chrome через Playwright: вход → назначенное синтетическое исследование → три MPR → полигон → редактирование точек → маска → сохранение → выход → повторный вход → повторное редактирование восстановленного контура → reload → экспорт NIfTI → существующая 3D-реконструкция.

Это локальный инструмент ручной разметки. Семь классов доступны для выбора; автоматическая сегментация семи структур и обучение в эту итерацию не включались. Клинические CT и фотографии заключений не использовались. В основной локальной БД создан один пользователь `resident`, один синтетический проект; другие аккаунты существуют только в изолированных тестовых БД под `outputs`.

Архитектура, сравнение Cornerstone/OHIF/vtk.js/Slicer, лицензии, геометрия, безопасность, HTTPS, деидентификация и backup описаны в [WEB_ANNOTATION_ARCHITECTURE.md](WEB_ANNOTATION_ARCHITECTURE.md).

## Запуск на этом компьютере

Из PowerShell:

```powershell
Set-Location 'C:\Users\Andre\Desktop\3D-heart\repository'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Start-Annotation.ps1
```

Открыть `http://127.0.0.1:5173/`. Терминал держать открытым; Ctrl+C завершает frontend и созданный этим скриптом backend. ExecutionPolicy Bypass действует только для данного процесса; системные настройки не изменяются. Порты 8000/5173 должны быть свободны. Серверы проверки при завершении разработки остановлены; скрипт запускает их заново.

Отдельный запуск в двух терминалах:

```powershell
# Терминал 1, из repository
.\.venv\Scripts\python.exe -m uvicorn heart3d.web.app:create_app --factory --host 127.0.0.1 --port 8000 --no-access-log
```

```powershell
# Терминал 2
Set-Location 'C:\Users\Andre\Desktop\3D-heart\repository\web'
npm.cmd run dev
```

Локальные зависимости уже установлены. Для нового окружения сначала установить основные imaging-зависимости проекта, затем:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web-lock.txt
Set-Location web
npm.cmd ci
npm.cmd run build
```

Проверено: Windows, Python 3.13.3, Node 22.15.0, Cornerstone3D 5.13.1. WebGL нужен для трёх MPR; CUDA для этого интерфейса не требуется.

## Пользователь и администрирование без секрета в коде

В текущем стенде логин `resident`. Пароль сгенерирован криптографически случайно, в исходниках/документации не сохранён, в БД находится только Argon2id hash. Чтобы задать собственный пароль, из `repository`:

```powershell
.\.venv\Scripts\python.exe -m heart3d.web.cli reset-password --username resident
```

CLI попросит новый пароль скрытым вводом, минимум 12 символов, и отзовёт старые сессии. На чистом новом стенде создание первого ординатора вместе с синтетическим исследованием:

```powershell
.\.venv\Scripts\python.exe -m heart3d.web.cli init --username resident
```

Повторный `init` существующего пользователя не перезаписывает. `--password-stdin` предусмотрен для защищённого автоматизированного запуска; аргумента `--password` нет. Владелец ОС администрирует через CLI, публичной регистрации нет. Будущие роли:

```powershell
.\.venv\Scripts\python.exe -m heart3d.web.cli create-user --username reviewer --role surgeon
.\.venv\Scripts\python.exe -m heart3d.web.cli assign --username reviewer --project <project-uuid>
```

Полноценный UI хирурга не реализован. API review/approval уже проверен отдельно; создавать дополнительные реальные аккаунты в рамках этой итерации не потребовалось.

## Демонстрация

1. Войти, открыть карточку `P-SYN-001` / `S-SYN-001`. Показаны три физические MPR одного синтетического объёма 96×96×64.
2. Выбрать LV/RV/LA/RA/MYO/AO/PA, инструмент «Контур». Поставить хотя бы три точки и кликнуть первую для замыкания.
3. Перетаскивать точки; Shift+клик по линии добавляет точку, Ctrl+клик по точке удаляет. На время незавершённого контура выход блокируется; Esc отменяет его.
4. Выбрать контур справа и нажать «Применить контур». Цветная маска вычисляется сервером в исходной сетке. Уже применённый контур обновляет маску при редактировании.
5. Колесо/ползунок меняют срез; средняя кнопка перемещает, правая масштабирует. Есть отдельные инструменты pan/zoom/W/L, настройки окна, размера кисти, прозрачности и видимости классов.
6. Дождаться «Сохранено · vN» либо нажать «Сохранить». Автосохранение — через 2.2 секунды без изменений. Кисть/ластик и undo/redo работают с тем же документом.
7. Выйти, войти заново и открыть исследование; точки и маска восстановлены. Выбрать сохранённый контур и снова переместить точку. Reload открывает текущее исследование по URL.
8. «Экспорт NIfTI» скачивает последнюю сохранённую версию; при несохранённых изменениях ссылка отключена. Маска uint8, классы 0..7, геометрия исходного CT.
9. «Передать на проверку» фиксирует текущий head как submitted и закрывает редактирование. Для продолжения понадобится назначенный surgeon с решением needs_revision через API. Утверждённая версия остаётся неизменяемой.

Снимок реального браузерного теста: `outputs/web_annotation_editor.png`. Рисунок теста демонстрирует инструменты, а не анатомически корректную экспертную разметку.

## Что проверено

| Проверка | Результат |
| --- | --- |
| `tests/test_web_annotation.py` | 10 passed: сессии, Argon2id, logout, CSRF, Origin, чтение/запись чужого исследования, Secure cookie, истечение сессии, версии/409, review/approval, CT integrity, NIfTI, backup restore, все 3 оси, rotated affine и fractional points. |
| Existing core regression | 23 passed: `test_annotation`, `test_geometry`, `test_independent_geometry`, `test_multiclass`. |
| Playwright Chrome | 1 полный E2E passed, 35.1 s (44.8 s с запуском стенда). Настоящие pointer/key события: создание/drag/add/delete, undo/redo, brush/erase, pan/zoom, autosave, relogin, редактирование восстановленных точек, reload, download, 401 и 404 для чужих объектов. |
| Внешняя сеть | E2E блокирует всё вне local origin и требует нулевой список попыток. Финальный прогон прошёл с 0 внешних запросов. |
| Production build | `npm.cmd run build` прошёл: TypeScript + Vite. Есть предупреждение о большом scientific viewer bundle (~3.68 MB JS, ~1.03 MB gzip). |
| Скрипт запуска | Backend health ok, frontend HTTP 200, доступ только loopback. |
| Экспорт и 3D | Браузерный экспорт совпал по SHA-256 с сохранённой серверной версией v5. Existing `heart3d build --preview` построил LV: 604 vertices / 1204 triangles; RV: 268 / 532. |

Команды тестов:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web_annotation.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_annotation.py tests/test_geometry.py tests/test_independent_geometry.py tests/test_multiclass.py -q
Set-Location web
npm.cmd run test:e2e
npm.cmd run build
```

E2E создаёт изолированный synthetic root под `outputs/web-e2e-private/<random>`, случайный тестовый пароль только в environment дочерних процессов, и собственные серверы 8001/5174. Нужен установленный Chrome; конфигурация использует `channel:'chrome'`. Рабочий пользователь и его БД не изменяются тестом. Тестовые traces могут содержать тестовые cookie; в Git они исключены.

Артефакты проверки:

- `outputs/web_backend_tests.xml`, `outputs/web_backend_tests.log`;
- `outputs/web_core_regression_tests.xml`, `outputs/web_core_regression_tests.log`;
- `outputs/web_e2e_results.json`, `outputs/web_e2e.log`, `outputs/web_build.log`;
- `outputs/web_annotation_export.nii.gz`, `outputs/web_annotation_verification.json`;
- `outputs/web_annotation_reconstruction/report.json`, `slices.png`, `preview.png`, `meshes/`.

Проверенная синтетическая пара:

```text
CT SHA-256:   bdb9989ab4c52855c6a398c0fe316d593792e05d204c7024326df0cf352eb3e7
Mask SHA-256: 42c844e45dcbd1e51412c9b15044fecb06587ed485f8cfaff0f7128d07be4e1a
```

Команда повторной реконструкции, используя пути из артефакта (новый пустой output обязателен):

```powershell
$pair = Get-Content outputs/web_annotation_verification.json -Raw | ConvertFrom-Json
.\.venv\Scripts\python.exe -m heart3d build --ct $pair.ct --mask $pair.export --out outputs/web_annotation_reconstruction_repeat --preview
```

Это два рисунка на одном срезе, поэтому 3D выглядит как тонкие пластинки/дуга. Отсутствие классов 3..7 в этом тесте ожидаемо. Реальная объёмная ручная сегментация потребует разметки многих срезов.

## Исправления, выявленные проверками

- Affine round trip мог терять граничные pixels polygon; добавлено устранение только численной погрешности и regression на трёх осях/повёрнутой сетке.
- Метаданные отдельных local-volume slices Cornerstone дополнены корректными физическими origin.
- Add/delete native callbacks не всегда порождают ANNOTATION_MODIFIED; завершённые действия считываются через штатные mouse events.
- Hydrate получал общий изменяемый массив с DTO: перемещение восстановленной точки не определялось как изменение. Передаётся независимая копия; повторное редактирование после relogin включено в E2E.
- Для Vite добавлены локальные браузерные `events`/`url`, необходимые транзитивному xmlbuilder2. Ошибок JS в финальном E2E нет.

## Ограничения и следующий пилот

1. Данные только синтетические. Импорта настоящих DICOM/NIfTI, проверки burned-in и медицинской валидации UI нет. Общая модель данных не означает допуск реальных исследований.
2. HTTP ограничен loopback. HTTPS-шаблон подготовлен, но сертификаты, private network, reverse proxy и удалённый доступ не развёрнуты и не проверены. До пилота обязательны инфраструктурная и правовая проверки из architecture doc.
3. При ранних HTTP-тестах Kaspersky внедрял свой скрипт и расширял CSP внешним доменом. Пользователь вышел из Kaspersky; свежий финальный браузерный прогон показал 0 внешних попыток, отдельный HTTP-запрос подтвердил отсутствие вставки и исходный CSP. Глобальные настройки антивируса агент не изменял. Перед клиническим пилотом требуется согласованная политика endpoint inspection и проверка доверенной цепочки HTTPS.
4. Сохранены замкнутые контуры; незавершённые не переживают crash. Есть предупреждение, но последние несохранённые секунды при аварии потеряются. Undo/redo не переносится между сессиями; camera/window settings также не сохраняются.
5. Нет отверстий, contour interpolation, livewire/region growing, topology QA UI, совместного редактирования, merge и surgeon UI. При конфликте 409 документ остаётся в редакторе; автоматического объединения нет.
6. Объём целиком загружается в память браузера, sparse patches и документы ограничены размером API. Большие клинические серии, arbitrary oblique/sheared grids, мобильные устройства и длительные нагрузки не валидированы. Основной target — desktop Chrome с WebGL.
7. Версии неизменяемы через API; владелец диска/БД технически может их изменить. До пилота нужны ACL, шифрование, управление backup/restore, независимый audit и эксплуатационный контроль.
8. Известные предупреждения тестовых библиотек: Starlette/httpx deprecation и scikit-image/NumPy shape deprecations. Они не вызвали падений. Полный suite ML/обучения не запускался; проверен относящийся к изменению imaging regression.

Перед ординаторским пилотом: независимая проверка геометрии на разрешённых деидентифицированных CT (несколько ориентаций), сравнение экспортов с Slicer, UX-проверка врачом, разметка полного объёма, управляемая HTTPS-среда, доступ/роли, утверждённый процесс deidentification/хранения/резервирования и подтверждённые правовые основания. После первого рабочего вертикального сценария расширение функциональности остановлено.

## Созданные файлы

Backend: `heart3d/web/__init__.py`, `db.py`, `imaging.py`, `app.py`, `cli.py`.

Frontend: `web/package.json`, `package-lock.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `src/api.ts`, `src/viewer.ts`, `src/main.tsx`, `src/style.css`.

Проверки: `tests/test_web_annotation.py`, `web/playwright.config.ts`, `web/e2e/setup.ts`, `web/e2e/annotation.spec.ts`.

Запуск и документация: `scripts/Start-Annotation.ps1`, `deploy/Caddyfile.annotation.example`, `requirements-web.txt`, `requirements-web-lock.txt`, этот отчёт и `docs/WEB_ANNOTATION_ARCHITECTURE.md`; дополнен `.gitignore`.

Локальные `.web-data` и `outputs` — рабочие/проверочные данные вне Git. Существующие материалы восстановления моделей (`MODEL_RECOVERY_REPORT`, `infer_cardiac.py`, `test_cardiac_inference.py`) сохранены без изменений. Commit/push и публикация не выполнялись.
