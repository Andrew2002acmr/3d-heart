# Клиническая реконструкция сердца: целевой workflow 0–17 лет

Дата: 2026-10-09. Требования уточнены пользователем по разговору с хирургом НИИ.
Цель **0–17 лет**, младшие группы проверять отдельно.
Это план развития и критерии проверки; клиническая СППР ещё не реализована.
Frozen baseline v1 (2–16, Heart OAR) не меняется; его scores не переносятся
на newborns, камеры/сосуды и другую клинику.

## Возраст и исследовательский вопрос

| Группа | Рабочий интервал |
|---|---|
| Новорождённые | 0–27 полных дней |
| Младенцы после neonatal period | 28 дней — <1 года |
| Ранний возраст | 1–<2 лет |
| Дети | 2–5, 6–11, 12–17 полных лет |

Первые28дней — neonatal period по [WHO](https://www.who.int/health-topics/newborn-health).
Другие strata исследовательские. Общий диапазон age<18.
DICOM AS в неделях/месяцах не выдавать за точную дату рождения/chronological age;
нужен age evidence с точностью или подтверждённым интервалом на acquisition.

Вопрос: насколько воспроизводима геометрия хирургических структур при разных
age/protocol/motion/cardiac phases и сколько ручной правки нужно для заданной
точности? [SCCT CHD CT technical recommendations](https://pubmed.ncbi.nlm.nih.gov/26679548/)
— техническая научная опора для CT quality; программа не назначает scan/sedation.

## Цепочка работы

1. DICOM inventory; выбрать series и phase, проверить spacing/coverage.
2. Один physical volume; три связанные плоскости, window/contrast и source refs.
3. Предварительные masks моделью либо ручное начало. Нынешний Heart OAR baseline
   не выдаёт LV/RV/LA/RA/MYO/AO/PA.
4. Ручная правка выбранных структур и проверка соответствия CT.
5. Reviewed mask version с source/model/edit provenance.
6. Surface/topology/scale QA; согласовать blood pool, стенки, сосуды, cuts.
7. Отдельные exports для VR и согласованного процесса 3D печати.

Полуручной подход принят как требование продукта. Автомаска — предложение.
При неразличимой границе ручная правка также не доказывает anatomy: сохранять
uncertainty/review; smoothing не восстанавливает потерянную CT информацию.

## Минимальный редактор

Расширять существующий viewer последовательно:

- Структуры/цвет/видимость, axial/coronal/sagittal +3D preview.
- Paint/erase с размером в mm, draw/polygon, undo/redo на source grid.
- Preview region growing/fill-between-slices с подтверждением и отменой.
- Удаление выбранного island как явное действие, без always-largest-component.
- Raw prediction, edited и reviewed versions отдельно; автор/операции/model/phase.
- Multi-structure layers с согласованной overlap policy, без склейки LV variants
  или переименования их в MYO по внешнему виду.

Это проектируемые функции, не заявленная готовность текущего viewer.
[3D Slicer Segment Editor](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html)
подходит как reference UX/independent reviewer; собственная модель остаётся в проекте.

## Astra Linux и ClearCanvas

По сообщению пользователя хирург работает на Astra Linux и использует ClearCanvas.
Версии и способ запуска неизвестны. Официальный
[ClearCanvas repository](https://github.com/ClearCanvas/ClearCanvas) описывает
Windows/Visual Studio build; нельзя угадывать deployment конкретной клиники.

Планировать независимый DICOM import/export и portable core.
Frontend кандидаты: перенос существующего Qt/VTK либо local browser +local backend.
Выбирать после проверки Astra release, CPU/RAM/GPU, OpenGL/WebGL,
Qt/system libraries и offline install. Linux support не доказывает Astra compatibility.
[Slicer system requirements](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html)
дают ориентир для внешнего reviewer; на Astra он пока не проверялся.
PowerShell/Windows paths — только машина разработки. Облачная передача
клинических DICOM из этих требований не следует.

## VR и печать

Native volumes/masks сохраняются. RAS/LPS, scale/axes, landmarks и hashes —
в manifest. Результат содержит anatomy identity, source/phase, approval,
model/edit versions и mesh parameters.

Для VR рассмотреть GLB: separate structures, transparency, clipping.
[glTF specification](https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/Specification.adoc)
задаёт linear units meters; mm→m/axes преобразовать явно с тестами.
Для печати рассмотреть 3MF с unit=millimeter либо STL+unit manifest.
[3MF specification](https://github.com/3MFConsortium/spec_core/blob/master/3MF%20Core%20Specification.md)
содержит явные units.

Уточнить printer/process/polymer, cuts/wall thickness/scale,
минимальную воспроизводимую деталь и expert approval. Не угадывать свойства полимера.
Print gate: anatomy/coverage, components/boundary/nonmanifold,
self-intersections/normals, thickness/scale и review в print slicer.
Watertight topology не гарантирует cardiac anatomy.
VR/печать пока не являются реализованным clinical workflow.

## Запросить у НИИ

| Нужно | Для чего |
|---|---|
| Selected native CT series/phase | Исключить mixing scouts/MPR/фаз |
| Age/diagnosis/surgery и stable pseudonymous patient ID | Strata/leakage/clinical labels |
| Final masks камер/сосудов и annotation protocol | Многоклассовый GT и expert review |
| Masks/meshes ПО другой клиники с source/phase/units | Сопоставление на тех же CT |
| Purpose каждого переданного SEG | Отличить anatomy от VOI/state |
| Astra release/hardware, рабочий сценарий | Deployment/UX acceptance |
| VR/printing requirements, важные детали | Geometry/export acceptance |

Два studies — format samples, не statistical neonatal validation cohort.
Clinical/healthy labels не выводятся из отсутствия CHD.
Новые ML runs, CT/MRI mixing, classifiers и clinical logic не начинались.

[Аудит файлов](external_ct_import_audit.md),
[baseline resume](pediatric_heart_segmentation_resume.md).
