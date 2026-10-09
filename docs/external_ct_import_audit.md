# Аудит внешних клинических CT и DICOM SEG

Дата: 2026-10-09. Ветка: `feature/pediatric-heart-segmentation`.
Аудит формата, геометрии и переносимости reconstruction pipeline.
Клиническая разметка не утверждена; новые training/inference runs не запускались.
Подробные case-level manifests и изображения остаются вне Git.

## Состав и разделение данных

В плоском каталоге 5273 DICOM файла, 3.01 GB: 5241 CT, 24 SEG, 8 SR;
2 StudyInstanceUID и 50 SeriesInstanceUID.
RTSTRUCT и исходных Surface Segmentation Storage в этой поставке нет.
SEG — BINARY Segmentation Storage. CT включает scouts, bolus monitoring,
MPR, изображения протокола/ECG и производные volume-rendering captures.

Разделение логическое: StudyInstanceUID → SeriesInstanceUID → SOP UID.
Оригинальные файлы не перемещались, не переименовывались и не изменялись.
Имена файлов и InstanceNumber не определяют порядок срезов.
Aliases воспроизводимы для одного полного inventory; добавление новых UID
может изменить их порядок. Для новых поставок создавать отдельный audit version.

Оба studies по PatientAge относятся к младшим группам до 2 лет.
Они нужны для проверки нового domain, а не для изменения frozen baseline v1.
Точные age tags/series locators доступны в локальном отчёте.

PatientName/PatientID/PatientBirthDate заполнены: это не доказывает реальность
идентификаторов, но подтверждения обезличивания нет. Один distinct PatientID
в двух studies не позволяет утверждать, что это два независимых пациента.
Clinical/healthy status = unknown. Имена, даты, исходные UID и изображения
в Git не добавлялись.

## CT gate и выбор серии

13 classic CT серий прошли header geometry; 5 требуют review.
Отказ single-volume gate не обязательно означает повреждение: встречаются
несколько фаз с повторением IPP, bolus monitoring и смешанные производные MPR.
Многофазные серии автоматически не разделялись. Из SeriesDescription/ProtocolName
экспортируются только числовые percentage tokens, не свободный текст.

Две выбранные single-phase серии реально decoded/exported:
512×512×251 при 0.22715×0.22715×0.5 mm и
512×512×158 при 0.26491×0.26491×0.6 mm.
У обеих SliceThickness=0.8 mm, что не является фактическим Z spacing.
Engineering selection не означает выбор оптимальной clinical cardiac phase.

Проверены IOP/IPP, regularity, SOP/Series/Frame, dimensions, PixelSpacing,
rescale tags. Порядок: IPP · cross(IOP_column, IOP_row).
Массив (column,row,slice), DICOM LPS; NIfTI явно RAS:
diag(−1,−1,1,1) × affine_LPS. Original grid без resampling.
Независимый SimpleITK/GDCM подтвердил порядок SOP, affine и decoded values:
максимальная разница 0. Header geometry не доказывает полноту исходной acquisition
за пределами поставки; отдельного PACS inventory нет.

RescaleSlope=1, RescaleIntercept=−8192; корректно применены.
Экстремумы необычно широкие. Не переносить автоматически baseline v1 clip
[−1000,969] и resolution на младенческую многоклассовую задачу.
Отдельно исследовать anatomy ROI distribution, padding/artifacts и contrast.
Independent decode agreement не заменяет scanner calibration/clinical QA.

## DICOM SEG → original CT grid

`heart3d.dicom.segmentation.map_binary_seg` поддерживает BINARY Storage:
Study, FrameOfReference и source series должны совпадать.
ReferencedInstanceSequence сверяется с CT SOP inventory.
Для каждого frame проверяются segment number, dimensions, IOP, PixelSpacing,
origin/plane residual≤0.05 mm и наличие CT SOP для соответствующей плоскости.
Per-frame SOP references, когда есть, должны совпадать с physical plane.
Дубли segment/slice не объединяются молча.

У этих Siemens SEG source SOP references глобальные/в Shared Derivation;
per-frame Derivation references отсутствуют. Сопоставление — по проверенному
global SOP inventory и per-frame PlanePosition/Orientation/PixelMeasures,
а не ordinal frame index. Отсутствующие плоскости — нули, без interpolation.

22/24 объектов импортированы технически (21 непустой segment и 1 пустой algorithm state); 2 оставлены `requires_review`
из-за FrameOfReference mismatch. References автоматически не исправлялись.
Каждый source object/segment сохраняется отдельно с label/algorithm type,
SHA-256 и CT geometry. Одинаковые labels не объединяются в один GT class.
[DICOM Segmentation Image Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.2.html).

## Значение масок и SR

Название Heart/LV и SEMIAUTOMATIC не являются expert approval.
В одном study есть Heart, Left Ventricle, Blood Pool, Aorta,
coronary/private algorithm state labels; в другом четыре
Table/TableRemovalAlgorithm masks, не идентифицированные как Heart GT.

Визуально просмотрены axial/coronal/sagittal CT и четыре SEG overlays.
LV labels дают крупные кольцевые/эллиптические области вокруг сердца;
один Heart крайне мал, другой охватывает большую область грудной клетки.
Возможны промежуточные VOI/algorithm states и display masks.
Назначение нужно уточнить у экспортировавшей стороны.
`ground_truth_certified=false`, `expert_approval=unknown`.

8 SR surveyed по coded concept names/ValueTypes: radiation dose/protocol report,
measurement/navigation/session-like objects, другие без ContentSequence.
TEXT/PNAME/DATETIME/UID values не экспортировались. Диагноз не определялся;
отсутствие диагноза не означает healthy.

## Mask → mesh

Для четырёх source variants одной CT series построены VTP/STL существующим
pipeline без smoothing, decimation, component removal.
Binary segment1 явно source label, не ImageCHD LV1. RAS/mm provenance сохранена.
STL numeric coordinates в mm сопровождаются manifest.

Есть nonmanifold edges у двух mesh, border contact у ряда masks и несколько
components. Self-intersections не проверены. Даже закрытый mesh служебной ROI
не становится анатомической моделью сердца. Это проверка переносимости
mask→mesh, а не готовность к печати или хирургическому применению.

## Воспроизведение

Пути задаются CLI; в Python диск E не зашит. Резерв 80 decimal GB.
Оригиналы не удаляются.

```powershell
$dataRoot='E:/3d-heart-data/pediatric_ct_heart'
$auditRoot="$dataRoot/external_ct/audit_v1"
$env:PYTHONUTF8='1'
# Окружение с requirements-pediatric.txt.
python -m scripts.audit_external_ct --input "$dataRoot/external_ct/incoming" --output $auditRoot
python -m scripts.audit_external_ct --output $auditRoot --prepare-series case_001/series_004 --prepare-series case_002/series_010
python -m scripts.import_external_seg --audit $auditRoot --mesh-series case_001/series_004
python -m scripts.qa_external_seg --audit $auditRoot --ct-series case_001/series_004 --objects case_001/series_005/object_001 case_001/series_005/object_002 case_001/series_005/object_003 case_001/series_005/object_008
python -m pytest -q
```

Aliases примера только для этого inventory; на других данных выбирать series
по audit/references/clinician review.
PNG — native grid, physical aspect, image-only center sampling, engineering preview.
GUI review SEG в независимом Slicer пока не выполнялся; независимая CT проверка
выполнена SimpleITK.

Local outputs: `audit_summary.json`, `source_locators.private.json`,
`segmentation_inventory.json`, `sr_structure_inventory.json`,
`seg_import_results.json`, `final_integrity.json`, `prepared/`, `imported_seg/`,
`visual_qa/`, `clinical_import_report.md`.
Private locators с paths/UID/hash остаются в data root.
Повторные SHA всех5273 совпали, новых/пропавших/изменённых файлов нет.
Свободно ~274 decimal GB, резерв80GB сохранён.

## Проверки и следующий шаг

Synthetic tests: grouping/immutable sources, omission identity fields,
optional demographics без ослабления geometry/HU gate, duplicates/missing slices,
binary decode, physical frame reordering/voxel landmarks, reference failures,
phase evidence и SR privacy. Итоговый suite/Windows locale caveat — в resume.

Получить от хирурга selected CT series/phase, список структур, purpose SEG,
final reference masks/3D models из ранее использованного ПО, подтверждение
case identity/diagnosis. Сообщение об обработке этих CT другой клиникой
не заменяет её exported reference artifacts с provenance.
Данные пригодны для engineering/reader/manual-workflow. Training GT и clinical
reconstruction accuracy на них ещё не подтверждены.

[Цель 0–17 и полуручной workflow](clinical_reconstruction_workflow.md).

[Aggregate machine-readable summary](../metadata/pediatric/external_ct_import_audit_v1.json).
