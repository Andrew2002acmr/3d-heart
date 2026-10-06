# Pediatric Heart baseline: cohort QA и вычислительная готовность

Начато 2026-10-06 в `feature/pediatric-heart-segmentation`. Full training запрещён
текущим заданием: разрешены полный data gate и короткий CPU benchmark после split.
Все raw/derived/ML artifacts находятся под CLI-selected data root на E, reserve 80 GB.

## Target и критерии до отбора

Target v1 — **исходный экспертный Heart OAR**, как определён RTSTRUCT источника.
Это общая Heart область; анатомическую полноту камер/сосудов не утверждаем.
[Публикация](https://pmc.ncbi.nlm.nih.gov/articles/PMC9090951/) подтверждает
экспертные OAR contours и исходную QA radiation oncologist. Точный cranial landmark
для Heart в статье не задан. Поэтому ровная planar граница сама по себе не ошибка:
она сохраняется как scope исходной аннотации, если references/planes корректны,
ROI непрерывен и scan не обрезает очевидную Heart область. Неоднозначность
аннотации оставляется `annotation_scope_review`, не достраивается автоматически.

`approved` означает техническую и визуальную пригодность для воспроизведения
source OAR protocol, не новую clinical annotation sign-off или healthy label.
Для всех пациентов healthy_status остаётся unknown.

Полный CT gate: исходные SOP inventory, возраст/ID/series/frame, IPP/IOP/spacing,
HU tags, regular grid, independently sorted GDCM/SimpleITK geometry + HU.
RT gate: original full RT, Heart definitions/nonempty contours, all source references,
FOV/plane residual, rasterization и independent VTK boundary agreement.

Review classes: `approved`, `coverage_incomplete`, `coverage_review`,
`annotation_scope_review`, `reference_failure`, `empty_roi`, `geometry_failure`.
Automatic technical pass никогда не означает visual approval. Scan face contact,
margin <=1 CT slice, внутренние пропуски или несколько components требуют review.
Необычный contour count, объём или spacing — признаки для внимательного просмотра,
а не автоматическое исключение по размеру/возрасту.

## Sampling

Заранее фиксирован seed 20261006. Начальные scanner квоты — existing preparation
config; replacements заранее упорядочены SHA256(seed|patient_id), round-robin strata.
Volume/красота маски не входят в selection. Metadata reference/empty failures
отсеиваются до full download. Цель 20/20/20 по рабочим age groups, не любой ценой.
Девять прежних engineering cases повторно проверяются; из будущего test исключены.

QA sampling v2: axial first−2, first, first+2, median active, last−2, last, last+2
(clamp к CT grid), native coronal/sagittal median active. Отдельные CT и
CT+mask panels, source contour overlay на axial. Original spacing aspect, без GT
interpolation. Скриншоты вне Git. Review описывает coverage, annotation scope,
alignment, причины и reviewer type.

## Текущее состояние

Approved cohort, frozen split, train statistics и benchmark ещё не сформированы.
Числа будут записаны из реальных QA и измерений. Estimate duration до benchmark
не публикуется. Full training/test evaluation не запускать.
