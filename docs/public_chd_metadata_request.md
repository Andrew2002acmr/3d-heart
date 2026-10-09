# Черновик запроса metadata авторам CHD68/ImageCHD

Письмо не отправлено. Адрес для вопросов указан в официальных репозиториях:
xiao.wei.xu@foxmail.com. При отправке не прикладывать клинические КТ НИИ.

Subject: Clarification of CHD68/ImageCHD release geometry and per-case metadata

Dear authors,

We are preparing a research baseline for pediatric cardiac CT segmentation
and audited the public CHD68 release linked from your repository.

All 68 image/label pairs have matching grids, but their NIfTI headers report
spacing 1×1×1 with unspecified spatial units. Could you clarify whether the
released arrays preserve the original voxel grid or were resampled, and provide
the per-case spacing, orientation and preprocessing transforms? Are image values
HU, or is an additional rescale slope/intercept needed?

The ImageCHD image-info spreadsheet contains acquisition/birth-date metadata,
PixelSpacing and calculate_z_thick. The rule case_id = 1000 + idx matches the
complete ImageCHD file-ID set. Is this the intended linkage? May these values
be used for the corresponding CHD68 files, or does that release represent a
different acquisition or grid?

Under this provisional linkage, ct_1016/1037/1062/1083/1103 have candidate ages
40/40/29/47/43 years, while the 2023 paper describes an age range up to 21 years.
Could you clarify the applicable cohort and provide verified per-case ages?
We have not applied these candidate values to a pediatric cohort.

Finally, could you confirm the annotation boundaries, particularly venae cavae
included in RA and pulmonary veins included in LA, and the interpretation of
additional labels outside 1–7?

Thank you for making these datasets available.
