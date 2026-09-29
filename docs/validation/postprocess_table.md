# Postprocessing measurements

Coordinates have unknown units: area is in coordinate units squared, volume in coordinate units cubed.
A dash denotes invalid topology; even other volumes have not been checked for self-intersections.
Deviation uses 10,000 samples per direction, not exact Hausdorff distance.

| Case / structure | Variant | Vertices | Triangles | Area | Volume | Mean deviation | Sampled max | Non-manifold edges |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ct_1001 LV | original | 88234 | 176536 | 66540.570 | 505783.833 | 0 | 0 | 0 |
| ct_1001 LV | taubin | 88234 | 176536 | 61552.779 | 505754.510 | 0.0774857 | 0.417391 | 0 |
| ct_1001 LV | decimate50 | 44100 | 88268 | 66540.781 | 505783.792 | 2.0581e-06 | 0.0267518 | 0 |
| ct_1001 LV | taubin_decimate50 | 44100 | 88268 | 62183.500 | — | 0.090169 | 0.728872 | 2 |
| ct_1001 MYO | original | 318839 | 638366 | 241387.661 | — | 0 | 0 | 12 |
| ct_1001 MYO | taubin | 318839 | 638366 | 224591.355 | — | 0.0737625 | 0.356225 | 12 |
| ct_1001 MYO | decimate50 | 159247 | 319182 | 241387.661 | — | 6.72086e-15 | 9.09939e-14 | 12 |
| ct_1001 MYO | taubin_decimate50 | 159247 | 319182 | 225756.547 | — | 0.081212 | 0.572098 | 12 |
| ct_1001 PA | original | 265259 | 530650 | 198544.239 | 864653.375 | 0 | 0 | 0 |
| ct_1001 PA | taubin | 265259 | 530650 | 184261.329 | 864262.187 | 0.077885 | 0.390154 | 0 |
| ct_1001 PA | decimate50 | 132596 | 265324 | 198544.246 | 864653.333 | 7.92695e-15 | 9.84557e-14 | 0 |
| ct_1001 PA | taubin_decimate50 | 132596 | 265324 | 185749.313 | 864240.846 | 0.0880578 | 0.803673 | 0 |
| ct_1004 LV | original | 79768 | 159604 | 59141.119 | 464454.792 | 0 | 0 | 0 |
| ct_1004 LV | taubin | 79768 | 159604 | 54118.366 | 464454.063 | 0.081778 | 0.394612 | 0 |
| ct_1004 LV | decimate50 | 39867 | 79802 | 59159.092 | 464459.292 | 0.000793782 | 0.265403 | 0 |
| ct_1004 LV | taubin_decimate50 | 39867 | 79802 | 54870.365 | — | 0.0975418 | 0.646276 | 1 |
| ct_1004 MYO | original | 570074 | 1141564 | 422858.966 | — | 0 | 0 | 3 |
| ct_1004 MYO | taubin | 570074 | 1141564 | 390970.439 | — | 0.0774781 | 0.394526 | 3 |
| ct_1004 MYO | decimate50 | 284683 | 570782 | 422860.374 | — | 3.56192e-07 | 0.00712384 | 3 |
| ct_1004 MYO | taubin_decimate50 | 284683 | 570782 | 394019.211 | — | 0.0881835 | 0.65483 | 5 |
| ct_1004 PA | original | 442913 | 885842 | 327009.478 | 1747699.125 | 0 | 0 | 0 |
| ct_1004 PA | taubin | 442913 | 885842 | 304237.141 | 1747311.687 | 0.0770521 | 0.333875 | 0 |
| ct_1004 PA | decimate50 | 221452 | 442920 | 327009.478 | 1747699.125 | 9.23451e-15 | 1.13687e-13 | 0 |
| ct_1004 PA | taubin_decimate50 | 221452 | 442920 | 306522.460 | 1747515.528 | 0.086174 | 0.774597 | 0 |
