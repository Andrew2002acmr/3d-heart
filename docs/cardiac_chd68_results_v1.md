# Результаты первого multiclass CHD68 baseline v1

09.10.2026. Ветка feature/pediatric-heart-segmentation.
Own 2.5D U-Net действительно обучена с нуля на RTX4090 и проверена на 10 test cases.
Mean case macro foreground Dice **0.7897**; median **0.8205**,
range **0.5794–0.8783**.
Это рабочий исследовательский baseline с выраженными ошибками сосудов и смешением
структур в некоторых случаях. Результат не подтверждает клиническую готовность
или качество на новорождённых; НИИ CT в этом эксперименте не использовались.

## Протокол и provenance

68 публичных CHD68 CT/label pairs; frozen released-case/group split **48/10/10**,
seed20261009. Patient independence, individual age и physical scale не подтверждены.
ImageCHD duplicates не добавлялись, private NII CT не передавались на Pod.
Execution commit **d89ea289124501654232b8ed6333d671410244d4**, чистый remote checkout
/workspace/3d-heart-cardiac-v1 на feature/pediatric-heart-segmentation.
Документационные commits после execution commit не меняли training code/configs.

Модель: 5 channels, 16/32/64/128, Conv/GN/ReLU, bilinear decoder, 8 logits,
489112 parameters. FP32/batch16, Adam0.001, CE + собственный foreground Soft Dice,
30 epochs. Preprocessing fit только на train: source clip[0,2015], index contexts
[-2,-1,0,1,2], full-FOV XY fit/pad256²; без HU/physical-mm утверждений или GT crop.
Все labels>7 исключены из loss/metrics через ignore255.

Checkpoint выбран только по validation mean-case macro Dice на prepared grid:
best **epoch23**, validation **0.8021**.
После обучения выполнена **одна** test evaluation. Probabilities восстановлены на
original XY grid до argmax; исходные Z/affine/units сохранены, GT не перезаписана.
Test не использовался для tuning/threshold/crop/postprocessing/повторного обучения.

## Фактические ресурсы

RTX4090 24564MiB, driver580.95.05, Python3.12.3, torch2.8.0+cu128/CUDA12.8;
32logical/16physical CPU, RAM134.12GB. Всё remote под persistent /workspace.
Full training+validation/checkpoint I/O: **7064.18s / 117.74min**,
около 1h57m44s. Train batches суммарно 6861.98s,
validation 197.07s.
Train epoch varied 153.7–300.2s:
network filesystem/cache/load влияют на throughput.
Peak RSS 2.164GB;
CUDA allocated 1.919GB,
reserved 2.852GB.
Mean process CPU 217.99% (100%=1logical CPU).

Best checkpoint 6094048bytes; last
6141664bytes. Checkpoints включают optimizer/RNG/history/provenance.
Fixed seeds/environment recorded; CUDA nll_loss2d warned about nondeterministic
implementation under warn_only. Побитовую идентичность повторных runs не заявлять.

## Test metrics

Числа — равновесные средние по случаям, затем foreground structures; background
не входит в macro Dice. Empty-reference classes не награждаются perfect-empty score.
Для этих 10 случаев все семь foreground classes представлены.
Population std case-macro Dice **0.0877**; small n=10
не позволяет надёжно оценить обобщение на все возрастные/протокольные группы.

| Структура | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| LV | 0.8298 | 0.7190 | 0.8317 | 0.8652 |
| RV | 0.7881 | 0.6573 | 0.7615 | 0.8515 |
| LA | 0.8384 | 0.7293 | 0.8755 | 0.8120 |
| RA | 0.7975 | 0.6795 | 0.8578 | 0.7836 |
| MYO | 0.8419 | 0.7323 | 0.8490 | 0.8410 |
| AO | 0.7474 | 0.6146 | 0.7724 | 0.7519 |
| PA | 0.6847 | 0.5470 | 0.7173 | 0.6807 |

| Test case | Macro foreground Dice | Опубликованные diagnosis codes |
|---|---:|---|
| ct_1032 | 0.8783 | PS |
| ct_1053 | 0.8570 | ASD, AAA |
| ct_1128 | 0.8415 | VSD |
| ct_1041 | 0.8396 | VSD |
| ct_1026 | 0.8234 | AD |
| ct_1048 | 0.8176 | ASD, AAA |
| ct_1044 | 0.8111 | ASD, VSD, AD |
| ct_1091 | 0.7707 | VSD, PDA, CAT, AAA |
| ct_1022 | 0.6784 | ASD, PAS |
| ct_1083 | 0.5794 | TGA |

Codes перенесены из авторского описания release; их наличие не подтверждает
возраст или clinical metadata linkage. Ни одной модели healthy/pathological/diagnosis
classification здесь нет. По одному case не делать вывод о зависимости quality от диагноза.
HD95/ASSD в mm, мл и физические площади не вычислялись: scale release unknown.

## Ошибки и визуальное сравнение

Камеры и миокард в среднем лучше сосудов, PA — наиболее слабая структура.
Самый слабый ct_1083 (author code TGA): macro Dice0.5794, PA0.3375,
PA recall0.2313; большая часть PA размечена моделью как другие структуры.
AO predicted voxel count1,560,568 против GT711,143; precision0.3188.
LV recall0.4681. Это существенные ошибки анатомической классификации и границ,
а не только недостаток сглаживания mesh. ct_1022: PA Dice0.4531,
RA recall0.4172; частичный пропуск структуры.
В среднем по рангу ct_1026 RA precision0.5295 при recall0.9381 — избыточная область.

Три явно post-hoc selected иллюстрации: лучший ct_1032 (0.8783), upper-median
ct_1026 (0.8234), худший ct_1083 (0.5794). Они объясняют диапазон ошибок,
не являются случайной representative sample. Sampling внутри case воспроизводим:
axial20/50/80% GT-positive slices, coronal/sagittal GT bbox centers; отдельно axial
максимального числа неверно классифицированных PA voxels. Это QA, не GT crop модели.
КТ window [0,2015] взят без изменений из train-only preprocessing, не HU.
Одинаковые цвета label1–7 для GT и Prediction; errors red=missed foreground,
blue=extra foreground, yellow=wrong structure. Labels>7 gray/excluded.
Figures показывают index aspect, не physical aspect; mm scale bar отсутствует.

`visual_QA_v2` во внешнем backup содержит full orthogonal comparisons,
компактные JPG previews, PA failure panels, learning_curves.png,
test_metrics.png, SHA/source sampling JSON. PNG/JPG проверены визуально;
источники CT/GT/prediction перед QA проверены SHA-256. Никакой коррекции масок
или component removal/smoothing не применялось.
Первый QA запуск остановился на неверном имени clip field до создания images;
reader исправлен, пустой visual_QA_v1 сохранён. Model/config/predictions не менялись.

## Сохранность и воспроизведение

39 critical payload files сохранены на E: и проверены SHA-256: best/last,
10 original-grid predictions/provenance, metrics, history, config, environment,
pip freeze, resource samples, frozen repository metadata и compute log.
Archive 18315524bytes, SHA-256
`5a34a297068dd0e97affc1cbf2864a2f6b2b07a39928a94b4eb3693d2731580e`; gzip CRC и каждый payload проверены.
Checkpoint SHA separately matched training_summary.
Локально: `$dataRoot/remote_runs/cardiac_chd68_v1_full_20261009/artifacts/`;
$dataRoot = E:/3d-heart-data/pediatric_ct_heart для текущей машины.
Generated outputs/images/weights в Git не включены.

```powershell
$dataRoot = 'E:/3d-heart-data/pediatric_ct_heart'
$backup = "$dataRoot/remote_runs/cardiac_chd68_v1_full_20261009"
$run = "$backup/artifacts/cardiac_chd68_v1"
python -m scripts.qa_cardiac_predictions --data $dataRoot `
  --evaluation "$run/evaluation/test_v1_20261009" `
  --training "$run/experiments/cardiac_chd68_v1/gpu_full_v1_20261009" `
  --output "$backup/visual_QA_repeat"
```

Output должен быть новым, существующие не перезаписываются.
[Machine-readable result](../metadata/pediatric/cardiac_chd68_results_v1.json),
[launch/completion](../metadata/pediatric/cardiac_chd68_full_run_v1.json),
[training readiness/runbook](cardiac_chd68_runpod_readiness.md).
Heartbeat cardiac-chd68-v1 **PAUSED** после завершения и verified backup.
GPU process завершён; Pod может быть остановлен пользователем. Автоматическая
остановка Pod не выполнялась.

## Следующий этап

Сохранить этот результат как v1, не менять split и не оптимизировать по просмотренным
test cases. Приоритет — ошибки PA/AO, class mixing и непрерывность масок в 3D.
На train/validation исследовать влияние context/resolution и сравнить следующий
заранее заданный baseline; это новый эксперимент, а не исправление v1 задним числом.
Повторные сравнения на уже просмотренном test трактовать как exploratory.

Для НИИ CT сначала подтвердить intensity/contrast/domain adapter и anatomical label
semantics. Затем локально сравнить draft prediction с собственным проверяемым reference
или ограниченным экспертным review; два неразмеченных development случая не дают
количественной внешней accuracy. Новые клинические volumes собирать отдельно для
новорождённых/младенцев и других возрастов. Полуручной редактор масок остаётся
нужным: текущая модель создаёт стартовую разметку, требующую проверки и исправления.

Подготовлен отдельный [resolution384 v2](cardiac_chd68_resolution384_v2.md):
вход384 вместо256, прежний frozen split и остальные hyperparameters;
пока только локальная подготовка/QA/CPU smoke, без нового GPU training/test.
