# Следующий CHD68 эксперимент: собственная 3D-модель v3

Дата: 10.10.2026. Статус: **аудит завершён, протокол подготовлен; собственная 3D-модель и новый trainer ещё не реализованы, GPU benchmark и обучение не запускались**.

Три агента независимо разобрали архитектуру и обучение, данные и предобработку, оценку и ошибки. Главный агент проверил результаты и свёл их в этот план. Он описывает улучшение всей системы, а не изолированный эксперимент «только сменить 2D на 3D».

## Основное решение

Следующим стоит обучать собственную residual 3D U-Net с равным весом случаев, выборкой по структурам, контролируемой аугментацией и снижением learning rate. Имеющиеся 48 размеченных объёмов позволяют проверить более сильный pipeline. Большой прирост качества до проверки обещать нельзя.

Методическая опора: [3D U-Net](https://arxiv.org/abs/1606.06650) рассматривает обучение объёмной сегментации с 3D-операциями. [Исследование строгой проверки моделей](https://arxiv.org/abs/2404.09556) показывает важность хорошо настроенных CNN/U-Net baseline и честного сравнения. Точные размеры и гиперпараметры ниже — наши инженерные гипотезы, а не готовая архитектура или доказанный оптимум из статьи.

## Что выяснили по фактическим данным

**Выборка.** 12 381 training slices за эпоху — коррелированные срезы всего 48 release cases. Независимость этих случаев на уровне пациентов отдельно не подтверждена. Текущий sampler даёт объёму с 500 срезами в **3,65 раза** больше примеров, чем объёму с 137 срезами. Его следует заменить равномерной выборкой случаев.

**Текущая модель.** V1 имеет 489 112 параметров, получает пять соседних индексных срезов и использует 2D-свёртки. Adam работает с постоянным lr=0.001; аугментация небольшая, обучение FP32. В обоих запусках loss снижается, но validation после начального роста колеблется. Это мотивирует проверку schedule и обобщения; одних кривых недостаточно, чтобы доказать причину плато или переобучение.

**Качество.** Сопоставимая original-grid validation v1: mean macro Dice=0.797833. PA Dice=0.605492, precision=0.610841, recall=0.651359. Resolution384 дал macro delta −0.002602: четыре случая улучшились, шесть ухудшились. Один seed не доказывает, что 384 всегда хуже 256. Прирост PA Dice у v2 сопровождался снижением recall, поэтому оценивать только Dice недостаточно.

**PA не просто самый редкий класс.** В prepared train PA занимает около 5.29M вокселей, RV — 4.39M, LV — 4.74M. Слепое увеличение веса PA в loss не объясняет проблему. Новый аудит всех десяти validation cases показывает в среднем по случаям:

- 19,81% истинной PA предсказано как background;
- 10,75% истинной PA предсказано как LA;
- среди предсказанной PA 13,99% — GT LA, 8,56% — GT RV;
- в ct_1125 68,23% настоящей PA потеряно в background.

Это крупные пропуски и путаница структур. Удаление мелких островков их не исправляет. Эти наблюдения поддерживают гипотезу об объёмном контексте и более равномерном обучении, но не доказывают заранее успех 3D-модели.

**Дополнительные GT labels.** В девяти train cases обнаружено 313 984 вокселя labels 8/12/13/14/15. [Авторы release](https://raw.githubusercontent.com/XiaoweiXu/Whole-heart-and-great-vessel-segmentation-of-chd_segmentation/master/README.md) задают cardiac labels 1–7 и предписывают игнорировать дополнительные несердечные labels. Политика ignore255 сохраняется во всём новом pipeline.

[Диагностические кривые обучения](E:/3d-heart-data/pediatric_ct_heart/experiments/cardiac_chd68_next_training_design_20261010/training_convergence_diagnostics.png) показывают собственную prepared-grid validation каждого run: 256 и 384 соответственно. Эти кривые не заменяют paired original-grid comparison.

## Геометрия и охват контекста

Аудит всех 48 train cases: медианный bbox семи классов в исходных координатах XYZ — 345.5×268.5×204; p90 — 439.6×369.3×300.2; максимум — 477×391×381. Медианная длина AO по Z — 204, p90 около 292.6. Для PA это 143.5 и 233.

Все числа — **индексные координаты, не миллиметры**. Headers объявляют spacing 1³ и неизвестные пространственные единицы. Это не доказывает физическую изотропность. Большинство объёмов имеют 512×512×Z, но ct_1081 уже имеет иной FOV: 181×170×249.

После существующего XY fit до 256 патч Z128×Y160×X160 приблизительно вмещает весь bbox AO только в одном случае из 48, PA — в 15. Объединённый bbox всех семи классов не помещается ни в одном. Raw bbox включает мелкие GT-компоненты, а пересчёт XY приближённый: это диагностика контекста, не прогноз accuracy.

**128 срезов значительно больше пяти, но это всё ещё локальный контекст.** Даже расширение XY не даст полного хода аорты в большинстве случаев. Поэтому предусмотрен следующий условный этап с глобальным coarse context, если локальная 3D-модель сохранит путаницу сосудов.

Для первого candidate сохраняем full-FOV 256XY/nativeZ, clip [0,2015] и линейную нормализацию v1. Так одновременно не добавляется непроверенная смена нормализации. Не подставляем неподтверждённые ImageCHD PixelSpacing и не применяем HU-window к числам 0–4095.

До дорогого run нужно проверить GT downsample→restore roundtrip по каждому классу и тонким ветвям. Наличие всех классов после downsampling не означает сохранение каждой PA-ветви. Большая потеря разметки в представлении потребует high-resolution refinement.

Центральный crop опасен: LA включает pulmonary veins, RA — venae cavae, а периферические сосуды важны. GT допустима для выбора training patches. Validation и inference должны покрывать весь FOV без GT-guided ROI.

## Предлагаемый первый рецепт

| Компонент | Предложение до реализации и benchmark |
|---|---|
| Модель | Своя residual 3D U-Net с нуля, один CT-канал → восемь logits |
| Каналы | 24/48/96/192/256; две Conv3d 3³, GroupNorm8, LeakyReLU, residual projection |
| Decoder | Trilinear upsample и skip concatenation; примерно 10M параметров, точный count после реализации |
| Начальный patch | Z128×Y192×X192: более широкий XY-контекст после аудита, локальный по Z |
| Batch / precision | 2; BF16 при подтверждённой поддержке, иначе FP16+GradScaler; losses и reductions FP32 |
| Memory fallback | При реальном OOM или peak >20GB: 128×160×160; затем каналы 16/32/64/128/256 |
| Sampler | Равный вес 48 cases; один uniform-FOV patch и один class-balanced foreground patch на batch |
| Loss | Unweighted CE + soft foreground Dice; ignore255 во всех слагаемых |
| Deep supervision | Full head 2/3 + half head 1/3; aux logits восстанавливаются к full target перед loss |
| Optimizer | AdamW, lr=3e−4, weight_decay=1e−4 |
| Schedule | Warmup 500 updates, cosine до 1e−6, максимум 30 000 updates |
| Seeds | Первый 20261010; при успехе дополнительные 20261011 и 20261012 |
| Validation | Updates 1000, 2000, 4000, …, 30000: все десять полных original-grid объёмов |
| Checkpoint | Максимальный raw mean-case macro Dice; при равенстве более ранний step; best и last |

Размер 192XY — выбор главного агента после обсуждения и геометрического аудита. Он ещё не проверен по памяти. Memory planning использует только train technical benchmark; нельзя выбирать patch по лучшему validation Dice. Если все предусмотренные варианты не помещаются, planning останавливается, а не незаметно меняет обучение.

В каждом batch случай выбирается равномерно. Обычный patch обучает отрицательным областям и полному FOV. Для второго patch класс равномерно выбирается среди присутствующих 1–7, затем выбирается его foreground centre с jitter. Нужно сохранять histogram показанных cases, classes и foreground voxels.

Batch Dice считается по reference-present foreground classes. Отсутствующие классы не получают награду empty-empty=1, их false positives штрафует CE. Для aux head logits поднимаются к полной сетке target, чтобы downsampling GT не удалял тонкие структуры. Политика loss фиксируется до обучения и отдельно проверяется тестами.

Padding target=255, все labels>7 игнорируются. Пространственные transforms согласованно меняют CT, target и valid mask; mask interpolation — nearest. После преобразований проверяется целочисленный namespace 0…7/255.

Аугментации: XY rotation ±15°, небольшие index-coordinate tilts ±5°, scale 0.85–1.15, ограниченные contrast/gamma/shift/noise и слабый blur. Точные intensity bounds и вероятности задаются в implementation config и просматриваются на train QA до freeze. Первый run не использует mirrors, крупные elastic-деформации или жёсткую «нормальную» топологию. Физическая семантика этим transforms не приписывается.

[PyTorch AMP](https://docs.pytorch.org/docs/2.8/amp.html) позволяет управлять mixed precision. Здесь это способ экономить память; само по себе AMP не является методом улучшения accuracy.

30 000 updates — предварительный максимальный бюджет, не доказанный оптимум и не эквивалент прежним 30 epochs / 23 220 updates. Теперь sample — объёмный patch. Для первого candidate не предусматривается остановка по validation plateau или продление бюджета по score: max updates и schedule заданы заранее. Технический failure или явная остановка пользователя сохраняют checkpoints.

Это дизайн, **ещё не executable config**. Конкретные параметры, data hashes и код фиксируются коммитом после технических проверок, до quality evaluation.

## Что сделать до полного обучения

1. Реализовать отдельные own3D model/data/loss/trainer/inference modules. Legacy v1/v2 pins, outputs и guards не ослаблять.
2. Проверить synthetic shape/skip alignment, gradients, ignore255 в основном и auxiliary loss, padding, coverage sliding windows, save/load/resume и воспроизводимый sampler. Интерфейс inference не принимает GT.
3. Подготовить portable public train/validation cache с manifest/SHA. Проверить конечность CT, namespace и обратимость grid transform. Heavy files остаются на E с reserve80GB. Перед transfer проверить фактическую quota remote120GB; старые данные сохранить.
4. Сделать fixed-patch learning sanity на двух train cases: finite nonzero gradients, изменённые weights и снижение loss. Sanity weights не использовать для инициализации полного run.
5. На RTX4090 измерить 50–100 updates с загрузкой/аугментацией и actual peak reserved VRAM. Выбрать memory config до просмотра качества, затем создать отдельный execution pin.
6. Проверить полный inference: overlap50%, Gaussian probability blending, положительное покрытие каждого voxel, восстановление original-grid probabilities перед argmax. Измерить full-volume validation time. Patch Dice и pooled Dice не использовать для выбора checkpoint.
7. Сохранять model, optimizer, scheduler, AMP state, Python/NumPy/CPU/CUDA RNG, sampler/step и config/data hashes. Не перезаписывать interrupted run и не запускать второй процесс рядом с живым.
8. Время считать по actual 3D benchmark: `30000 × seconds/update + 16 × measured validation + save/export`. Throughput прежней 2.5D-модели сюда не переносить. Новое сопровождение подключать после конкретного preflight; старая heartbeat остаётся PAUSED.

## Цели качества, заданные до результата

Это инженерные цели, не медицинские нормы и не заявление статистической значимости:

- Raw original-grid mean macro Dice ≥0.807833, то есть baseline+0.01 (один процентный пункт). Это цель, не обещанный прирост.
- Mean PA Dice ≥0.605492; mean PA recall ≥0.631359. Допускаемый trade-off recall до 0.02 задан заранее. V2 с recall около 0.6249 этот guard не проходит.
- Ни один из остальных шести class mean Dice не снижается более чем на 0.01.
- Нижний квартиль case macro Dice не ухудшается; quantile method заранее фиксирован как `np.quantile(method='linear')`.
- Публикуются все десять paired differences, class precision/recall/IoU и confusion. Case macro regression <−0.03 помечается и блокирует замену baseline до объяснения и проверки.
- QA показывает best/median/worst по фиксированной метрике и case с наибольшим регрессом; красивые примеры не заменяют остальные.

Средний score не гарантирует сохранение каждой ветви. Connected components, skeleton continuity и boundary QA нужны как диагностика. Нельзя навязывать одну компоненту или нормальные сосудистые соединения всем CHD. Пока physical scale не подтверждён, HD95/ASSD в mm не отчитывать; index-grid distances не становятся физическими измерениями.

## Ограниченный порядок следующих экспериментов

Если цели достигнуты — два дополнительных фиксированных seeds, всего три runs. Отчёт содержит каждый результат и mean/range, без выбора лучшего seed. Seeds проверяют устойчивость обучения, но не исправляют uncertainty split и не заменяют CV.

При неуспехе выбирается один следующий фактор по диагностике:

- Потеря ветвей при уменьшении изображения → higher-resolution refinement.
- Сохранившаяся путаница LA/RV/AO/PA → собственный global coarse3D + fine3D refiner. Coarse получает весь уменьшенный объём; fine использует CT и coarse probabilities. Для обучения refiner нужны OOF coarse predictions, чтобы избежать слишком хорошего in-sample context. На validation никакого GT crop или GT context.
- Подтверждённый intensity shift/saturation → class-conditioned train intensity audit и одна normalization ablation. Foreground-fitted clip/mean/std — принцип из [официальной документации normalization](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation_normalization.md). Здесь это эксперимент над source intensity, не HU calibration.

Широкий поиск десятков вариантов на тех же десяти validation cases не планируется. Cascade, topology loss, self-supervised pretraining и ensemble не добавляются одновременно в первый run.

Для более сильной методической проверки затем возможен matched 3-fold CV внутри исходных 48 train cases: свежие candidate и baseline на 32 cases каждого fold, preprocessing fit только на этих 32, оценка 16 OOF cases. Это минимум шесть runs. Старый v1 checkpoint, обучавшийся на всех 48, нельзя оценить на fold-val и назвать OOF. Такой CV стоит делать после технически и практически успешного candidate. Принцип fold validation и ensemble описан в [официальном workflow](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md); порядок здесь выбран с учётом бюджета проекта.

## Какие дополнительные данные действительно нужны

Сначала стоит использовать 48 размеченных объёмов лучше. Затем полезны разнообразные экспертно размеченные анатомии, сканеры и возрастные группы. Ещё сотни похожих срезов не заменяют такое разнообразие.

ImageCHD пересекается с CHD68: повторы не являются новыми независимыми данными. Adult/binary heart datasets не становятся педиатрической семиклассовой разметкой без согласования labels и семантики. Неразмеченные CT и pseudo-labels с ошибочной PA могут закрепить ошибку; в первом recipe их нет.

Исторические десять v1 test cases уже просмотрены, их ошибки участвовали в формировании гипотез. Frozen48/10/10 artifact сохраняется, новые test pixels не читаются, но эти десять нельзя объявить новым непредвзятым holdout. Дальнейшие результаты на validation — development / validation-selected. Для независимой проверки нужен новый локальный экспертный reference, замороженный до выбора модели.

Два неразмеченных НИИ CT позволяют проверить engineering и domain assumptions, но не количественную accuracy. В cloud они не передаются. Возраст, независимость пациентов и physical scale CHD68 не подтверждены: сильный публичный pilot не равен neonatal или clinical validation.

## Проверяемые результаты этого разбора

[Machine-readable design](../metadata/pediatric/cardiac_chd68_3d_training_design_v3.json) содержит proposed recipe, assumptions, readiness status и hashes. Полные публичные diagnostics находятся вне Git на E:

- [Train geometry, все 48 cases](E:/3d-heart-data/pediatric_ct_heart/experiments/cardiac_chd68_next_training_design_20261010/train_geometry_audit.json). SHA256: `3289a5ddbb92818a02f3d80d3c582278f79b4779600ed9aa48aec65bec78c9d1`.
- [Validation PA confusion, все 10 cases](E:/3d-heart-data/pediatric_ct_heart/experiments/cardiac_chd68_next_training_design_20261010/validation_PA_confusion_audit.json). SHA256: `1eeeb9acee32f567501f43f8dcbcf78d26913de80856022058f29a035230431a`.
- [Sampler, pipeline и источники кривых](E:/3d-heart-data/pediatric_ct_heart/experiments/cardiac_chd68_next_training_design_20261010/training_pipeline_diagnostics.json).

Train CT/GT SHA и grids перепроверены; train GT pixels читались только для аудита. Validation GT/prediction SHA перепроверены, PA metrics воспроизведены с допуском 1e−12. Source files сохранены. Новых test/private pixels, GPU benchmark или training в этом разборе не было.
