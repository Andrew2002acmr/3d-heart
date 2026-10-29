"""Generate the stage-02 report from measured results, never synthetic stand-ins."""
import csv
import json
from pathlib import Path
import numpy as np
from ct_lab.spatial_run import OUT,REPO,previous_hashes,protected_hashes


def run():
    result=json.loads((OUT/'results.json').read_text());plan=result['protocol']
    main=[s for s in result['scores'] if s['scope']=='main unfiltered'];filters=[s for s in result['scores'] if s['scope']!='main unfiltered']
    errors=json.loads((OUT/'error_diagnostics.json').read_text());correlations=json.loads((OUT/'correlations.json').read_text())
    preserved=json.loads((OUT/'preserved_stage01.json').read_text());protected=json.loads((OUT/'protected_files.json').read_text())
    if previous_hashes()!=preserved:raise ValueError('Previous results changed')
    if protected_hashes()!=protected:raise ValueError('Production preprocessing/split/weights changed')
    with (OUT/'by_case.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['pair','configuration','case','macro_f1','balanced_accuracy','silhouette','fisher_ratio','samples'])
        writer.writeheader();writer.writerows({'pair':s['pair'],'configuration':s['configuration'],'case':case,**{k:m[k] for k in writer.fieldnames[3:]}}
                                             for s in main for case,m in s['per_case'].items())
    lines=['# SPATIAL_FEATURE_EXPERIMENT_02','', 'Дата: 28.10.2026. Реальные ImageCHD68 CT; без обучения нейросетей и без изменения production preprocessing.', '',
           '## Протокол и воспроизводимость','',
           '**Это продолжение разработки на validation, уже использованной на этапе 1; не независимая финальная оценка.** Sealed test CT не открывались.',
           'Сохранены те же 5 train/development groups и 10 validation groups, три Z-позиции 0.35/0.50/0.65 и все sampled IDs/labels/folds из этапа 1. Для каждого release-case feature sets A–H имеют строго одинаковое число наблюдений. Пересечения train/validation release-groups нет.',
           'Train: '+', '.join(plan['train_cases'])+'. Validation: '+', '.join(plan['validation_cases'])+'.',
           'Протокол сохранён до validation extraction/scoring в `outputs/spatial_feature_02/protocol.json`; все 8 основных и 9 фильтрационных конфигураций предзаданы. k=15, distance weighting, exact cKDTree; mean/std fit только train. Никакого global refit CHD68.',
           'NIfTI affine проверен на native axis-aligned RAS; все плоскости получены из одного XYZ volume. Sampled IDs — C-order logical source indices с привязкой к case, не byte offsets. Oblique данные не транспонируются молча: текущий fixed-ID протокол их отклоняет. Spacing не подтверждён как мм, интенсивности не считаются HU.', '',
           '## Матрица признаков','', '| Set | Columns | Input features |','|---|---:|---|']
    for s in [s for s in main if s['pair']=='LV/MYO']:lines.append(f"| {s['configuration']} | {s['feature_count']} | {', '.join(s['features'])} |")
    lines.extend(['',
                  '2D texture: local mean/std 5×5, central gradient, LoG σ=1 grid unit. Tri-planar: те же четыре величины в XY/XZ/YZ. 3D texture: mean/std в 5³, 3D gradient/LoG; G дополнительно включает Gaussian 3D σ=0.5/1/2. Поэтому выигрыш G нельзя приписать только замене плоскости на 3D: нужны отдельные заранее запланированные ablations Gaussian channels.',
                  'Z context каждого радиуса: соседние intensities и differences (neighbor−center), mean/std/range по окну, central Z gradient. Центральная intensity не дублируется в classifier. Raw XYZ profiles экспортируются/показываются отдельно; F не получает скрытые дополнительные Z-profile channels. H явно добавляет Z±3.',
                  'На границах profiles/context используют deterministic edge replication, texture — half-sample reflect. Градиенты texture соответствуют np.gradient (one-sided на границе), central context gradient использует replicated ±1. Флаги выхода за Z сохранены отдельно в context_flags_*.npz.', '',
                  '## Основные результаты на frozen validation','',
                  'Главная метрика — mean case Macro F1; ± — SD по 10 случаям, не confidence interval. F1 сохраняет правило этапа 1: два заранее заданных класса, zero_division=0, включая empty-empty=0. В выбранных трёх срезах некоторых случаев один класс отсутствует; это не означает его отсутствия во всём сердце. Balanced Accuracy усредняет reference-present recalls. Fisher и silhouette усредняются только по случаям, где оба класса доступны; silhouette — до 128 samples/class/case, Euclidean train-standardized space. Fisher — squared standardized mean distance / sum within-class variance traces. Они диагностические, не Dice.', '',
                  '| Pair | Features | Case Macro F1 ± SD | Balanced accuracy | Silhouette | Fisher |','|---|---|---:|---:|---:|---:|'])
    for s in main:
        m=s['case_macro'];lines.append(f"| {s['pair']} | {s['configuration']} | {m['macro_f1_mean']:.4f} ± {m['macro_f1_std']:.4f} | {m['balanced_accuracy_mean']:.4f} | {m['silhouette_mean']:.4f} | {m['fisher_ratio_mean']:.4f} |")
    class_rows=[]
    for s in main:
        for cls,name in enumerate(s['pair'].split('/')):
            recalls=[];f1s=[]
            for metric in s['per_case'].values():
                cm=np.array(metric['confusion_matrix']);truth=cm[cls].sum();pred=cm[:,cls].sum();tp=cm[cls,cls]
                if truth:
                    recalls.append(tp/truth);f1s.append(2*tp/(truth+pred))
            class_rows.append({'pair':s['pair'],'configuration':s['configuration'],'class':name,
                               'mean_case_f1':float(np.mean(f1s)),'mean_case_recall':float(np.mean(recalls)),'reference_present_cases':len(recalls)})
    with (OUT/'per_class.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(class_rows[0]));writer.writeheader();writer.writerows(class_rows)
    lines.extend(['','Per-class таблица ниже использует только reference-present случаи, поэтому её столбцы не обязаны усредняться в основной fixed-label Macro F1.',
                  '', '| Pair | Class | GT-present cases | B case F1 | G case F1 | B recall | G recall |','|---|---|---:|---:|---:|---:|---:|'])
    for pair in ('LV/MYO','RV/PA'):
        for cls in pair.split('/'):
            a=next(r for r in class_rows if r['pair']==pair and r['class']==cls and r['configuration'].startswith('B '))
            b=next(r for r in class_rows if r['pair']==pair and r['class']==cls and r['configuration'].startswith('G '))
            lines.append(f"| {pair} | {cls} | {a['reference_present_cases']} | {a['mean_case_f1']:.4f} | {b['mean_case_f1']:.4f} | {a['mean_case_recall']:.4f} | {b['mean_case_recall']:.4f} |")
    lines.extend(['','## 1–4. Соседние срезы, три плоскости и 3D','',
                  'LV/MYO: исходная intensity A остаётся сильнее всех испытанных пространственных k-NN вариантов по среднему F1. Относительно B добавление Z/tri-planar почти не меняет результат и в среднем слегка снижает F1; G немного выше B, но ниже A. Это не основание добавлять все пространственные каналы для LV/MYO.',
                  'RV/PA: B с текстурой заметно сильнее A. Z±1/±2 не улучшили B, Z±3 дал небольшой прирост среднего, но меньше межслучайного разброса. Нельзя назвать ±3 универсально оптимальным. Tri-planar F и H дают небольшой средний эффект; полноценный G — самый высокий средний F1 среди основных A–H в данном протоколе.',
                  'Эффекты не монотонны по радиусу и не одинаковы у пациентов. Простой список соседних intensities/differences не заменяет 3D representation, а высокий размер feature space может ухудшать k-NN. Все варианты оцениваются на одних и тех же voxels, так что отличие не связано с изменением sample size.', '',
                  '| Pair | Variant vs B | Mean paired case ΔF1 | Cases improved / worsened / equal |','|---|---|---:|---|'])
    for pair in ('LV/MYO','RV/PA'):
        baseline=next(s for s in main if s['pair']==pair and s['configuration'].startswith('B '))
        for s in main:
            if s['pair']!=pair or s['configuration'][0] not in 'CDEFGH':continue
            delta=np.array([s['per_case'][c]['macro_f1']-baseline['per_case'][c]['macro_f1'] for c in baseline['per_case']])
            lines.append(f"| {pair} | {s['configuration']} | {delta.mean():+.5f} | {(delta>1e-12).sum()} / {(delta< -1e-12).sum()} / {(np.abs(delta)<=1e-12).sum()} |")
    lines.extend(['','## 5. Фильтрация: отдельный контроль','',
                  'Диагностическая подвыборка: фиксированные первые до 32 исходных sampled ID каждого класса на каждом выбранном Z; одинакова у всех девяти фильтрационных вариантов и всех diagnostic feature sets. Она меньше основной выборки: её F1 сравнивается только с её собственным Baseline, а не с основной таблицей A–H.',
                  'Gaussian σ=0.5/1/2 в XY посрезово и согласованно в XYZ; truncate=4. Bilateral spatial σ=1, intensity σ=100 source units, truncate=3 в обеих размерностях. Здесь bilateral — exact finite kernel; этап 1 использовал skimage bins=10000, поэтому его F1 не считается побайтово идентичным этой реализации. CLAHE не включён.',
                  'В diagnostic 3D-core — пять каналов (Intensity + 3D gradient/mean/std/LoG), без Gaussian auxiliary channels G основной матрицы.', '',
                  '| Pair | Filter | 2D texture F1 | 3D-core F1 |','|---|---|---:|---:|'])
    for pair in ('LV/MYO','RV/PA'):
        for f in plan['filters']:
            a=next(s for s in filters if s['pair']==pair and s['filter']==f['name'] and s['configuration']=='B 2D')
            b=next(s for s in filters if s['pair']==pair and s['filter']==f['name'] and s['configuration']=='G 3D-core')
            lines.append(f"| {pair} | {f['name']} | {a['case_macro']['macro_f1_mean']:.4f} | {b['case_macro']['macro_f1_mean']:.4f} |")
    lines.extend(['',
                  'Для RV/PA Gaussian σ=2 3D + 3D-core показывает более высокий описательный F1 на этой reduced-проверке, чем её baseline. Для LV/MYO согласованная 3D-фильтрация не даёт универсального выигрыша, а сильное сглаживание может терять детали. Это secondary development finding; параметры по результатам не перестраивались и вывод требует отдельного подтверждения.', '',
                  '## 6. Где исправляются ошибки','',
                  'Distance-to-GT вычислена после extraction, отдельно от feature bank: расстояние до центра ближайшего 6-connected boundary voxel своего класса, в header grid units. Это не точное расстояние до непрерывной анатомической поверхности и не мм. В k-NN/PCA/UMAP это поле не передаётся.',
                  'Ниже validation counts: corrected = B wrong → variant correct; new wrong = B correct → variant wrong. Counts суммируются по sampled voxels, это не case-macro F1. У границ и в центре имеются разные baseline error counts; сырые числа без знаменателя нельзя трактовать как предпочтительную область улучшения.', '',
                  '| Pair | Variant | Region | Baseline wrong | Corrected | New wrong | Mean case correction fraction |','|---|---|---|---:|---:|---:|---:|'])
    for pair in ('LV/MYO','RV/PA'):
        for letter in ('E','F','G'):
            for region in ('boundary ≤1','transition (1,3]','interior >3'):
                rows=[r for r in errors if r['role']=='validation' and r['pair']==pair and r['configuration']==letter and r['region']==region]
                fractions=[r['correction_fraction'] for r in rows if r['correction_fraction'] is not None]
                fraction=f'{np.mean(fractions):.4f}' if fractions else 'n/a'
                lines.append(f"| {pair} | {letter} | {region} | {sum(r['baseline_wrong'] for r in rows)} | {sum(r['corrected'] for r in rows)} | {sum(r['new_wrong'] for r in rows)} | {fraction} |")
    lines.extend(['',
                  'Проверка показывает и исправления, и новые ошибки во внутренних областях, а не только на границе. Нет оснований объявлять эффект исключительно boundary denoising. Spatial Error Maps доступны для заранее выбранного ct_1001 (optimistic train voxel holdout) и первого validation case ct_1027; полные sampled predictions/errors сохранены по всем 15 случаям.', '',
                  '## 7–8. Полезность по классам и избыточность','',
                  'Наибольшая полезность в проверенных парах — у сложной RV/PA; для LV/MYO более высокое исходное intensity separation оставляет меньше возможностей для выбранного k-NN. Это вывод о бинарных GT-selected samples, не о качестве multiclass сегментации.',
                  'В reference-present class-анализе основная прибавка G относительно B у RV: F1 и recall растут заметнее, чем у PA. У PA F1 меняется слабо, recall может слегка снизиться; нельзя приписать одинаковую пользу обоим классам.',
                  'Z differences линейно выражаются через соседнюю и центральную intensity; context gradient повторяется у разных радиусов, а mean/std/range могут быть сильно коррелированы. Добавление всех таких столбцов не равно добавлению независимой информации.', '',
                  '| Pair | Train feature 1 | Train feature 2 | Pearson r |','|---|---|---|---:|'])
    used=set(n for names in plan['feature_sets'].values() for n in names)
    for pair,corr in correlations.items():
        names=corr['names'];a=np.array(corr['correlation']);pairs=[]
        for i,n in enumerate(names):
            for j in range(i+1,len(names)):
                if n in used and names[j] in used and abs(a[i,j])>=.98:pairs.append((abs(a[i,j]),n,names[j],a[i,j]))
        for _,n,m,r in sorted(pairs,reverse=True)[:12]:lines.append(f'| {pair} | {n} | {m} | {r:.5f} |')
    lines.extend(['','## Проекции и интерфейс','',
                  'Spatial Feature Explorer связывает Canvas viewer трёх плоскостей с одним source XYZ и affine; щелчок по любой плоскости обновляет crosshair, scatter selection подсвечивает source voxels. Сплошные маркеры — в текущем срезе, кольца — проекция выбранных voxels других срезов, а не утверждение, что они лежат в этой плоскости. Error overlay рисует только in-plane evaluated voxels.',
                  'Independent PCA/UMAP до/после fit только на train voxel fold ct_1001; IDs одинаковы, seed=20261028, UMAP neighbors=15/min_dist=0.1/epochs=200, n_jobs=1. Независимые basis нельзя сравнивать по абсолютным расстояниям.',
                  'Shared projection реализована на пересечении реально измеренных столбцов B∩context. В неизменённых общих столбцах before/after совпадают: это математически корректный negative control. Он не отображает пользу исключённых дополнительных осей; применять одну PCA к матрицам разной размерности без явного отображения нельзя. Нулевые/выдуманные context channels не подставляются.',
                  'PCA при одном столбце имеет нулевую вторую координату. Проекции — визуальная диагностика, не доказательство роста качества.', '',
                  '## Производительность и корректность','',
                  'CPU-only, float32 retained features, временные float64 moments; координатные блоки по 64 points, filtered blocks по 16. Halo = feature kernel radius + filter radius. Вычисляются только локальные source patches, а не dense feature tensor на всём наборе. Память ограничена размером patch/chunk, результаты case-cached.',
                  'Тесты сравнивают sampled и плотные 2D/3D расчёты, разные chunk sizes/halos, global boundaries, Gaussian и exact bilateral, voxel IDs и world-coordinate соответствие, train-only fit, отсутствие GT features и неизменность файлов. Training-тесты нейросетей не включаются.',
                  f'Предыдущие результаты: {len(preserved)} файлов этапа 1 и аудита совпали по SHA256. Existing CHD68 weights, frozen split и preprocessing совпали по SHA256.', '',
                  '## 9. Рекомендации по следующему этапу магистерской','',
                  'Сначала зафиксировать отдельный ablation-протокол: intensity baseline, 2D texture, 3D-core без auxiliary Gaussian и 3D-core с ними. Для RV/PA проверить согласованное 3D сглаживание и learned spatial representation; сравнить boundary/interior ошибки и тонкие структуры. Не выбирать радиус только по среднему этой reused validation.',
                  'Перед физически заданным контекстом подтвердить исходную геометрию/spacing. Отдельно исследовать уменьшение избыточности context channels и объём k-NN training samples. Обучение настоящего сегментатора — только отдельным следующим этапом после фиксации гипотез, без GT-distance/масочных признаков и без изменений sealed test для подбора.', '',
                  '## 10. Что не подтверждено','',
                  'Не подтверждены рост Dice, польза 3D для всех классов/случаев, оптимальность конкретного sigma/radius, независимая клиническая обобщаемость и результаты whole-volume segmentation. Три Z-позиции и ограниченные samples не покрывают всё сердце. Validation уже участвовала в разработке, поэтому следующий подтверждающий эксперимент должен учитывать это ограничение.', '',
                  '## Артефакты и команды','',
                  '`outputs/spatial_feature_02/`: frozen protocol, main/filter scores JSON/CSV, by_case.csv, separate train_development.json, sampled features/IDs/context flags, validation_predictions.npz, errors_*.npz, error_diagnostics.json, projection arrays/manifests, charts.html и интерактивные графики всех 10 запрошенных типов, PNG/SVG, runtime/provenance.',
                  'Команды из repository: `python -m ct_lab.spatial_run`; `python -m ct_lab.spatial_artifacts`; `python -m ct_lab.spatial_report`; `python -m ct_lab.spatial_launch` (local UI 8510). Повторные запуски используют frozen case caches; изменение протокола поверх завершённого эксперимента отклоняется.', ''])
    (REPO/'SPATIAL_FEATURE_EXPERIMENT_02.md').write_text('\n'.join(lines),encoding='utf-8')
    print('REPORT',len(main),'main comparisons;',len(filters),'filter diagnostics;',len(preserved),'prior files unchanged',flush=True)


if __name__=='__main__':run()
