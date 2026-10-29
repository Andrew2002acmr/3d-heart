"""Generate the requested evidence-backed Markdown report from completed real runs."""
import collections
import json
from pathlib import Path
import numpy as np
import ct_lab.launch  # optional workspace-local plotting packages
from ct_lab.core import LABELS


def main():
    repo=Path(__file__).resolve().parents[1]
    root=repo/'outputs/ct_preprocessing_lab'
    dataset=json.loads((root/'dataset.json').read_text(encoding='utf-8'))
    rows=[r for r in dataset['records'] if 'error' not in r]
    research=json.loads((root/'research/measurements.json').read_text(encoding='utf-8'))
    cohort={r['case_id']:r for r in json.loads((repo/'metadata/pediatric/cardiac_chd68_cohort_v1.json').read_text())['records']}
    mismatches=[r['id'] for r in rows if r['id'] not in cohort or r['image_sha256']!=cohort[r['id']]['image_SHA256'] or r['mask_sha256']!=cohort[r['id']]['mask_SHA256']]
    provenance={'matched_pairs':len(rows)-len(mismatches),'mismatched_pairs':mismatches,
                'missing_cohort_cases':sorted(set(cohort)-{r['id'] for r in rows})}
    (root/'provenance_check.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    import plotly.graph_objects as go
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    figures=[]
    fig=go.Figure()
    for c,name in LABELS.items():
        fig.add_trace(go.Box(y=[r['class_statistics'].get(str(c),{}).get('fraction',0) for r in rows],name=name,boxpoints='all'))
    fig.update_layout(title='Class fractions: one observation per released case',yaxis_title='Fraction of case voxels')
    figures.append(fig)
    fig=go.Figure()
    for axis,name in enumerate('XYZ'):
        fig.add_trace(go.Scatter(x=[r['id'] for r in rows],y=[r['geometry']['shape_xyz'][axis] for r in rows],name=name,mode='markers'))
    fig.update_layout(title='Source grid dimensions by case',yaxis_title='Voxels (not mm)')
    figures.append(fig)
    (root/'summary.html').write_text('<meta charset="utf-8">'+''.join(fig.to_html(full_html=False,include_plotlyjs=(True if i==0 else False)) for i,fig in enumerate(figures)),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(10,4))
    ax.boxplot([[r['class_statistics'].get(str(c),{}).get('fraction',0) for r in rows] for c in LABELS],tick_labels=list(LABELS.values()))
    ax.set_ylabel('Voxel fraction per case');fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(root/f'class_fractions.{ext}',bbox_inches='tight')
    plt.close(fig)
    import sys,importlib
    environment={'python':sys.version,'packages':{name:getattr(importlib.import_module(name),'__version__','unknown') for name in
                                                ['numpy','scipy','nibabel','skimage','torch','streamlit','plotly','matplotlib']}}
    (root/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    extra=[r for r in rows if set(r.get('unique_mask_values',[]))-set(LABELS)]
    shapes=np.array([r['geometry']['shape_xyz'] for r in rows])
    means=np.array([r['image_statistics']['mean'] for r in rows])
    anomalies=collections.Counter(a for r in rows for a in r['anomalies'])
    lines=['# IMAGECHD68: анализ реальных данных','', 'Дата: 28.10.2026. Обучение и refit не выполнялись.', '',
           '## 1. Набор данных', '',
           f"Прочитано {len(rows)}/{dataset['discovered_pairs']} пар NIfTI из `../laboratories/data/chd68/pairs`. Ошибок: {dataset['failed']}.",
           f"SHA256 image+mask совпали с frozen cohort для {provenance['matched_pairs']} пар. Несовпадения: {mismatches}. Отсутствующие случаи: {provenance['missing_cohort_cases']}.",
           'Релиз не является независимо подтверждённой педиатрической когортой; уникальность пациентов не подтверждена.', '',
           '## 2. Интенсивности', '',
           f"Минимум по всем случаям: {min(r['image_statistics']['min'] for r in rows):g}; максимум: {max(r['image_statistics']['max'] for r in rows):g} (исходная шкала, не HU).",
           f"Среднее равновесных case means: {means.mean():.6f}; SD между случаями: {means.std():.6f}; диапазон case means [{means.min():.6f}, {means.max():.6f}].",
           f"NaN: {sum(r['image_statistics']['nan'] for r in rows)}; Inf: {sum(r['image_statistics']['inf'] for r in rows)}.",
           'Это распределение по случаям; исследование с большим числом вокселей не получает больший вес. Детальные percentiles — в patients/classes JSON.', '',
           '## 3. Геометрия', '',
           f"Диапазон XYZ размеров: {shapes.min(0).tolist()} — {shapes.max(0).tolist()} (покомпонентные пределы, не форма одного пациента).",
           f"Единицы: {dict(collections.Counter(r['geometry']['spatial_unit'] for r in rows))}.",
           f"Ориентации: {dict(collections.Counter(r['geometry']['orientation'] for r in rows))}.",
           f"Несовпадений активных qform/sform: {sum(r['geometry']['qform_sform_disagree'] for r in rows)}.",
           'Совпадение геометрии CT/mask проверялось по размерам, affine и единицам. Нельзя переводить эти сетки в мм без подтверждения источника.', '',
           '## 4. Экспертные маски', '',
           f"Дополнительные source labels вне 0..7 встречаются в {len(extra)} случаях: "+', '.join(r['id'] for r in extra)+'.',
           'Они сохранены в анализе. Cardiac training/evaluation переводит их в ignore=255; это не background и не дополнительные logits.', '',
           '| Label | Name | Медиана доли по случаям | Min / max доли |','|---|---|---:|---:|']
    for c,name in LABELS.items():
        fractions=np.array([r['class_statistics'].get(str(c),{}).get('fraction',0) for r in rows])
        lines.append(f'| {c} | {name} | {np.median(fractions):.7f} | {fractions.min():.7f} / {fractions.max():.7f} |')
    lines.extend(['','## 5. Перекрытие классов','',
                  'Измерено на полном исходном объёме ct_1001, foreground only; overlap = сумма minimum двух нормированных гистограмм, 128 общих bins. Значение зависит от bins и не является точностью классификатора.', '',
                  '| Пара | Overlap |','|---|---:|'])
    for pair,value in sorted(research['overlap_coefficient_128bins_fullvolume'].items(),key=lambda t:t[1],reverse=True):
        lines.append(f'| {pair} | {value:.6f} |')
    lines.extend(['','## 6. Эксперименты с предобработкой','',
                  'Реальный ct_1001, source axial Z=110; все эксперименты начинаются независимо от одного исходного среза. Глобальные параметры не подгонялись.',
                  f"Clipping [0,2015]: выше границы {research['clipped_above_fraction']:.6%}, ниже {research['clipped_below_fraction']:.6%}.",
                  'Рассчитаны fixed train window, slice Z-score, slice percentile, Gaussian sigma=1 + fixed window, CLAHE + Min-Max, а также три source-scale окна [0,1000], [500,1500], [1000,2015].',
                  'Изображения отображаются с индивидуальным display range; визуальное сходство affine-normalized изображений не означает совпадения нейросетевых тензоров.',
                  'PNG/SVG и интерактивный HTML: `outputs/ct_preprocessing_lab/research/`. Статистики и профиль — `measurements.json`. Изменения распределений не доказывают улучшение сегментации.', '',
                  '## 7. Соседние срезы','',
                  'ct_1001, исходный Z=110, весь FOV. Средняя абсолютная разность с центральным срезом:', '',
                  '| Offset | Distance grid units | Mean abs difference |','|---:|---:|---:|'])
    for item in research['neighbor_context']:
        lines.append(f"| {item['offset']} | {item['distance_grid_units']} | {item['mean_abs_difference_center']:.6f} |")
    lines.extend(['','Это индексный пространственный контекст: расстояния в мм не подтверждены. Ненулевая разность поддерживает исследование 2.5D-контекста, но не доказывает выигрыш конкретной модели.', '',
                  '## 8. Настоящий preprocessing и диагностика моделей','',
                  'Подробный аудит: `DATA_PIPELINE_AUDIT.md`. Настоящий порядок: XY fit/pad → clipping [0,2015] → [-1,1] → offsets [-2,-1,0,1,2]. v1: 256², v2: 384². Нет mm resampling или новой ориентации; source gate RAS.',
                  'Inspector вызывает production prepare_volume; forward_pre_hook подтвердил побитовое совпадение входов обеих исторических моделей на ct_1001. Training/inference preparation сравнивается отдельным тестом.',
                  'Frozen split 48/10/10; v2 test sealed. Ниже — exploratory train/development case ct_1001, не независимая оценка качества.', ''])
    for variant in ('v1','v2'):
        path=root/f'real_volume_{variant}.json'
        if path.exists():
            report=json.loads(path.read_text())
            lines.extend([f'### {variant}: полный объём', '',
                          f"Macro foreground Dice (reference-present classes): {report['metrics']['macro_foreground_Dice']:.6f}.", '',
                          '| Class | Dice | Prediction 3D components | Expert 3D components |','|---|---:|---:|---:|'])
            pred={r['name']:r for r in report['prediction_anatomy']['classes']};gt={r['name']:r for r in report['expert_anatomy']['classes']}
            for name,metric in report['metrics']['classes'].items():
                lines.append(f"| {name} | {metric['Dice']:.6f} | {pred.get(name,{}).get('components_3d_26',0)} | {gt.get(name,{}).get('components_3d_26',0)} |")
            lines.extend(['','3D: 26-connectivity. Отдельные 2D 8-connected counts по каждой оси, размеры компонентов, площади, центры и протяжённость сохранены в JSON. Компоненты разметки сами могут быть множественными; count не равен числу ошибок.', ''])
            if 'fragment_gt_comparison' in report:
                satellites=[r for r in report['fragment_gt_comparison'] if not r['largest']]
                wrong=[r for r in satellites if r['gt_same_class_fraction']==0]
                lines.extend([f"Satellite компонентов (все, кроме крупнейшего компонента каждого класса): {len(satellites)}; из них без единого совпадения с GT своего класса: {len(wrong)}.",
                              'Их размеры, bbox, центры RAS и пересечения с каждой source GT меткой сохранены в fragment_gt_comparison; это локализованная проверка фрагментов, а не только подсчёт компонентов.', ''])
        else:
            lines.append(f'Полный объём {variant}: отчёт ещё не сформирован; результаты не подставляются.\n')
    lines.extend(['## 9. Аномалии','',str(dict(anomalies)), '',
                  'Флаг extra source labels — ограничение cardiac taxonomy, а не автоматически повреждённая разметка. Проверка outlier использует 1.5 IQR case means; это статистический сигнал для ручного изучения. Выбросы: '+', '.join(r['id'] for r in rows if any('Outlier' in a for a in r['anomalies']))+'.', '',
                  '## 10. Подтверждённые выводы и гипотезы','',
                  '**Подтверждено:** неподтверждённые HU/physical units; дополнительные source labels; отсутствие физического контекста в модели; интенсивностное перекрытие ряда foreground классов; различный размер входа v1/v2.', '',
                  '**Гипотезы, требующие проверки:** индексные ±2 planes могут представлять разные физические расстояния в разных исходных сканах; XY downsampling может терять тонкие детали; перекрытие интенсивностей требует анатомического контекста; отсутствие topology constraint может допускать островки. Это возможные механизмы, не установленная причина каждого фрагмента.', '',
                  '## 11. Следующие исследовательские проверки','',
                  'Подтвердить linkage к исходной DICOM геометрии и шкале; сравнить ошибки по размерам структур/компонентов; отдельно исследовать границы и восстановление probabilities; зафиксировать prospective протокол до подбора параметров. Не менять preprocessing обученных весов без отдельной оценки distribution shift.', '',
                  '## Проверки и воспроизводимость','',
                  'Автоматические тесты: `tests/test_ct_lab.py` + существующие cardiac inference/training tests. UI smoke всех девяти разделов: `outputs/ct_preprocessing_lab/ui_verification.json`. Real forward capture: `real_model_verification.json`. Полные конфигурации запусков и commands: `ct_lab/README.md`. Synthetic fixtures используются только в тестах математики, не в исследовательских статистиках.', ''])
    (repo/'IMAGECHD68_DATA_ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('REPORT',len(rows),provenance,flush=True)


if __name__=='__main__': main()
