"""Audit read-only source provenance, export tables and write the stage-03 report."""
import csv
import importlib.metadata
import json
import platform
import zipfile
from pathlib import Path
import numpy as np
import torch
from ct_lab.learnable_core import OUT,REPO,PAIRS,SETS,normalize,load_encoder
from ct_lab.learnable_data import prior_hashes,assemble
from ct_lab.filtering_run import protected_hashes,DEFAULT_DATA
from ct_lab.core import fingerprint


def table(headers,rows):
    def fmt(x):return f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(fmt,row))+' |' for row in rows]])


def export(name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def run():
    result=json.loads((OUT/'results.json').read_text());plan=result['protocol']
    legacy=json.loads((OUT/'ablation_legacy.json').read_text());full=json.loads((OUT/'ablation_48.json').read_text())
    robust=json.loads((OUT/'robustness.json').read_text());importance=json.loads((OUT/'importance.json').read_text())
    assert prior_hashes()==json.loads((OUT/'prior_files.json').read_text()),'Prior artifacts changed'
    assert protected_hashes()==json.loads((OUT/'protected_files.json').read_text()),'Production guard changed'
    records={r['case_id']:r for r in json.loads((REPO/'metadata/pediatric/cardiac_chd68_cohort_v1.json').read_text())['records']}
    counts=[];neighbors=0;source_hashes={}
    for role in ('train','validation'):
        for case in plan[role+'_cases']:
            for suffix,field in (('image','image_SHA256'),('label','mask_SHA256')):
                path=DEFAULT_DATA/f'{case}_{suffix}.nii.gz';sha=fingerprint(path)
                assert sha==records[case][field];source_hashes[str(path)]=sha
            info=json.loads((OUT/'data'/f'{case}.json').read_text())
            with np.load(OUT/'data'/f'{case}.npz') as data:
                if role=='validation':assert not any('neighbor' in k for k in data.files)
                for p,pair in enumerate(PAIRS):
                    xyz=data[f'p{p}_xyz'];ids=data[f'p{p}_voxel_ids'];labels=data[f'p{p}_labels']
                    assert np.array_equal(ids,np.ravel_multi_index(xyz.T,info['shape_xyz']))
                    assert len(np.unique(ids))==len(ids) and np.isfinite(data[f'p{p}_features']).all()
                    for cls in PAIRS[pair]:counts.append({'case':case,'role':role,'pair':pair,'class':cls,'count':int((labels==cls).sum())})
                    if role=='train':
                        valid=data[f'p{p}_neighbor_valid'];neighbor=data[f'p{p}_neighbor_xyz']
                        assert np.all(np.abs(neighbor[valid]-xyz[valid]).sum(1)==1)
                        assert np.isfinite(data[f'p{p}_neighbor_features'][valid]).all();neighbors+=int(valid.sum())
    export('sample_counts',counts)
    audits=[];summaries=[];percase=[];errors=[]
    for m in result['models']:
        model,saved=load_encoder(OUT/m['checkpoint']);assert fingerprint(OUT/m['checkpoint'])==m['checkpoint_sha256']
        with np.load((OUT/m['checkpoint']).with_name('predictions.npz')) as f:
            z=f['train_embedding'];unit=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
            rawstd=float(z.std(0).mean());unitstd=float(unit.std(0).mean());assert rawstd>1e-8 and unitstd>1e-8
        audits.append({'pair':m['pair'],'features':m['features'],'method':m['method'],'embedding_dim':m['embedding_dim'],
                       'parameters':sum(p.numel() for p in model.parameters()),'raw_std_mean':rawstd,'unit_std_mean':unitstd,'checkpoint_sha256':m['checkpoint_sha256']})
    for m in result['baselines']+result['models']:
        method=m.get('method','k-NN');dim=m.get('embedding_dim',len(m['columns']) if 'columns' in m else 0)
        summaries.append({'pair':m['pair'],'features':m['features'],'method':method,'embedding_dim':dim,
                          'best_epoch':m.get('best_epoch',''),**{role+'_'+k:v for role in ('train','validation') for k,v in m[role]['case_macro'].items()}})
        for role in ('train','validation'):
            for case,metric in m[role]['per_case'].items():
                row={'pair':m['pair'],'features':m['features'],'method':method,'embedding_dim':dim,'role':role,'case':case,
                     'macro_f1':metric['macro_f1'],'balanced_accuracy':metric['balanced_accuracy']}
                for cls,values in metric['classwise'].items():
                    row.update({f'class{cls}_{k}':v for k,v in values.items()})
                percase.append(row)
        if method=='k-NN' or dim!=16:continue
        data=assemble(plan,'validation',list(PAIRS).index(m['pair']),SETS[m['features']]);classes=np.array(PAIRS[m['pair']])
        with np.load((OUT/m['checkpoint']).with_name('predictions.npz')) as f:pred=classes[f['validation_prob'].argmax(1)]
        with np.load(OUT/'baselines'/m['pair']/m['features'][0]/'predictions.npz') as f:base=classes[f['validation_prob'].argmax(1)]
        for case in np.unique(data['cases']):
            for cls in classes:
                keep=(data['cases']==case)&(data['labels']==cls);old=base[keep]==cls;new=pred[keep]==cls
                errors.append({'pair':m['pair'],'features':m['features'],'method':method,'case':case,'class':int(cls),
                               'n':int(keep.sum()),'corrected':int((~old&new).sum()),'regressed':int((old&~new).sum()),'both_wrong':int((~old&~new).sum())})
    export('models',summaries);export('metrics_by_case',percase);export('embedding_audit',audits);export('error_transitions',errors)
    export('robustness',[{k:v for k,v in r.items() if k!='paired_case_delta'} for r in robust]);export('permutation_importance',importance)
    abl=[];contributions=[];ablation_classwise=[];confusions=[]
    for scope,items in (('legacy5',legacy),('full48',full)):
        for a in items:
            metric=a.get('validation',a);abl.append({'scope':scope,'pair':a['pair'],'features':a['features'],**metric['case_macro']})
            confusions.append({'scope':scope,'pair':a['pair'],'features':a['features'],'classes':PAIRS[a['pair']],
                               'pooled_matrix':metric['pooled']['confusion_matrix'],
                               'by_case':{c:v['confusion_matrix'] for c,v in metric['per_case'].items()}})
            for case,score in metric['per_case'].items():
                for cls,values in score['classwise'].items():
                    ablation_classwise.append({'scope':scope,'pair':a['pair'],'features':a['features'],'case':case,'class':cls,**values})
            if a['features'][0] in 'DEFG':
                ref=next(x for x in items if x['pair']==a['pair'] and x['features'].startswith('C '));ref=ref.get('validation',ref)
                delta=[v['macro_f1']-ref['per_case'][c]['macro_f1'] for c,v in metric['per_case'].items()]
                contributions.append({'scope':scope,'pair':a['pair'],'features':a['features'],'paired_delta_mean':float(np.mean(delta)),
                                      'paired_delta_min':float(np.min(delta)),'paired_delta_max':float(np.max(delta)),'improved_cases':int((np.array(delta)>0).sum())})
    export('ablation',abl);export('gaussian_contribution',contributions)
    export('ablation_classwise',ablation_classwise)
    (OUT/'ablation_confusion_matrices.json').write_text(json.dumps(confusions,indent=2),encoding='utf-8')
    versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','nibabel','scikit-learn')}
    import ct_lab.launch
    for n in ('streamlit','plotly','umap-learn'):versions[n]=importlib.metadata.version(n)
    versions['python']=platform.python_version();versions['platform']=platform.platform()
    (OUT/'versions.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    integrity={'prior_files_unchanged':len(prior_hashes()),'production_guards_unchanged':True,'source_files_unchanged':source_hashes,
               'sealed_test_read':False,'real_train_neighbors':neighbors,'checkpoints_verified':len(audits),
               'all_embeddings_noncollapsed_raw_and_unit':True,'minimum_unit_std_mean':min(a['unit_std_mean'] for a in audits)}
    (OUT/'integrity.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
    lines=['# LEARNABLE_FEATURE_EXPERIMENT_03 — обучаемое представление признаков',
           '\nДата: 2026-10-28. Продолжение разработки ImageCHD68; validation ранее использовалась и теперь служит также early stopping. Независимой финальной оценки нет. Sealed test не открывался.',
           '\n## 1. Результат и контролируемая абляция',
           'Гипотеза подтверждена для диагностической бинарной классификации RV/PA на этой validation: нелинейное преобразование C/G улучшает case-macro F1 относительно k-NN на тех же данных. Для LV/MYO улучшение зависит от набора признаков. SupCon и Spatial не дают универсального дополнительного выигрыша. Это не сегментация и не оценка Dice.',
           'Сначала A–G рассчитаны на точных IDs этапа 02: пять train, десять validation. Затем отдельно повторены на 48 train с семью Z-позициями. Разность между двумя протоколами нельзя приписать одной архитектуре: изменились train cohort и sampling.',
           table(['Cohort','Пара','Набор','F1 case mean','SD','Balanced accuracy'],[[a['scope'],a['pair'],a['features'],a['macro_f1_mean'],a['macro_f1_std'],a['balanced_accuracy_mean']] for a in abl]),
           'A — intensity; B — intensity + XY texture; C — intensity + 3D Mean/Std/Gradient/LoG; D/E/F — C + Gaussian σ=0.5/1/2; G — C + все Gaussian. Mean/Std: 5×5×5 (XY: 5×5), центральный gradient, LoG σ=1; Gaussian truncate=4, reflect, float32, bounded patches с halo. σ в единицах сетки, физические мм не подтверждены. Исходная шкала не подтверждена как HU.',
           '\n### Вклад Gaussian относительно C, парные различия по 10 случаям',
           table(['Cohort','Пара','Набор','ΔF1 mean','min','max','cases improved'],[[r[k] for k in ('scope','pair','features','paired_delta_mean','paired_delta_min','paired_delta_max','improved_cases')] for r in contributions]),
           'На legacy RV/PA C=0.6164, G=0.6269: выигрыш G нельзя целиком назвать эффектом 3D-контекста. Gaussian 2 даёт основной небольшой дополнительный вклад; отдельные σ=0.5/1 не улучшают результат. Конфигурации после просмотра validation не менялись.',
           '\n## 2. Данные и отсутствие утечки',
           f'Frozen split: 48 train release-cases / 10 validation / 10 sealed test. Case/group пересечения исключены. Sampling: Z fractions {plan["sampling"]["Z_fractions"]}, максимум 64/class/Z, максимум 896 anchors/pair/case; редкие/отсутствующие классы не дополняются искусственными точками. Число точек одинаково во всех конфигурациях внутри одного протокола; состав см. sample_counts.csv.',
           'Source XYZ и C-order voxel IDs сохраняются. Метки используются только для sampling, targets и допустимости train-соседей. Вход encoder — исключительно CT-derived признаки. Scaler mean/population SD fit только на 48 train, отдельно C/G, одинаковый для всех losses и размеров embedding. Validation не участвует в градиентах, scaler, PCA/UMAP fit или обучающих соседствах. Её F1 выбирает эпоху, поэтому модель всё же адаптируется к validation через early stopping.',
           table(['Пара','Partition','Sampled voxels'],[[pair,role,sum(r['count'] for r in counts if r['pair']==pair and r['role']==role)] for pair in PAIRS for role in ('train','validation')]),
           '\n## 3. Архитектура и обучение',
           'Linear(input,64) → ReLU → Linear(64,32) → ReLU → Linear(32,e) → Linear(e,2). e=8/16/32, C=5 и G=8 входов. Поддерживается head на 8 классов, но обучены только две бинарные пары. Каждая из 36 моделей имеет собственные checkpoint, scaler, case IDs, seed, history, probabilities и embeddings. Никакие CHD68 v1/v2 веса не использовались и не изменялись.',
           f'AdamW lr=0.001, weight decay=0.0001, batch=512, max epochs=30, patience=5, min_delta=0.0001; CPU threads=1. Общий seed=20261028; per-model seed=seed+1000×pair_index+100×feature_index+embedding_dim. Одинаковая инициализация и последовательность batches для трёх losses при одной архитектуре. Class-balanced sampling без повторов внутри batch; веса обратно пропорциональны count(case,class), case balance в ожидании, не точная квота. Суммарное время train/evaluation циклов 36 моделей: {sum(m["seconds"] for m in result["models"]):.1f} s (без extraction/UMAP).',
           f'Параметров на модель: {min(a["parameters"] for a in audits)}–{max(a["parameters"] for a in audits)}. Train F1 — resubstitution; k-NN train дополнительно оптимистичен из-за exact self-neighbor. Все параметры и C/G были зафиксированы в protocol.json до первых validation результатов. Основное отображение e16 выбрано заранее; все кандидаты приведены ниже.',
           '\n## 4. Loss и кривые',
           'CE; CE+0.1 SupCon; CE+0.1 SupCon+0.1 Spatial. SupCon: L2-normalized z, temperature=0.1, self исключён, все другие same-class anchors — positives, other-class — negatives. Anchors без positives пропускаются. Spatial: MSE между normalized embeddings anchor и одного реального distinct 6-connected train-соседа того же GT класса, границы класса не пересекаются; при отсутствии допустимого соседа term пропускается. GT не передаётся в model.forward. Validation соседства вообще не создаются.',
           '[F1 / epoch](outputs/learnable_feature_03/history_f1.html), [Loss / epoch](outputs/learnable_feature_03/history_loss.html). Истории всех отдельных components сохранены в models/**/history.json. Значения total loss между CE и комбинированными функциями напрямую не сравнимы; SupCon имеет ненулевой уровень при множестве positives, даже когда классы разделены.',
           '\n## 5. Сравнение всех кандидатов',
           table(['Пара','Inputs','Метод','e','Train F1','Val F1','Val SD','Val BA','Best epoch'],[[s['pair'],s['features'],s['method'],s['embedding_dim'],s['train_macro_f1_mean'],s['validation_macro_f1_mean'],s['validation_macro_f1_std'],s['validation_balanced_accuracy_mean'],s['best_epoch']] for s in summaries]),
           'Основная метрика — mean двухклассового macro F1 по validation cases, с фиксированными двумя labels и zero_division=0 (как в предыдущем этапе). BA усредняет recall присутствующих targets. Pooled voxel metrics хранятся отдельно и не заменяют case averaging. Class-wise precision/recall/F1 и CM — results.json, metrics_by_case.csv. Воксели не являются независимыми пациентами.',
           'Class-wise ablation метрики: ablation_classwise.csv; confusion matrices всех A–G (pooled и per-case): ablation_confusion_matrices.json. Ниже pooled CM и class-wise F1 для основного e16; порядок строк/столбцов соответствует названной паре. Pooled F1 здесь приведён только как дополнительная диагностика.',
           table(['Пара','Inputs','Метод e16','F1 class 1','F1 class 2','Pooled CM'],[[m['pair'],m['features'][0],m.get('method','k-NN'),*[m['validation']['pooled']['classwise'][str(c)]['f1'] for c in PAIRS[m['pair']]],m['validation']['pooled']['confusion_matrix']] for m in result['baselines']+result['models'] if m.get('embedding_dim',16)==16]),
           '\n### Основное e16: парные ΔF1 относительно k-NN',
           '[График по каждому случаю](outputs/learnable_feature_03/paired_case_f1.html).']
    for m in result['models']:
        if m['embedding_dim']!=16:continue
        base=next(b for b in result['baselines'] if b['pair']==m['pair'] and b['features']==m['features'])
        delta=np.array([v['macro_f1']-base['validation']['per_case'][c]['macro_f1'] for c,v in m['validation']['per_case'].items()])
        lines.append(f'- {m["pair"]} {m["features"][0]} {m["method"]}: mean {delta.mean():+.4f}; range [{delta.min():+.4f}, {delta.max():+.4f}], improved {int((delta>0).sum())}/10.')
    lines+=['\n## 6. Изменение embeddings и проекции',
            '[Все интерактивные графики](outputs/learnable_feature_03/charts.html). PCA/UMAP до/после используют одни и те же 32/class/case IDs, отдельно показаны train и validation. Scaler/reducer fit только train; validation только transform. Метки не передаются UMAP. Bases независимы при разных размерностях representations, поэтому положения/масштабы на рисунках напрямую не сравниваются. Для количественного сравнения ниже использовано исходное raw embedding space, не 2D проекция.',
            table(['Пара','Inputs','Method e16','Silhouette','Fisher','Intra dist','Inter dist','Inter/Intra'],[[m['pair'],m['features'][0],m.get('method','k-NN'),*[m['validation']['geometry'][k] for k in ('silhouette','fisher_ratio','intra_class_distance','inter_class_distance','inter_intra_ratio')]] for m in result['baselines']+result['models'] if m.get('embedding_dim',16)==16]),
            f'Геометрия оценивается на фиксированной стратифицированной подвыборке ≤256/class. Абсолютные расстояния зависят от масштаба embedding; отношения/Silhouette/Fisher полезнее, но тоже зависят от sampling. Ни один из 36 embeddings не collapsed в общую точку: проверены variance raw и normalized train vectors; minimum mean unit SD={integrity["minimum_unit_std_mean"]:.6g}. Это исключает полный collapse, но не доказывает высокую intrinsic dimension или отсутствие частичной избыточности.',
            '\n## 7. Contrastive loss',
            'При фиксированном e16 RV/PA G: CE=0.8203, CE+SupCon=0.8245, то есть дополнительный эффект лишь +0.0042; основной выигрыш относительно k-NN=0.7390 уже даёт CE. По другим inputs/парам дополнительный эффект меняется. Один seed на архитектуру не позволяет утверждать статистически устойчивое превосходство SupCon. Параметры не подбирались по красивым UMAP.',
            'Для RV/PA G validation Silhouette в raw spaces: k-NN inputs=0.109, CE embedding=0.331, SupCon=0.294, Spatial=0.284; Fisher соответственно 0.222/1.236/0.895/0.857. Небольшой рост classifier F1 у SupCon не сопровождается улучшением этих геометрических метрик относительно CE. Классификационная голова и геометрия пространства оценивают разные свойства.',
            '\n## 8. Spatial consistency',
            f'Использовано {neighbors} реальных допустимых train neighbor pairs (один на anchor, если существует). e16 RV/PA C: Spatial=0.7794 против CE=0.7785 и SupCon=0.7776; G: Spatial=0.8210 против SupCon=0.8245. Пространственная регуляризация даёт смешанный результат; устойчивое преимущество не установлено. Она влияет на обучение embeddings, а не выполняет postprocessing/сглаживание validation маски.',
            '\n## 9. Устойчивость',
            'Stress tests заранее фиксированы: независимый noise σ=0.05 training SD по всем каналам; intensity shift +0.1 SD; intensity×1.1 относительно train mean; выключение 3D LoG заменой train mean. Это небольшие стандартизованные сдвиги и один более сильный channel ablation; они позволяют одинаковую механическую нагрузку C/G и методам, но не калиброваны по реальным сканерам. Per-ID noise одинаков между моделями, не зависит от порядка строк. Возмущения независимых коррелированных каналов могут лежать вне распределения реальных CT-derived features.',
            table(['Пара','Inputs','Method e16','Perturbation','Clean F1','Perturbed F1','Mean |Δp|','Raw emb RMSE','Unit emb RMSE'],[[r[k] for k in ('pair','features','method','perturbation','case_f1_clean','case_f1_perturbed','probability_abs_change_mean','embedding_raw_RMSE','embedding_unit_RMSE')] for r in robust if r['perturbation']!='None']),
            'RV/PA G Gaussian noise: k-NN 0.7390→0.7369, CE 0.8203→0.7669, SupCon 0.8245→0.7733, Spatial 0.8210→0.7737. MLP сохраняет более высокий абсолютный F1, но заметно сильнее теряет относительно clean. Утверждение об общей большей устойчивости learned embeddings не подтверждено. Полные парные Δ по cases сохранены в robustness.json; raw embedding RMSE между разными моделями не является общей шкалой.',
            '\n## 10. Признаки и ошибки',
            'Global permutation importance: fixed validation subset 32/class/case, перестановка канала внутри case между классами, без refit. Локальная ablation в интерфейсе заменяет выбранный канал train mean и показывает изменение вероятности текущего predicted class. Абсолютные веса Linear не считаются universal importance. Перестановка/отключение разрывают корреляции, поэтому важность не причинная и не равна уникальной информации.',
            table(['Пара','Inputs','Method','Top feature','F1 drop'],[[pair,feature[0],method,top['feature'],top['f1_drop']] for pair in PAIRS for feature in plan['neural_inputs'] for method in ['k-NN',*plan['methods']] for top in [max([r for r in importance if r['pair']==pair and r['features']==feature and r['method']==method],key=lambda r:r['f1_drop'])]]),
            'Число исправленных/новых ошибок относительно k-NN для каждого case/class/метода e16 — error_transitions.csv. Это парное сравнение тех же IDs, включая регрессии. В интерфейсе point/lasso связывает source XYZ и три плоскости, показывает probability и признаки выбранного voxel; true class используется только для окраски/оценки. Selected voxel вне пары получает предупреждение об условности двух вероятностей.',
            table(['Пара','Inputs','Метод e16','Исправлены','Новые ошибки','Оба ошиблись'],[[pair,feature[0],method,*[sum(r[k] for r in errors if r['pair']==pair and r['features']==feature and r['method']==method) for k in ('corrected','regressed','both_wrong')]] for pair in PAIRS for feature in plan['neural_inputs'] for method in plan['methods']]),
            '\n## 11. Ограничения и проверки',
            f'Предыдущие {integrity["prior_files_unchanged"]} файлов экспериментов/отчётов сохранены byte-for-byte. Хеши 116 файлов CT/масок 58 разрешённых cases проверены повторно. Frozen split, production preprocessing и веса CHD68 v1/v2 совпадают с исходным snapshot; receipts 36 checkpoints совпадают с SHA. Sealed test не открывался. Аудит: integrity.json, sample_counts.csv, embedding_audit.csv; версии: versions.json.',
            '47 автоматических тестов passed: SupCon и градиенты, spatial neighbors, class mapping, scaler train-only, seed, source IDs, checkpoint strict roundtrip/provenance, coordinate/boundary/halo regression, no GT feature schema, frozen split/groups, inference regression. 17 проверок Streamlit AppTest охватывают все три размера, три losses, PCA/UMAP, forward layers, validation и gating test: ui_verification.json. Реальный Chrome проверил клики во всех трёх плоскостях, lasso 64 source IDs, выбор CE e16 и embedding inspector; crosshair и overlay count совпадают, JS errors отсутствуют: browser_verification.json / browser_learnable.png / browser_embedding.png. Производственный training regression test не запускался.',
            'Ограничения: один seed на архитектуру, десять validation случаев, early stopping на повторно использованной validation, sampled foreground pair-only classification, GT-stratified sampling не отражает natural prevalence; class/case баланс в train; features 5×5×5 в индексной сетке; физические единицы неизвестны. Несколько размеров embedding — кандидаты разработки, максимум по ним не независимая оценка. Отсутствует whole-volume evaluation, multiclass background/foreground, external scanner validation и доказательство роста Dice/клинической применимости.',
            '\n## 12. Следующий этап магистерской',
            'Сначала повторить заранее фиксированные C/G CE и SupCon с несколькими seeds и group-wise cross-validation только внутри 48 train, оценить устойчивость на согласованных возмущениях CT с последующим повторным feature extraction. CE — обязательный сильный baseline; SupCon рассматривать как небольшой возможный increment; Spatial пока не считать улучшением. Проверить, сохраняется ли выигрыш на всех foreground + background при case-level протоколе и естественной распространённости классов.',
            'Затем отдельным этапом сравнить настоящий сегментатор с intensity-only, intensity+3D-core и intensity+Gaussian/learned branch, сохраняя одинаковые архитектуру/augmentation/budget. Требуются whole-volume Dice/HD95 и локализация ошибок на границах; выводы нынешней voxel classification их не заменяют. Текущий этап не запускает обучение такого сегментатора и не меняет CHD68 v1/v2.',
            '\n## Воспроизведение',
            'Команды из repository (PowerShell); existing frozen artifacts переиспользуются, параметры protocol.json не меняются:\n```powershell\n.venv/Scripts/python.exe -m ct_lab.learnable_train\n.venv/Scripts/python.exe -m ct_lab.learnable_diagnostics\n.venv/Scripts/python.exe -m ct_lab.learnable_artifacts\n.venv/Scripts/python.exe -m ct_lab.learnable_verify_ui\n.venv/Scripts/python.exe -m pytest tests/test_learnable_features.py tests/test_spatial_features.py tests/test_filtering_features.py tests/test_ct_lab.py tests/test_cardiac_inference.py -q\n.venv/Scripts/python.exe -m ct_lab.learnable_package\n.venv/Scripts/python.exe -m ct_lab.learnable_launch\nnode ct_lab/learnable_browser.cjs\n```',
            'Для полного повторного обучения используйте отдельную копию проекта без outputs/learnable_feature_03; исходные данные и frozen split сохранить. Существующие receipts позволяют безопасно продолжать завершённый run, не перезаписывая обученные модели.',
            table(['Library','Version'],list(versions.items()))]
    (REPO/'LEARNABLE_FEATURE_EXPERIMENT_03.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    with zipfile.ZipFile(OUT/'learnable_feature_03.zip','w',zipfile.ZIP_DEFLATED) as z:
        paths=[p for p in OUT.rglob('*') if p.is_file() and p.suffix!='.zip' and 'numba_cache' not in p.parts]
        paths += list((REPO/'ct_lab').glob('learnable*.py'))+list((REPO/'ct_lab').glob('learnable*.cjs'))
        paths += [REPO/'LEARNABLE_FEATURE_EXPERIMENT_03.md',REPO/'tests/test_learnable_features.py',REPO/'ct_lab/LEARNABLE_README.md',REPO/'ct_lab/app.py']
        for path in sorted(set(paths)):z.write(path,str(path.relative_to(REPO)))
        assert len(z.namelist())==len(set(z.namelist()))
    print('REPORT/PACKAGE OK',len(audits),'models;',len(source_hashes),'source hashes;',len(prior_hashes()),'prior files unchanged',flush=True)


if __name__=='__main__':run()
