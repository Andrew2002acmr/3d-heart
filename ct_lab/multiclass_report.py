"""Measured figures, source/protected integrity receipts and reproducible report."""
import csv
import importlib.metadata
import json
import platform
import zipfile
import nibabel as nib
import numpy as np
import ct_lab.launch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ct_lab.multiclass_core import *
from ct_lab.multiclass_metrics import bootstrap
from ct_lab.multiclass_projection import projection
from ct_lab.multiclass_train import slug


def table(headers,rows):
    def fmt(v):
        if isinstance(v,(list,tuple)):return '['+', '.join(fmt(x) for x in v)+']'
        return f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(fmt,row))+' |' for row in rows]])


def csv_file(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def run():
    result=json.loads((OUT/'results.json').read_text());plan=result['protocol'];summary=json.loads((OUT/'summary.json').read_text())
    dense=json.loads((OUT/'dense_summary.json').read_text());importance=json.loads((OUT/'importance.json').read_text());correlation=json.loads((OUT/'correlations.json').read_text())
    clean=[s for s in summary if s['transformation']=='clean'];noise=[s for s in summary if s['transformation']=='noise12'];charts=[]
    def chart(name,fig,explanation):
        fig.update_layout(template='plotly_white');path=OUT/'charts'/(name+'.html');path.parent.mkdir(exist_ok=True)
        path.write_text('<meta charset="utf-8"><p>'+explanation+'</p>'+fig.to_html(full_html=False,include_plotlyjs=True),encoding='utf-8');charts.append((name,explanation))
    def key(s):return s['features'][0]+' '+s['method']+' '+s['augmentation']
    primary=[s for s in clean if s['features']=='E core+all' and s['method']!='k-NN']
    fig=go.Figure()
    for method in ('k-NN','CE'):
        points=[b for b in result['baselines']] if method=='k-NN' else [s for s in clean if s['method']=='CE' and s['augmentation']=='clean']
        fig.add_trace(go.Bar(x=[p['features'] for p in points],y=[p['validation']['case_macro']['macro_f1_mean'] if method=='k-NN' else p['seed_mean_F1'] for p in points],
                            error_y={'type':'data','array':[0 if method=='k-NN' else p['seed_SD_F1'] for p in points]},name=method))
    chart('feature_sets',fig.update_layout(title='Natural validation: F1 vs Feature Set'), 'Одинаковые spatial IDs, естественные частоты. CE — среднее трёх seed; error bars — SD между seed, не CI пациентов.')
    fig=go.Figure()
    for s in primary:fig.add_trace(go.Scatter(x=list(s['per_case_seed_mean']),y=list(s['per_case_seed_mean'].values()),mode='lines+markers',name=key(s)))
    chart('F1_by_case',fig.update_layout(title='F1 по исследованиям · seed mean'),'Единица наблюдения — release-case. Разброс между случаями не заменяется количеством sampled voxels.')
    fig=go.Figure()
    for s in primary:fig.add_trace(go.Bar(x=LABELS,y=list(s['per_class_F1_seed_mean'].values()),name=key(s)))
    chart('F1_by_class',fig.update_layout(title='F1 по классам · natural validation'),'Class-wise F1 усреднён между тремя seed; background включён в задачу.')
    fig=go.Figure();stability=go.Figure();sensitivity=go.Figure()
    robust=json.loads((OUT/'robustness.json').read_text())
    for s in primary:
        parts=[r for r in robust if all(r[k]==s[k] for k in ('features','method','augmentation'))]
        names=list(VAL_AUG);means=[np.mean([r['metrics']['case_macro']['macro_f1_mean'] for r in parts if r['transformation']==n]) for n in names]
        fig.add_trace(go.Scatter(x=names,y=means,mode='lines+markers',name=key(s)))
        stability.add_trace(go.Bar(x=names,y=[np.mean([r['embedding_unit_RMSE'] for r in parts if r['transformation']==n]) for n in names],name=key(s)))
        noisy=next(r for r in noise if all(r[k]==s[k] for k in ('features','method','augmentation')))
        sensitivity.add_trace(go.Bar(x=LABELS,y=[noisy['per_class_F1_seed_mean'][str(c)]-s['per_class_F1_seed_mean'][str(c)] for c in CLASSES],name=key(s)))
    chart('clean_vs_noisy',fig.update_layout(title='Clean vs heldout CT transformations'), 'Преобразованы реальные CT, затем повторно извлечены признаки. Параметры validation отличны от train; не модель реальных сканеров. Для translations парные Δ в JSON используют common-valid IDs.')
    chart('embedding_stability',stability.update_layout(title='Unit embedding RMSE'), 'L2-normalized embeddings сравниваются на тех же source IDs; меньше — стабильнее, но это не метрика Dice.')
    chart('class_noise_sensitivity',sensitivity.update_layout(title='Noise12 minus clean: F1 по классам'), 'Отрицательное значение показывает чувствительность класса к Gaussian noise на исходном изображении, σ=12 source counts.')
    principal=next(m for m in result['models'] if m['features']=='E core+all' and m['method']=='CE' and m['augmentation']=='combined' and m['seed']==SEEDS[0])
    chart('confusion_8x8',go.Figure(go.Heatmap(z=principal['validation']['pooled']['confusion_matrix'],x=LABELS,y=LABELS,texttemplate='%{z}',colorscale='Blues')).update_layout(title='8×8 Confusion Matrix · E CE combined first seed'), 'Valid 0–7 voxels, ignore=255 исключён. Это pooled диагностическая таблица; основная метрика усредняется по cases.')
    fig=go.Figure()
    for variant in sorted({(r['method'],r['augmentation']) for r in importance}):
        rows=[r for r in importance if (r['method'],r['augmentation'])==variant]
        fig.add_trace(go.Bar(x=[r['feature'] for r in rows],y=[r['F1_drop'] for r in rows],name='/'.join(variant)))
    chart('importance',fig.update_layout(title='Permutation importance · E first seed'), 'Фиксированные natural validation IDs, перестановка внутри case. Нарушение корреляций ограничивает причинную интерпретацию; параметры после этого не менялись.')
    chart('correlations',go.Figure(go.Heatmap(z=correlation['matrix'],x=FEATURES,y=FEATURES,zmin=-1,zmax=1,colorscale='RdBu')), 'Корреляции только по train. Дополнительные Z-каналы ограничены тремя; feature ablations A–F проверены отдельно.')
    fig=go.Figure()
    for d in dense:
        if d['scope']=='full native volume':continue
        bins=d['background_distance_bins_index_units']
        fig.add_trace(go.Scatter(x=list(bins),y=[v['FPR'] for v in bins.values()],mode='lines+markers',name=d['case']+' '+d['model']))
    chart('background_distance_errors',fig.update_layout(title='Background → foreground FPR vs GT distance (index units)'), 'Distance до GT вычисляется только после inference для диагностики. Одинаковые dense ROI; расстояния не feature и не правило выбора ROI.')
    fig=go.Figure()
    for case in plan['dense_cases']:
        entries=[d for d in dense if d['case']==case and d['scope']!='full native volume']
        fig.add_trace(go.Bar(x=[d['model'] for d in entries],y=[sum(d['dense_classwise'][str(c)]['false_components_no_GT_overlap'] for c in range(1,8)) for d in entries],name=case))
    chart('false_components',fig.update_layout(title='False 3D components · dense ROI, six-connectivity'),'Компонент не пересекает GT того же класса; A/C CE clean изолируют дополнительный 3D context. E combined меняет и признаки, и augmentation. ROI компоненты усечены; полное ct_1001 измерено отдельно.')
    for metric in ('loss','validation_natural_case_f1'):
        fig=go.Figure()
        for m in result['models']:
            history=json.loads((OUT/m['checkpoint']).with_name('history.json').read_text())
            fig.add_trace(go.Scatter(x=[h['epoch'] for h in history],y=[h[metric] for h in history],name=slug(m),visible=True if m['seed']==SEEDS[0] and m['features']=='E core+all' else 'legendonly'))
        chart('training_'+metric,fig.update_layout(title=metric+' / epoch'), 'Early stopping на reused development validation; разные loss имеют разные масштабы. Истории всех 36 моделей сохранены.')
    for kind in ('PCA','UMAP'):
        configurations=[({'features':'E core+all','method':m,'augmentation':a,'seed':SEEDS[0]},rep) for m,a,rep in [('CE','clean','Inputs'),('CE','clean','Embedding'),('CE','combined','Embedding'),('CE+SupCon','combined','Embedding')]]
        fig=make_subplots(rows=4,cols=4,subplot_titles=[f'{role} {condition} · {j[0]["method"]}/{j[0]["augmentation"]} {j[1]}' for role,condition in [('train','clean'),('train','noise8'),('validation','clean'),('validation','noise12')] for j in configurations])
        for col,(job,rep) in enumerate(configurations,1):
            sample,_=projection(job,kind,rep)
            for row,(role,condition) in enumerate([('train','clean'),('train','noise8'),('validation','clean'),('validation','noise12')],1):
                for c in CLASSES:
                    keep=(sample['role']==role)&(sample['condition']==condition)&(sample['labels']==c)
                    fig.add_trace(go.Scattergl(x=sample['embedding'][keep,0].tolist(),y=sample['embedding'][keep,1].tolist(),mode='markers',name=LABELS[c],legendgroup=LABELS[c],showlegend=row==1 and col==1,marker={'size':2,'color':['#9aa0a6','#ef476f','#118ab2','#9b5de5','#f15bb5','#06d6a0','#fb8500','#ffd166'][c]}),row=row,col=col)
        chart('projection_'+kind,fig.update_layout(height=1450,title=kind+' · before/after, clean/noisy'), 'Реальные ImageCHD68 voxels, фиксированные IDs. Общая clean TRAIN basis внутри каждой колонки; разные representations имеют независимые bases. Train diagnostic σ=8, heldout validation σ=12. Вид кластеров не доказывает рост сегментационного качества.')
    # Dense native-grid slice panels, all methods use exactly the same ROI and source GT.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cmap=ListedColormap(['#000000','#ef476f','#118ab2','#9b5de5','#f15bb5','#06d6a0','#fb8500','#ffd166'])
    for case in plan['dense_cases']:
        entries=[d for d in dense if d['case']==case and d['scope'].startswith('predeclared')];source=nib.load(DEFAULT_DATA/f'{case}_image.nii.gz');x=source.get_fdata(dtype=np.float32)
        y=cardiac_target(np.asanyarray(nib.load(DEFAULT_DATA/f'{case}_label.nii.gz').dataobj));lo=np.array(entries[0]['lower_xyz']);hi=np.array(entries[0]['upper_xyz']);z=(lo[2]+hi[2])//2
        fig,axes=plt.subplots(1,len(entries)+2,figsize=(3*(len(entries)+2),4));sl=(slice(lo[0],hi[0]),slice(lo[1],hi[1]),z)
        for i,ax in enumerate(axes):
            ax.imshow(x[sl].T,cmap='gray',origin='lower',vmin=0,vmax=4095)
            if i==0:ax.set_title('Исходная КТ',fontsize=9);ax.axis('off');continue
            elif i==1:seg=y[sl].T;title='GT'
            else:
                d=entries[i-2];pred=np.load(OUT/'dense'/case/d['model']/'prediction.npy',mmap_mode='r');seg=pred[:,:,z-lo[2]].T;title=d['model']
            ax.imshow(np.ma.masked_where((seg==0)|(seg==255),seg),cmap=cmap,vmin=0,vmax=7,alpha=.55,origin='lower');ax.set_title(title.replace('_','\n'),fontsize=8);ax.axis('off')
        fig.legend(handles=[Patch(color=cmap(c/7),label=LABELS[c]) for c in range(1,8)],loc='lower center',ncol=7,frameon=False)
        fig.suptitle(case+' · фиксированная ROI, исходная сетка · Z='+str(z));fig.tight_layout(rect=(0,.08,1,.96));fig.savefig(OUT/f'dense_{case}.png',dpi=130);plt.close(fig)
    # Flatten tables while preserving all class metrics and per-case differences.
    model_rows=[];case_rows=[];class_rows=[]
    for m in result['models']:
        head={k:m[k] for k in ('features','method','augmentation','seed')};model_rows.append(head|{'best_epoch':m['best_epoch'],'epochs_run':m['epochs_run'],'seconds':m['seconds'],
              'train_natural_F1':m['train_natural']['case_macro']['macro_f1_mean'],**m['validation']['case_macro']})
        for role in ('train_natural','validation'):
            for case,s in m[role]['per_case'].items():
                case_rows.append(head|{'role':role,'case':case,'F1':s['macro_f1'],'BA':s['balanced_accuracy']})
                for c,values in s['classwise'].items():class_rows.append(head|{'role':role,'case':case,'class':c}|values)
    csv_file('models',model_rows);csv_file('metrics_by_case',case_rows);csv_file('metrics_by_class',class_rows)
    csv_file('robustness',[{k:r[k] for k in ('features','method','augmentation','seed','transformation','probability_abs_change_mean','embedding_raw_RMSE','embedding_unit_RMSE')}|
                          {'F1':r['metrics']['case_macro']['macro_f1_mean'],'BA':r['metrics']['case_macro']['balanced_accuracy_mean'],'paired_F1_delta':float(np.mean(list(r['case_delta'].values())))} for r in robust])
    csv_file('dense_metrics',[{'case':d['case'],'model':d['model'],'scope':d['scope'],'class':LABELS[c]}|{k:v for k,v in d['dense_classwise'][str(c)].items() if k!='component_sizes_desc'} for d in dense for c in CLASSES])
    csv_file('importance',[{k:v for k,v in r.items() if k!='per_class_F1_drop'} for r in importance])
    paired=[]
    for s in clean:
        if s['method']=='k-NN':continue
        reference=next(b for b in result['baselines'] if b['features']==s['features'])
        values=[v-reference['validation']['per_case'][c]['macro_f1'] for c,v in s['per_case_seed_mean'].items()]
        paired.append({'features':s['features'],'method':s['method'],'augmentation':s['augmentation'],**bootstrap(values)})
    write(OUT/'paired_case_differences.json',paired)
    eclean=next(s for s in clean if s['features']=='E core+all' and s['method']=='CE' and s['augmentation']=='clean')
    ecombo=next(s for s in clean if s['features']=='E core+all' and s['method']=='CE' and s['augmentation']=='combined')
    supcombo=next(s for s in clean if s['features']=='E core+all' and s['method']=='CE+SupCon' and s['augmentation']=='combined')
    spatial=next(s for s in clean if s['features']=='E core+all' and s['method']=='CE+Spatial')
    ns=[next(n for n in noise if all(n[k]==s[k] for k in ('features','method','augmentation'))) for s in (eclean,ecombo,supcombo)]
    cbase=next(s for s in clean if s['features']=='C 3D-core' and s['method']=='CE')
    dbase=next(s for s in clean if s['features']=='D core+Gaussian2' and s['method']=='CE')
    fbase=next(s for s in clean if s['features']=='F core+Z' and s['method']=='CE')
    supclean=next(s for s in clean if s['features']=='E core+all' and s['method']=='CE+SupCon' and s['augmentation']=='clean')
    ablation_pairs=[]
    for label,left,right in [('Gaussian2: D−C',dbase,cbase),('All Gaussian: E−C',eclean,cbase),('Z context: F−C',fbase,cbase),
                             ('Combined augmentation: E CE combined−clean',ecombo,eclean),('SupCon combined−CE combined',supcombo,ecombo),
                             ('SupCon clean−CE clean',supclean,eclean),('Spatial clean−CE clean',spatial,eclean)]:
        values=[left['per_case_seed_mean'][c]-v for c,v in right['per_case_seed_mean'].items()]
        ablation_pairs.append({'comparison':label,**bootstrap(values)})
    write(OUT/'paired_ablation_differences.json',ablation_pairs)
    spatial_ci=ablation_pairs[-1]['case_bootstrap_CI95'];sup_ci=ablation_pairs[-3]['case_bootstrap_CI95']
    baseline=next(b for b in result['baselines'] if b['features']=='E core+all')
    rank=sorted(range(1,8),key=lambda c:ecombo['per_class_F1_seed_mean'][str(c)],reverse=True)
    full=[d for d in dense if d['scope']=='full native volume'];full_mlp=next(d for d in full if d['model'].startswith('E_'))
    def mean_dice(d):return float(np.mean([d['dense_classwise'][str(c)]['dice'] for c in range(1,8) if d['dense_classwise'][str(c)]['dice'] is not None]))
    off=np.array(principal['validation']['pooled']['confusion_matrix']);np.fill_diagonal(off,0)
    pairs=sorted([(int(off[a,b]),LABELS[a],LABELS[b]) for a in range(1,8) for b in range(1,8) if a!=b],reverse=True)[:3]
    measured_answers=[
        f'Восьмиклассовое преимущество: E CE clean {eclean["seed_mean_F1"]:.4f} против E k-NN {baseline["validation"]["case_macro"]["macro_f1_mean"]:.4f}; combined CE {ecombo["seed_mean_F1"]:.4f}. Это natural development F1, а не прежняя binary задача.',
        'Лучшие классы у заранее выбранной E CE combined: '+', '.join(f'{LABELS[c]}={ecombo["per_class_F1_seed_mean"][str(c)]:.4f}' for c in rank[:3])+'. Трудные: '+', '.join(f'{LABELS[c]}={ecombo["per_class_F1_seed_mean"][str(c)]:.4f}' for c in rank[-3:])+'.',
        'Наиболее частые FG→FG путаницы first seed: '+', '.join(f'{a}→{b}: {n}' for n,a,b in pairs)+'. Background ошибки показаны отдельно в CM/FPR.',
        f'Gaussian2: D CE clean {dbase["seed_mean_F1"]:.4f} против C {cbase["seed_mean_F1"]:.4f}, Δ={dbase["seed_mean_F1"]-cbase["seed_mean_F1"]:+.4f}. Канал не считается полезным автоматически.',
        f'CT augmentation/noise12: E clean-trained CE {ns[0]["seed_mean_F1"]:.4f}, combined-trained CE {ns[1]["seed_mean_F1"]:.4f}; clean F1 соответственно {eclean["seed_mean_F1"]:.4f}/{ecombo["seed_mean_F1"]:.4f}. Это измерение переноса на новый synthetic recipe, не реальный scanner domain.',
        f'SupCon поверх combined CE: clean Δ={supcombo["seed_mean_F1"]-ecombo["seed_mean_F1"]:+.4f}, noise12 Δ={ns[2]["seed_mean_F1"]-ns[1]["seed_mean_F1"]:+.4f}. Clean paired case CI95=[{sup_ci[0]:+.4f}, {sup_ci[1]:+.4f}] включает ноль: убедительное дополнительное преимущество не установлено.',
        f'Spatial E clean Δ к CE={spatial["seed_mean_F1"]-eclean["seed_mean_F1"]:+.4f}; paired case CI95=[{spatial_ci[0]:+.4f}, {spatial_ci[1]:+.4f}] включает ноль. Убедительное устойчивое улучшение не установлено.',
        f'E CE combined: SD(seed)={ecombo["seed_SD_F1"]:.4f}, SD(case seed-means)={ecombo["case_summary"]["case_SD"]:.4f}, case CI95=[{ecombo["case_summary"]["case_bootstrap_CI95"][0]:.4f}, {ecombo["case_summary"]["case_bootstrap_CI95"][1]:.4f}]. Интервал условен на reused development протоколе.',
        'Dense prediction выполнен на полном ct_1001 и fixed ROI трёх случаев. Классовые Dice/HD95 и примеры ниже являются измеренными dense результатами, не следствием sampled F1.',
        f'Полное ct_1001 MLP: {sum(full_mlp["dense_classwise"][str(c)]["false_components_no_GT_overlap"] for c in range(1,8))} false foreground components без overlap с GT того же класса; BG→FG={sum(full_mlp["confusion_matrix"][0][1:])} voxels. Это реальный dense connected-component analysis, не sparse fragment count.',
        'Полное ct_1001 mean Dice present FG: '+', '.join(f'{d["model"]}={mean_dice(d):.4f}' for d in full)+'. MLP существенно уступает CHD68 в этом train-case сравнении; отдельно приведены development ROI. Локальный encoder не подтвердил пригодность как самостоятельная основа whole-heart segmentation.',
        'Для следующего сегментатора необходим global/multiscale anatomical context и более реалистичный background/prior protocol. Class-balanced MLP training не отражает natural priors; calibration/prior correction требует отдельного train-only эксперимента, в этой frozen матрице не подбиралась.']
    # Immutable inputs and all prior experiment artifacts, including their caches.
    assert prior_hashes()==json.loads((OUT/'prior_files.json').read_text()),'Earlier artifacts changed'
    assert protected_hashes()==json.loads((OUT/'protected_files.json').read_text()),'CHD68/split/preprocessing changed'
    _,records=freeze();hashes={};counts=[]
    for role in ('train','validation'):
        for case in plan[role+'_cases']:
            for suffix,field in [('image','image_SHA256'),('label','mask_SHA256')]:
                path=DEFAULT_DATA/f'{case}_{suffix}.nii.gz';sha=fingerprint(path);assert sha==records[case][field];hashes[str(path)]=sha
            info=json.loads((OUT/'data'/f'{case}.json').read_text());counts.append({'case':case,'role':role,**info['counts_natural']})
            with np.load(OUT/'data'/f'{case}.npz') as f:
                assert np.array_equal(f['natural_ids'],np.ravel_multi_index(f['natural_xyz'].T,info['shape_xyz']))
                if role=='validation':assert not any('neighbor' in n or n.startswith('train_') for n in f.files)
                else:assert not np.any(f['train_labels']==255)
    for m in result['models']+result['CV']:
        model,saved=load_model(OUT/m['checkpoint']);assert fingerprint(OUT/m['checkpoint'])==m['checkpoint_sha256']
        assert m['unit_embedding_std_mean']>1e-8
    csv_file('natural_class_counts',counts)
    dense_audit=[];audit_rng=np.random.default_rng(SEEDS[0])
    for d in dense:
        directory=OUT/('dense_full' if d['scope']=='full native volume' else 'dense')/d['case']/d['model']
        pred=np.load(directory/'prediction.npy',mmap_mode='r');prob=np.load(directory/'probabilities.npy',mmap_mode='r');entropy=np.load(directory/'entropy.npy',mmap_mode='r')
        assert prob.shape==(*pred.shape,8) and entropy.shape==pred.shape
        ids=audit_rng.choice(pred.size,min(512,pred.size),replace=False);xyz=np.unravel_index(ids,pred.shape);q=prob[xyz].astype(np.float32)
        assert np.isfinite(q).all() and np.all((q>=0)&(q<=1));np.testing.assert_allclose(q.sum(1),1,atol=.002)
        e=entropy[xyz];assert np.isfinite(e).all() and np.all((e>=-1e-6)&(e<=np.log(8)+1e-5))
        assert np.array_equal(np.asanyarray(nib.load(directory/'prediction.nii.gz').dataobj),pred)
        dense_audit.append({'case':d['case'],'model':d['model'],'scope':d['scope'],'mask_NIfTI_roundtrip_exact':True,'probability_entropy_samples_checked':len(ids)})
    write(OUT/'integrity.json',{'source_hashes':hashes,'prior_files_unchanged':len(prior_hashes()),'production_guards_unchanged':True,'sealed_test_read':False,
                               'checkpoints_verified':len(result['models'])+len(result['CV']),'all_embeddings_noncollapsed':True,'ROI_GT_input':False,'dense_output_audit':dense_audit})
    import ct_lab.launch
    versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','nibabel','scikit-learn','umap-learn','plotly','streamlit')};versions['python']=platform.python_version()
    write(OUT/'versions.json',versions)
    lines=['# MULTICLASS_FEATURE_EXPERIMENT_04',
           'Начат 2026-10-28; завершён 2026-10-29. Восьмиклассовое representation learning на ImageCHD68, продолжение разработки. Validation многократно использовалась; независимой финальной/клинической оценки нет. Test закрыт. Полноценная сегментационная сеть не обучалась; CHD68 v1/v2 только inference.',
           '## 1. Протокол и границы выводов',
           '### Ответы на исследовательские вопросы\n\n'+'\n\n'.join(f'{i+1}. {answer}' for i,answer in enumerate(measured_answers)),
           'Frozen 48 train / 10 development validation / 10 sealed test. 0 Background, 1 LV, 2 RV, 3 LA, 4 RA, 5 MYO, 6 AO, 7 PA. Прочие source labels → ignore=255, исключаются из loss и метрик; ignore не background. Полная предрегистрация: outputs/multiclass_feature_04/protocol.json. 36 основных моделей, три seed: '+str(SEEDS)+', плюс шесть обучений для 3-fold group-wise CV внутри train. Case/group не делятся между folds.',
           'Все A–F и augmentation/loss варианты зафиксированы до validation. Основная метрика — mean fixed-eight-label macro F1 по cases (absent labels: zero_division=0); дополнительно present-class macro, foreground macro и BA по присутствующим targets. Macro F1 на natural samples и ROI/full-volume Dice — различные измерения, одно не выводится из другого.',
           '## 2. Sampling и natural validation',
           'Train: 16 Z-strata, 128 uniform points/class + до 32 boundary points/class. Background: 64 near (6-connected dilation radius 6 index steps) + 64 far, затем boundary extra. Экспертные masks разрешены только для train sampling/targets и same-class spatial loss. Natural train: 4096, validation: 8192 равномерных уникальных IDs/case по всему source volume; IDs выбираются по image shape до чтения GT. Foreground и ignore не исключаются при выборе. Маска затем только задаёт targets/valid для оценки. Это spatial point sample естественного распределения, дополненный отдельным dense ROI/full-volume экспериментом.',
           'Source C-order voxel IDs/XYZ/seed/case сохранены. Natural class counts: natural_class_counts.csv. Scaler fit только clean train anchors текущего fold, никогда validation; одинаковые scaler и anchors между losses/seed для одного набора. GT, distance-to-GT и absolute coordinates не входят в признаки.',
           table(['Partition','Cases','Natural selected points','Valid 0–7','Ignored'],[[role,sum(r['role']==role for r in counts),sum(sum(r[str(c)] for c in CLASSES+[255]) for r in counts if r['role']==role),sum(sum(r[str(c)] for c in CLASSES) for r in counts if r['role']==role),sum(r['255'] for r in counts if r['role']==role)] for role in ('train','validation')]),
           '## 3. Архитектура, признаки и обучение',
           'Input → Linear64 → ReLU → Linear32 → ReLU → Linear16 → Linear8. CE; CE+0.1 SupCon (L2-normalized, temperature=.1, self excluded); CE+0.1 Spatial (без SupCon, в отличие от этапа03). Spatial использует distinct 6-neighbor того же train класса, без пересечения GT границ, только как loss; отсутствующие пары пропускаются. Восемь классов поддерживаются существующим FeatureEncoder; веса CHD68 не являются инициализацией.',
           table(['Набор','Каналы'],[[k,', '.join(v)] for k,v in SETS.items()]),
           'Uniform mean/std window 5; central gradients с one-sided global edges; LoG σ=1; Gaussian .5/1/2, truncate4, reflect. F добавляет только Z−1, Z+1 и Z local SD, не десятки дублирующих каналов. Source intensity не HU. Sigma/spacing — неизвестные header grid units, не мм. Новый point extractor повторно использует stage02 stencil/kernels; dense engine сверяется с ним на всех краях и стыках блоков.',
           'AdamW lr=.001, wd=.0001, batch256, максимум20 эпох, patience4, min_delta=.0002, CPU threads1. Class-balanced batches, внутри класса inverse count(case,class) sampling без replacement внутри batch. Early stopping по natural development case F1; одинаковая инициализация при одинаковом input/seed. Train augmentation: 50% clean / 50% выбранная augmented version; valid translation сохраняет тот же анатомический target, вне FOV используется clean вместо испорченной пары. Основной embedding=16, API допускает8/32, в этом этапе dimension search не выполняется.',
           '## 4. Сохраняется ли преимущество MLP? Все clean результаты',
           table(['Inputs','Метод','Train augmentation','F1 mean(seed)','SD(seed)','SD(case means)','Case bootstrap CI95','BA mean'],[[s['features'],s['method'],s['augmentation'],s['seed_mean_F1'],s['seed_SD_F1'],s['case_summary']['case_SD'],s['case_summary']['case_bootstrap_CI95'],float(np.mean([r['metrics']['case_macro']['balanced_accuracy_mean'] for r in robust if all(r[k]==s[k] for k in ('features','method','augmentation')) and r['transformation']=='clean']))] for s in clean]),
           'Сравнение с k-NN делается на тех же train/validation IDs и channels. Числа этапа03 binary GT-stratified sampling нельзя напрямую сопоставлять с нынешними natural восьмиклассовыми F1.',
           '## 5. Gaussian2, дополнительные Gaussian и Z-context',
           'A–F CE clean сравниваются при одинаковых sampling, architecture, three seeds и protocol. Таблица выше и feature_sets.html показывают отдельно C→D (Gaussian2), C→E (all Gaussian) и C→F (Z). Классовые значения и CM каждого seed: results.json / metrics_by_class.csv. Сильные train correlations |r|>.95:',
           table(['Feature1','Feature2','r'],[[p['left'],p['right'],p['r']] for p in correlation['strong_pairs']]),
           '## 6. Какие классы лучше и какие путаются',
           table(['Модель','BG','LV','RV','LA','RA','MYO','AO','PA'],[[key(s),*[s['per_class_F1_seed_mean'][str(c)] for c in CLASSES]] for s in primary]),
           'Per-class precision/recall/F1, one-vs-rest FPR и BG→class FPR — metrics_by_class.csv; 8×8 pooled и case CM — results.json. Ignore никогда не становится ошибкой background. Основной первый seed E CE combined CM:',
           table(['GT / pred',*LABELS],[[LABELS[c],*principal['validation']['pooled']['confusion_matrix'][c]] for c in CLASSES]),
           '## 7. CT augmentations и устойчивость',
           'Обоснование амплитуд: шум8/12 — около0.2/0.3% source range4095; shift24–40 ≈0.6–1%; scale2–4%, mild gamma; smoothing σ.35/.55; integer shifts один voxel. Эти воздействия мягкие, заранее фиксированы, но не измерены на реальных сканерах. Rotation не добавлялась: для проверки гипотезы достаточно exact translations без интерполяционного размытия targets. Все CT-derived features пересчитаны после каждой операции, independent feature noise не использован.',
           'Train recipes:\n```json\n'+json.dumps(TRAIN_AUG,indent=2)+'\n```\nHeldout validation recipes (другие параметры и seeds):\n```json\n'+json.dumps(VAL_AUG,indent=2)+'\n```',
           'Intensity scale вокруг fixed source1024 (не HU); gamma по clipped source/4095. Translation: source XYZ→XYZ+integer shift, nearest image/target при train, вне поля valid=False. При validation GT не задаёт положение: исходные IDs сохраняются и переносятся известным image transform; baseline для Δ ограничен теми же common-valid IDs. Natural denominator при intensity-only transformations неизменен. Значения noisy ниже — именно noise12, все семь unseen transformations в robustness.json.',
           table(['Model','Clean F1','Noise12 F1','ΔF1','Noise seed SD'],[[key(s),s['seed_mean_F1'],n['seed_mean_F1'],n['seed_mean_F1']-s['seed_mean_F1'],n['seed_SD_F1']] for s in primary for n in [next(r for r in noise if all(r[k]==s[k] for k in ('features','method','augmentation')))]]),
           'Абсолютный noisy F1 и относительное падение — разные свойства. E CE combined имеет более высокий noise12 F1, чем clean-trained CE, но и более высокий clean F1; его падение относительно clean больше. Универсальное повышение устойчивости аугментацией не подтверждено. Независимый feature-space noise этапа03 и нынешний image-space noise имеют разные амплитуды/задачи и не сравниваются причинно напрямую.',
           'Чувствительность классов (noise12 минус clean), seed mean:',
           table(['Model',*LABELS],[[key(s),*[n['per_class_F1_seed_mean'][str(c)]-s['per_class_F1_seed_mean'][str(c)] for c in CLASSES]] for s in primary for n in [next(r for r in noise if all(r[k]==s[k] for k in ('features','method','augmentation')))]]),
           '## 8. SupCon и экспериментальная Spatial Consistency',
           'Сравнивать CE/SupCon/Spatial необходимо при E и одинаковой augmentation. Clean E три losses и E combined CE/SupCon показаны выше. Полезность регуляризации оценивается также по шумовым метрикам и seed SD; отдельный удачный seed или красивый UMAP не считаются доказательством. Collapse проверен по raw и unit embedding variance; все обученные checkpoints прошли проверку.',
           '## 9. Стабильность seed, случаев и group CV',
           'Case CI: 2000 bootstrap resamples cases после усреднения scores между seed. Это описательная неопределённость десяти development случаев, не коррекция за reuse validation/model selection; seed SD показан отдельно (sample SD, ddof=1, n=3). Sampled voxels не считаются независимыми пациентами. Paired ΔF1 относительно k-NN на том же feature set:',
           table(['Inputs','Method','Aug','Mean paired Δ','Case CI95','Case SD'],[[p['features'],p['method'],p['augmentation'],p['mean'],p['case_bootstrap_CI95'],p['case_SD']] for p in paired]),
           'Прямые парные ablations: усреднение трёх seed внутри каждого случая, затем bootstrap только случаев. Это описательные CI без поправки на множественные сравнения и reuse development validation; параметры по ним не менялись.',
           table(['Comparison','Mean paired ΔF1','Case CI95','Case SD'],[[p['comparison'],p['mean'],p['case_bootstrap_CI95'],p['case_SD']] for p in ablation_pairs]),
           '3-fold CV: все voxels/slices release-group остаются вместе. Scaler и weights только на остальных train folds. Fold holdout служит early stopping, поэтому это development CV, не nested unbiased evaluation. Внешние 10 validation cases не входят ни в один CV fit.',
           table(['Fold','Inputs','Method','Aug','Train natural F1','Holdout F1','Cases'],[[m['CV_fold'],m['features'],m['method'],m['augmentation'],m['train_natural']['case_macro']['macro_f1_mean'],m['validation']['case_macro']['macro_f1_mean'],len(m['validation']['per_case'])] for m in result['CV']]),
           '## 10. Dense prediction, фон и ложные островки',
           'Fixed ROI center fraction [.65,.45,.5], size[128,128,96], clipped только по CT shape. Никакой GT-localized crop. Три разрешённых случая: ct_1001 train, ct_1027/ct_1028 development validation. Сравниваются A CE clean, C CE clean, E CE combined first seed, CHD68 v1/v2. Полное ct_1001 дополнительно рассчитано для E CE combined и обеих CHD68. Только inference, без morphology/GT cleanup. Полные маски и 8-class probability/entropy maps сохранены на native grid.',
           'Dense block64×64×16 с exact radius8 halo; float32 features/softmax, float16 disk probability, uint8 final mask из float32 argmax; float16 может создавать округлённые ties, финальная маска authoritative. Нет полного dense feature tensor датасета. CPU-only поддерживается; CHD68 inference использует RTX2060 при наличии, CPU fallback явный. Component counts — по реальной dense prediction, не sparse point graph.',
           table(['Case','Scope','Model','Dense macro F1','Mean Dice present FG','False FG voxels on BG','False FG components no GT overlap'],[[d['case'],d['scope'],d['model'],d['macro_f1'],float(np.mean([d['dense_classwise'][str(c)]['dice'] for c in range(1,8) if d['dense_classwise'][str(c)]['dice'] is not None])),sum(d['confusion_matrix'][0][1:]),sum(d['dense_classwise'][str(c)]['false_components_no_GT_overlap'] for c in range(1,8))] for d in dense]),
           'False component = компонент данного predicted класса без overlap с GT того же класса; включает wrong-structure components и не тождественен только фоновому FP. BG→FG voxel count отдельно. Размеры ВСЕХ компонент: dense/**/metrics.json; background distance bins0–2/2–6/6–20/>20 index units вычислены только после inference по экспертным foreground структурам. Они никогда не являются features или ROI criterion.',
           'Полный ct_1001: Dice / HD95(index) / 6-connected components:',
           table(['Model','Class','Dice','HD95 index','Components','FP voxels','False islands'],[[d['model'],LABELS[c],d['dense_classwise'][str(c)]['dice'],d['dense_classwise'][str(c)]['HD95_index_units'],d['dense_classwise'][str(c)]['components_6connected'],d['dense_classwise'][str(c)]['false_positive_voxels'],d['dense_classwise'][str(c)]['false_components_no_GT_overlap']] for d in dense if d['scope']=='full native volume' for c in CLASSES]),
           '## 11. Сопоставимость CHD68 v1/v2',
           'Обе модели загружены существующим load_recovered с исторической проверкой SHA/config/preprocessing, используются их frozen clipping/context/resize/normalization. Вероятности восстановлены на исходную XY сетку до argmax. Для всех методов одинаковые исходные CT, cardiac_target 0–7/ignore255, source-grid ROI и evaluation functions. Dense Dice измерен независимо от sampled F1; HD95 только в индексных единицах. Поверхности у ignore и crop edges censored; ROI компоненты усечены, поэтому полное ct_1001 приведено отдельно. Это reused train/development сравнение, не независимая оценка.',
           '## 12. Embeddings, importance и интерфейс',
           'Multiclass Feature Lab добавлен в существующий CT Preprocessing Lab; Learnable Feature Explorer также переключается в восьмиклассовый режим. Выбор модели/seed/features/CT, все классы и background toggle, linked point/lasso с Three-plane Viewer, clean/noisy PCA/UMAP, CT perturbation preview, class probabilities, dense prediction/errors/entropy, 3D components и history. Служебные детали скрыты в expanders.',
           'Проекции: фиксированные24/class/case, одинаковые source IDs clean/noisy; scaler/reducer fit только clean TRAIN; noisy train/validation только transform. Train preview noise8, heldout validation noise12. До/после encoder используют независимые bases; clean/noisy в одном representation — common basis. GT только для окраски/diagnostic subsampling, не reducer fit. Количественная geometry в raw embeddings: results.json/robustness.json, не PCA/UMAP coordinates. Importance — within-case permutation first-seed E на фиксированных natural validation1024/case; взаимодействия и корреляции ограничивают причинные выводы.',
           table(['E model','Val raw-space Silhouette seed mean','Val Fisher seed mean','Train unit embedding STD mean'],[[key(s),float(np.mean([m['validation']['geometry']['silhouette'] for m in result['models'] if all(m[k]==s[k] for k in ('features','method','augmentation'))])),float(np.mean([m['validation']['geometry']['fisher_ratio'] for m in result['models'] if all(m[k]==s[k] for k in ('features','method','augmentation'))])),float(np.mean([m['unit_embedding_std_mean'] for m in result['models'] if all(m[k]==s[k] for k in ('features','method','augmentation'))]))] for s in primary]),
           '## 13. Рекомендации и неподтверждённые выводы',
           'Локальный MLP не имеет whole-heart context и shape prior; похожие по интенсивности кровь/сосуды/ткани вне сердца могут порождать FP, даже при хорошем binary representation. Для полноценного сегментатора нужны глобальный multiscale spatial context, background training, строгая ignore handling, image-level augmentation и проверка spatial coherence. CE остаётся обязательным baseline; Gaussian2 и SupCon следует переносить только если их измеренные paired эффекты устойчивы по seed/cases и dense метрикам. Spatial loss пока экспериментален.',
           'Не подтверждены клиническая применимость, устойчивость к реальным scanner/domain shifts, независимое test качество и превосходство одной архитектуры вне рассмотренных cases. Natural samples конечны, rare classes могут отсутствовать; fixed-label macro штрафует такие случаи, present-class metric сохранена отдельно. CV early stopping не nested. Dense full-volume выполнен на одном train case, ROI на двух development cases; этих данных недостаточно для обобщения whole-volume quality на популяцию. Augmentation recipes синтетические и так обозначены. Segmentation network на этом этапе не обучалась.',
           '## 14. Графики и воспроизведение']
    for name,explanation in charts:lines.append(f'- [{name}](outputs/multiclass_feature_04/charts/{name}.html) — {explanation}')
    for case in plan['dense_cases']:lines.append(f'![Dense comparison {case}](outputs/multiclass_feature_04/dense_{case}.png)')
    lines+=['Команды из repository, PowerShell. Завершённые receipts/cache переиспользуются; для полного повторения — отдельная копия без outputs/multiclass_feature_04, прежние эксперименты не удалять.\n```powershell\n.venv/Scripts/python.exe -m ct_lab.multiclass_train\n.venv/Scripts/python.exe -m ct_lab.multiclass_analysis\n.venv/Scripts/python.exe -m ct_lab.multiclass_dense\n.venv/Scripts/python.exe -m ct_lab.multiclass_projection\n.venv/Scripts/python.exe -m ct_lab.multiclass_verify_ui\n.venv/Scripts/python.exe -m pytest tests/test_multiclass_lab.py tests/test_learnable_features.py tests/test_spatial_features.py tests/test_filtering_features.py tests/test_ct_lab.py tests/test_cardiac_inference.py -q\nnode ct_lab/multiclass_browser.cjs\n.venv/Scripts/python.exe -m ct_lab.multiclass_report\n.venv/Scripts/python.exe -m ct_lab.multiclass_launch\n```',
            f'Integrity: 116 исходных CT/mask файлов повторно проверены; {len(prior_hashes())} прежних артефактов byte-for-byte сохранены; frozen split, production preprocessing и CHD68 v1/v2 hashes не изменились. Все42 checkpoints имеют verified SHA/provenance, source IDs проверены, validation training-neighbors отсутствуют. Автотесты: tests.xml; UI: ui_verification.json / browser_verification.json. Производственный cardiac training regression test не запускался.',
            table(['Library','Version'],versions.items())]
    (REPO/'MULTICLASS_FEATURE_EXPERIMENT_04.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    write(OUT/'chart_index.json',charts)
    (OUT/'charts.html').write_text('<meta charset="utf-8"><h1>Multiclass Feature Lab</h1>'+''.join(f'<p><a href="charts/{n}.html">{n}</a>: {e}</p>' for n,e in charts),encoding='utf-8')
    # Keep all outputs on disk; share a compact reproducible package without repeated large prediction arrays.
    archive=OUT/'multiclass_feature_04.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        paths=[p for p in OUT.rglob('*') if p.is_file() and p.suffix!='.zip' and 'numba_cache' not in p.parts
               and p.name not in ('probabilities.npy','entropy.npy','prediction.npy','predictions.npz')
               and not (p.suffix=='.npz' and p.name.startswith('robust_'))]
        paths+=list((REPO/'ct_lab').glob('multiclass*.py'))+list((REPO/'ct_lab').glob('multiclass*.cjs'))
        paths+=[REPO/'MULTICLASS_FEATURE_EXPERIMENT_04.md',REPO/'tests/test_multiclass_lab.py',REPO/'ct_lab/app.py',REPO/'ct_lab/learnable_ui.py',REPO/'ct_lab/MULTICLASS_README.md']
        for p in sorted(set(paths)):z.write(p,str(p.relative_to(REPO)))
        assert len(z.namelist())==len(set(z.namelist()))
    print('MULTICLASS REPORT COMPLETE',len(result['models']),len(result['CV']),len(dense),flush=True)


if __name__=='__main__':run()
