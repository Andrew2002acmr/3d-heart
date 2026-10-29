"""Rebuild Stage06 tables/figures from completed measurements; never train or infer."""
import argparse,importlib.metadata,math
from ct_lab.dynamics_core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

RUN=OUT/'runs/baseline_s20261009_n48'
FIG=OUT/'figures'
TABLE=OUT/'tables'

def read(path,default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default

def md(frame):
    if frame.empty:return 'Измерения ещё не завершены.'
    def cell(v):
        if isinstance(v,(float,np.floating)):return '—' if not np.isfinite(v) else f'{v:.5f}'
        return str(v).replace('|','/')
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)

def csv(frame,name):frame.to_csv(TABLE/(name+'.csv'),index=False,encoding='utf-8-sig')

def collect():
    rows=[];topologies=[]
    for directory in sorted((OUT/'runs').glob('*'))+sorted((OUT/'references').glob('*')):
        paths=sorted((directory/'evaluation').glob('*/summary.json')) if (directory/'evaluation').exists() else [directory/'summary.json']
        for path in paths:
            if not path.exists():continue
            point=path.parent.name if path.parent.parent.name=='evaluation' else 'historical_selected'
            run=directory.name;step=int(point[4:]) if point.startswith('step') else None
            for r in read(path)['records']:
                cm=np.array(r['native_confusion']);row={'run':run,'checkpoint':point,'step':step,'case':r['case'],'role':r['role'],
                  'native_Dice':r['macro_foreground_Dice'],'prepared_Dice':r['prepared_grid']['macro_foreground_Dice'],
                  'macro_class_recall':r['foreground_recall'],'union_foreground_recall':cm[1:,1:].sum()/max(1,cm[1:,:].sum()),
                  'BG_FP_voxels':r['BG_to_foreground_voxels'],'BG_FP_fraction_valid':r['BG_to_foreground_voxels']/max(1,cm.sum())}
                row.update({c+'_Dice':r['classes'][c]['Dice'] if r['classes'][c]['target_voxels'] else None for c in LABELS[1:]});rows.append(row)
                topo=read(path.parent/r['case']/'topology.json')
                if topo:
                    for k,v in topo['topology'].items():
                        t={key:row[key] for key in ['run','checkpoint','step','case','role']};t['class']=LABELS[int(k)]
                        t.update({key:value for key,value in v.items() if key not in ['components','component_sizes_desc']})
                        topologies.append(t)
    return pd.DataFrame(rows),pd.DataFrame(topologies)

def build_figures(frame,topo,history,audit,sampling):
    h=pd.DataFrame(history);local=frame[(frame.run==RUN.name)&frame.step.notna()];dev=local[local.role=='development'];points=dev.groupby('step').mean(numeric_only=True)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for old in audit['historical']:
        oh=pd.DataFrame(old['history']);axes[0,0].plot(oh.epoch*774,oh.training_loss,'--',label=old['variant']+' historical')
        axes[0,1].plot(oh.epoch*774,oh.validation_mean_case_foreground_Dice,'--',label=old['variant']+' prepared')
    if not h.empty:
        axes[0,0].plot(h.step,h.training_loss,label='Stage06');axes[0,1].plot(h.step,h.prepared_development_Dice,label='Stage06 prepared')
        axes[0,2].plot(h.step,h.prepared_gap,'o-',label='prepared train8 − dev10')
        axes[1,0].plot(h.step,h.learning_rate,'-');axes[1,0].set_ylim(0,.0012)
    if not points.empty:
        axes[0,1].plot(points.index,points.native_Dice,'o-',label='Stage06 native controls')
        axes[1,1].plot(points.index,points.union_foreground_recall,'o-',label='union foreground recall')
        axes[1,1].plot(points.index,points.macro_class_recall,'s-',label='macro class recall')
    if not topo.empty:
        t=topo[(topo.run==RUN.name)&(topo.role=='development')&topo.step.notna()]
        if not t.empty:
            values=t.groupby(['step','case']).false_components_no_GT_overlap.sum().groupby('step').mean()
            axes[1,2].plot(values.index,values.values,'o-',label='mean per-case; sum7 classes')
    for ax,title in zip(axes.flat,['Training Loss','Development Dice (grids labelled)','Clean train-monitor / development gap','Learning Rate','Foreground / class recall','False components (6-connectivity)']):
        ax.set_title(title);ax.set_xlabel('Optimizer steps');ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
    save(fig,'training_dynamics')
    fig,ax=plt.subplots(figsize=(9,5))
    for c in LABELS[1:]:
        if not points.empty:ax.plot(points.index,points[c+'_Dice'],'o-',label=c)
    ax.set(xlabel='Optimizer steps',ylabel='Native case-macro Dice',ylim=(0,1),title='Development Dice per class');ax.legend(ncol=7);ax.grid(alpha=.2);save(fig,'dice_by_class')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    names=[r['sampler'] for r in sampling]
    axes[0].bar(names,[r['unique_slices'] for r in sampling]);axes[0].set_title('Unique sampled Z / 7740 steps')
    axes[1].bar(names,[r['foreground_slice_fraction'] for r in sampling]);axes[1].set_title('Foreground slice fraction')
    for r in sampling:
        edges=np.array(r['input_histogram_edges']);values=np.array(r['augmented_input_histogram'],float)
        axes[2].plot((edges[:-1]+edges[1:])/2,values/max(1,values.sum()),label=r['sampler'])
    axes[2].set_title('Actual augmented inputs (128 examples each)');axes[2].legend(fontsize=7)
    for ax in axes[:2]:ax.tick_params(axis='x',rotation=25)
    save(fig,'sampling_exposure')
    fig,ax=plt.subplots(figsize=(10,5))
    for name,group in dev.groupby('checkpoint'):
        group=group.sort_values('case');ax.plot(group.case,group.native_Dice,'o-',label=name)
    for name in ['CHD68_v1','ROI05_Full256']:
        g=frame[(frame.run==name)&(frame.role=='development')].sort_values('case')
        if not g.empty:ax.plot(g.case,g.native_Dice,'--',label=name)
    ax.tick_params(axis='x',rotation=30);ax.set(ylabel='Native foreground Dice',ylim=(0,1),title='Individual reused development cases');ax.legend(fontsize=8);save(fig,'dice_by_case')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    if not topo.empty:
        g=topo[(topo.run==RUN.name)&(topo.role=='development')&topo.step.notna()]
        for c in LABELS[1:]:
            t=g[g['class']==c].groupby('step')
            axes[0].plot(t.components_6connected.mean().index,t.components_6connected.mean(),'-o',label=c)
            axes[1].plot(t.false_component_fraction.mean().index,t.false_component_fraction.mean(),'-o',label=c)
            axes[2].plot(t.HD95_index_units.mean().index,t.HD95_index_units.mean(),'-o',label=c)
    for ax,title in zip(axes,['6-connected components','False island volume / predicted volume','HD95 (index units; missing excluded)']):
        ax.set_title(title);ax.set_xlabel('Optimizer steps');ax.grid(alpha=.2)
    axes[0].legend(ncol=4,fontsize=7);save(fig,'topology_dynamics')

def generalization(frame,stats):
    selected=frame[(frame.run==RUN.name)&(frame.checkpoint=='selected')&(frame.role=='development')]
    if selected.empty:
        controls=frame[(frame.run==RUN.name)&frame.step.notna()&(frame.role=='development')]
        selected=controls[controls.step==controls.step.max()]
    s=pd.DataFrame([{k:v for k,v in r.items() if k not in ['prepared_slice_class_counts']} for r in stats])
    joined=selected.merge(s,on='case',suffixes=('','_data'));joined['inference_ROI_FOV_fraction']=1.;features=['slice_count','foreground_fraction','source_sample_mean','source_sample_std','GT_bbox_FOV']
    train=s[s.role=='train'];correlations=[]
    if not joined.empty:
        joined['shape_X']=joined.shape_xyz.map(lambda x:x[0]);joined['shape_Y']=joined.shape_xyz.map(lambda x:x[1])
        for axis in range(3):joined['heart_center_'+str(axis)]=joined.normalized_center_mass.map(lambda x:x[axis])
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for ax,f in zip(axes.flat,features):
            mean=train[f].mean();sd=train[f].std(ddof=0);joined[f+'_train_z']=(joined[f]-mean)/max(sd,1e-12)
            ax.scatter(joined[f+'_train_z'],joined.native_Dice)
            for _,r in joined.iterrows():ax.annotate(r['case'],(r[f+'_train_z'],r.native_Dice),fontsize=7)
            ax.set(title=f,xlabel='Train-normalized deviation (diagnostic)',ylabel='Native Dice')
            correlations.append({'feature':f,'Pearson':joined[f].corr(joined.native_Dice),'Spearman':joined[f].corr(joined.native_Dice,method='spearman'),'n_cases':len(joined),'checkpoint':joined.checkpoint.iloc[0]})
        axes.flat[-1].axis('off');save(fig,'generalization')
    # Include class volumes and centers in a machine-readable diagnostic table.
    for c in range(1,8):
        if not joined.empty:joined[LABELS[c]+'_GT_voxels']=joined.class_source_voxels.map(lambda x:x[c])
    csv(joined.drop(columns=[c for c in ['shape_xyz','class_source_voxels','prepared_class_counts','GT_bbox','normalized_center_mass','source_sample_percentiles'] if c in joined]),'generalization_cases')
    corr=pd.DataFrame(correlations);csv(corr,'generalization_correlations');return joined,corr

def case_bootstrap(frame):
    ref=frame[(frame.run=='CHD68_v1')&(frame.role=='development')][['case','native_Dice']].rename(columns={'native_Dice':'historical'})
    rows=[]
    for point,g in frame[(frame.run==RUN.name)&(frame.role=='development')].groupby('checkpoint'):
        paired=g.merge(ref,on='case');delta=(paired.native_Dice-paired.historical).to_numpy();rng=np.random.default_rng(20261029)
        if not len(delta):continue
        means=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
        rows.append({'checkpoint':point,'n_cases':len(delta),'mean_paired_Dice_delta':delta.mean(),'case_SD_delta':delta.std(ddof=1),'descriptive_bootstrap_low':np.quantile(means,.025),'descriptive_bootstrap_high':np.quantile(means,.975),'better_cases':int((delta>0).sum())})
    return pd.DataFrame(rows)

def source_examples():
    import nibabel as nib
    from matplotlib.colors import ListedColormap
    palette=ListedColormap(['#000000','#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#ff69b4'])
    for case in ['ct_1001','ct_1027']:
        allowed(case);image=nib.load(DATA/f'{case}_image.nii.gz');z=image.shape[2]//2
        raw=np.asarray(image.dataobj[:,:,z],np.float32).T;truth=cardiac_target(np.asarray(nib.load(DATA/f'{case}_label.nii.gz').dataobj[:,:,z])).T
        candidates=[('GT',None)]+[(n,RUN/'evaluation'/n/case/'prediction.nii.gz') for n in ['step000480','step003840','step007740','step015480','step023220','selected']]
        candidates += [('CHD68 v1',OUT/'references/CHD68_v1'/case/'prediction.nii.gz')]
        candidates=[(n,p) for n,p in candidates if p is None or p.exists()]
        fig,axes=plt.subplots(2,len(candidates),figsize=(3*len(candidates),6),squeeze=False)
        for i,(name,path) in enumerate(candidates):
            pred=truth if path is None else np.asarray(nib.load(path).dataobj[:,:,z]).T
            axes[0,i].imshow(raw,cmap='gray',vmin=0,vmax=2015);axes[0,i].imshow(np.ma.masked_where((pred==0)|(pred>7),pred),cmap=palette,vmin=0,vmax=7,alpha=.55)
            error=np.zeros_like(pred);valid=truth<8;error[valid&(truth==0)&(pred>0)]=1;error[valid&(truth>0)&(pred==0)]=2;error[valid&(truth>0)&(pred>0)&(pred!=truth)]=3
            axes[1,i].imshow(raw,cmap='gray',vmin=0,vmax=2015);axes[1,i].imshow(np.ma.masked_where(error==0,error),cmap=ListedColormap(['black','red','cyan','yellow']),vmin=0,vmax=3,alpha=.8)
            axes[0,i].set_title(name);axes[1,i].set_title('FP red / FN cyan / class yellow',fontsize=8)
        for ax in axes.flat:ax.axis('off')
        fig.suptitle(f'{case} · native Z={z} · '+('train example' if case=='ct_1001' else 'first frozen development case'))
        save(fig,'spatial_errors_'+case)

def build(final=False):
    FIG.mkdir(parents=True,exist_ok=True);TABLE.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    result=read(RUN/'result.json');history=read(RUN/'history.json',[]);audit=read(OUT/'training_audit.json');sampling=read(OUT/'sampling_analysis.json',[])
    if final and (not result or not read(OUT/'topology_completion.json')):raise ValueError('Full baseline/topology still running; cannot label report final')
    frame,topo=collect();csv(frame,'case_checkpoint_metrics');csv(topo,'class_topology_metrics')
    columns=['native_Dice','prepared_Dice','macro_class_recall','union_foreground_recall','BG_FP_voxels']
    summary=frame.groupby(['run','checkpoint','role'])[columns].agg(['mean','std','count']);summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index();csv(summary,'aggregate_metrics')
    csv(pd.DataFrame(audit['differences'],columns=['Фактор','CHD68 v1/v2','ROI05 Full FOV']),'pipeline_comparison')
    csv(pd.DataFrame([{k:r[k] for k in ['sampler','steps','examples','unique_cases','unique_slices','foreground_slice_fraction','background_slice_fraction','multiple_structure_fraction','repeat_mean','repeat_max']} for r in sampling]),'sampling_exposure')
    bootstrap=case_bootstrap(frame);csv(bootstrap,'paired_case_bootstrap')
    build_figures(frame,topo,history,audit,sampling);joined,corr=generalization(frame,read(OUT/'case_statistics.json'));source_examples()
    reference=read(OUT/'reference_comparison.json',[]);metric_table=pd.DataFrame([{'Model':r['method'],'Frozen native Dice':r['frozen_original_native_Dice'],'Common FP32 native':r['common_FP32']['native_Dice'],'Common prepared':r['common_FP32']['prepared_Dice'],'Delta native':r['common_minus_frozen']} for r in reference])
    main=summary[(summary.run==RUN.name)&(summary.role=='development')][['checkpoint','native_Dice_mean','native_Dice_std','prepared_Dice_mean','union_foreground_recall_mean']]
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','nibabel','pandas','matplotlib','scikit-learn','plotly','streamlit']};write(OUT/'report_versions.json',versions)
    # Actual checkpoints/seed coverage, including subsequently completed manual jobs.
    actual=[]
    for path in sorted((OUT/'runs').glob('*/result.json')):
        r=read(path);cfg=read(path.parent/'config.json');job=cfg.get('stage06_job',read(path.parent/'run_manifest.json',{}))
        actual.append({'run':path.parent.name,'family':job.get('family','baseline'),'n_train':r['n_train'],'seed':r['seed'],'steps':r['steps'],'native_Dice':r['evaluation']['by_role']['development']['native_Dice']})
    csv(pd.DataFrame(actual),'completed_runs')
    selected=frame[(frame.run==RUN.name)&(frame.checkpoint=='selected')&(frame.role=='development')]
    native=float(selected.native_Dice.mean()) if not selected.empty else None
    status='Завершён один локальный baseline и контрольные 3D-диагностики.' if final else 'Промежуточный отчёт: полный локальный baseline ещё выполняется; итог воспроизведения не установлен.'
    reproduction=(f'Новый selected native Dice = **{native:.6f}**, разность с общим FP32 CHD68 v1 = **{native-.7978329771435937:+.6f}**. '+('Инженерный допуск ±0.03 пройден.' if abs(native-.7978329771435937)<=.03 else 'Инженерный допуск ±0.03 не пройден; следующие обучения заблокированы до разбора причин.')) if native is not None else 'Итоговое качество пока не измерено.'
    temporal=frame[(frame.run==RUN.name)&frame.step.notna()&(frame.role=='development')].groupby('step').native_Dice.mean()
    growth='Контроль3840 ещё не завершён.'
    if 3840 in temporal.index:
        last_step=int(temporal.index.max());growth=f'На3840 шагах native Dice {temporal.loc[3840]:.6f}; на последней измеренной контрольной точке {last_step} — {temporal.loc[last_step]:.6f} (Δ {temporal.loc[last_step]-temporal.loc[3840]:+.6f}). Это один training trajectory, не изоляция всех отличий Stage05.'
    worst=joined.sort_values('native_Dice')[['case','checkpoint','native_Dice','foreground_fraction','slice_count','source_sample_mean']] if not joined.empty else pd.DataFrame()
    bestclass='Итоговая class-wise таблица появится после selected evaluation.'
    if not selected.empty:
        cc=selected[[c+'_Dice' for c in LABELS[1:]]].mean().sort_values();bestclass='Слабее всего на этой development-выборке: '+', '.join(f'{c[:-5]} {v:.4f}' for c,v in cc.iloc[:3].items())+'. Это диагностирует ошибки, но не доказывает потребность именно в дополнительных данных.'
    report=f'''# Training Dynamics & Generalization — эксперимент06

{status}

По решению пользователя локальный бюджет ограничен одним полным историческим baseline. Дополнительные seed, sampling-модели, loss-абляции и размеры train подготовлены для **ручного RunPod**, их результаты не подменяются simulation. Sealed test закрыт. Все числа ниже относятся к многократно использованной development validation, а не независимой финальной оценке.

## 1. Что восстановлено и чем отличается Stage05

{md(pd.DataFrame(audit['differences'],columns=['Фактор','CHD68 v1/v2','ROI05 Full FOV']))}

Источники: frozen configs, model/augmentation/loss/train/data source, сохранённые optimizer states и history best-checkpoints, training-only поля архивных metadata. HeartUNet25D:5 входов,8 выходов, encoder16/32/64/128, GroupNorm8,489112 параметров. Adam lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=0; scheduler/early stopping отсутствуют. CE + batchwise foreground soft Dice без class weights. Raw intensity clip0..2015 был fitted только на train; нормализация[-1,1], fit/padXY, Z без resampling, offsets−2…+2, края clamp. Это не подтверждённые HU и не физические миллиметры.

Исторические full reports/configs показывают30 эпох.12381 examples/epoch, batch16 и последний batch13 дают774 steps/epoch,23220 steps и371430 examples за полный бюджет. Доступный v1 best содержит23 эпохи и Adam step17802; v2 best —10 эпох и step7740. **Хвосты history после best не доступны**, полный23220-budget выведен из отчёта о завершённых30 эпохах, а не выдан за непосредственно найденный optimizer state. Historical v1/v2 model/loss/augmentation/train/runtime source проверены по Git snapshots; v1 data source отличается добавленным guard sealed partitions, v2 совпадает.

ROI05:3840 updates /7680 examples против23220 /371430 — примерно6.05× updates и48.36× examples. Pool~707 уникальных Z против12381; AdamW,AMP/F16cache, другие аугментации и fixed-last selection. Различия потенциально влияют одновременно; baseline trajectory не позволяет приписать весь разрыв одному фактору. CE+Dice при batch2 и batch16 также задаёт различающийся batch-Dice objective.

## 2. Сопоставимость метрик

{md(metric_table)}

Общий FP32 adapter: классы0–7, GT255 исключён; исходная NIfTI grid/affine; inverse probabilities **до** argmax, никакого удаления компонент. Dice усредняется по reference-present foreground classes внутри case, затем равновесно по10 cases; empty-empty не даёт бонус1. False predictions отсутствующего класса не добавляют класс в macro, поэтому отдельно сохранены FP объёмы, recall, confusion и топология. Prepared-grid и native-grid всегда подписаны раздельно. Ни mixed11-case среднее, ни ct_1001 train не включаются в dev10.

Frozen Stage05 native prediction masks пересчитаны тем же helper без изменения исходных файлов: определение Dice совпало точно. Различие общего FP32 для ROI05 ~−0.00000213; для v1/v2 <0.00000005. Это не объясняет разрыв~0.155. Selection различается: historical максимум prepared Dice по эпохам, ROI05 fixed last; даже единый adapter не уравнивает это преимущество отбора.

## 3. Воспроизведение baseline и продолжительность

{reproduction}

{md(main)}

{growth}

Заранее заданные native controls:480,3840,7740,15480,23220 и selected по максимуму epoch-end prepared development Dice.480×16=7680 examples совпадает с Stage05 числом примеров;3840 совпадает по числу updates, но не batch size/examples. Полные actual counters/unique Z записаны в history/steps/checkpoints. Train-monitor — заранее фиксированные8 train cases на clean prepared grid; это не training minibatch score и не полный train48. Ct_1001 сохранён отдельно как train-example.

![Loss, Dice, gap, LR, recall, false components](outputs/training_dynamics_06/figures/training_dynamics.png)

Кривые показывают измеренные значения. Historical пунктир заканчивается доступным best-history; отсутствующие эпохи не дорисованы. Prepared/native значения представлены отдельно; рост union foreground recall не гарантирует правильный анатомический класс.

![Class-wise native Dice](outputs/training_dynamics_06/figures/dice_by_class.png)

![Cases](outputs/training_dynamics_06/figures/dice_by_case.png)

Разброс по cases описывает10 release-cases; это не SD между seed. Paired case differences и описательный10000-resample case-bootstrap сохранены в tables/paired_case_bootstrap.csv. Интервалы не устраняют selection bias reused development и не являются независимой статистической проверкой.

{md(bootstrap)}

## 4. Sampling: что измерено

{md(pd.read_csv(TABLE/'sampling_exposure.csv'))}

Все4 train-only simulation используют7740 updates,123810 examples, одинаковые batch/augmentation и seed. Отличается только sampler. B воспроизводит **pool/case allocation ROI05**, сохраняя историческую augmentation ради изоляции sampling; это не полная копия Stage05 training. C уравнивает cases; D относительно C вдвое уменьшает вероятность eligible negative slices внутри case. Sampled class counts относятся к clean targets, input histograms —128 фактически augmented examples каждого варианта. Ниже не результаты четырёх обученных моделей.

![Sampling](outputs/training_dynamics_06/figures/sampling_exposure.png)

Подтверждена существенно более узкая экспозиция ROI05 pool. Какой sampling даёт лучший Dice, **пока не установлено**: fixed-budget training абляции отложены на RunPod.

## 5. Пространственные ошибки и генерализация

![Topology](outputs/training_dynamics_06/figures/topology_dynamics.png)

Raw3D connected components, connectivity6, по каждому классу; false island — компонент без пересечения с GT того же класса. Это не все FP: крупный ошибочный компонент с малым GT overlap не считается false island, поэтому отдельно учитываются class FP voxels и BG→foreground. HD95 — symmetric pooled surface-distance percentile в единицах индексной сетки, без мм; поверхности у ignore/границы источника censored. При отсутствующей поверхности HD95=null, не0; CSV содержит случаи/классы для проверки доступности. Основное сравнение без component cleanup. Изменение числа островков следует читать вместе с Dice/recall: исчезнувший класс не является улучшением топологии.

![Train case](outputs/training_dynamics_06/figures/spatial_errors_ct_1001.png)

![First development case](outputs/training_dynamics_06/figures/spatial_errors_ct_1027.png)

Показаны заранее фиксированные ct_1001 (train) и первый frozen development ct_1027, центральный индексный Z; это не отбор лучших картинок. Верх — GT/prediction поверх CT, низ — BG FP красный, foreground FN голубой, путаница foreground-классов жёлтая. Все predictions соответствуют одному source CT/affine; полный набор3D масок и метрик сохранён отдельно.

{md(worst)}

![Generalization](outputs/training_dynamics_06/figures/generalization.png)

Размеры, число Z, raw intensity statistics, foreground fraction, GT bbox-FOV, class volumes и нормированный центр сердца сохранены в generalization_cases.csv. GT bbox характеризует анатомический охват; фактический inference ROI здесь всегда full FOV. Диагностические z-scores fitted по train48; они не подаются модели и не меняют preprocessing. Корреляции по10 cases описательные и не устанавливают причинность.

{md(corr)}

## 6. Отложенная матрица и воспроизводимость

33 заранее заданных jobs /301860 optimizer updates:3 полных baseline seeds;9 sampling;12 subset (N8/16/24/32);9 loss (CE, weighted CE+Dice, focal+Dice). Reference N48/historical/CE+Dice на7740 берётся из prefix полного baseline без повторного обучения. Все nonbaseline budgets7740, fixed-final checkpoint. Subsets вложены, seed состава20261029, только frozen train/release groups; initialization seeds20261009/10/11. Это **не** три seed состава данных. Weighted CE веса только по unique prepared train pixels выбранного subset: inverse sqrt frequency,unit mean,clip[.25,4]; focal gamma2,без alpha. Для subset experiments preprocessing clip сохраняется исторически fitted на48 train, поэтому это изоляция числа размеченных training cases при фиксированном preprocessing, не обучение end-to-end из N cases.

При одном N48/seed нельзя построить достоверную пяти-точечную learning curve, seed SD или выбрать лучший loss. UI показывает только реальные точки и явно помечает отсутствующие. Matrix/manual plan и runner подготовлены; никакая платная инфраструктура не создана. Engineering fidelity gate ±.03 native Dice относительно historicalv1, finite/decreasingloss и полный budget проверяются до downstream jobs; при провале runner останавливается, параметры автоматически не подбираются.

Команды/перенос/RunPod: [ct_lab/TRAINING_DYNAMICS_README.md](ct_lab/TRAINING_DYNAMICS_README.md). Локальный smoke: CPU64/B2 и GPU256/B16 FP32; GPUpeak1.91GiB, ~.752s/update (10 timed updates после5 warmup), ~4.85h training-only оценка; epoch validation/native/topology добавляют время. Это оценка RTX2060, не обещание времени4090. Новые immutable step/selected weights лежат отдельно; last.pt — mutable resume state, не UI checkpoint. Resume восстанавливает optimizer/counters/order, tiny CPU comparison доказал совпадение uninterrupted/resume; CUDA warn-only не обещает bitwise на другой платформе.

Версии локальной среды: {json.dumps(read(OUT/'environment.json'),ensure_ascii=False)}. Дополнительные библиотеки: {json.dumps(versions,ensure_ascii=False)}. Исторически4090, Python3.12.3/Torch2.8+cu128, CPUthreads6; локально RTX2060/threads1. Поэтому это близкое воспроизведение pipeline, не утверждение о bitwise повторе старых весов.

## 7. Ответы на12 исследовательских вопросов

1. **Отличия v1 и ROI05:** budget, batch-Dice, sampler/pool, optimizer/WD, precision/cache, augmentation и checkpoint selection (таблица выше); архитектура Full256 та же.
2. **Что могло влиять:** все эти различия правдоподобны; metric-adapter различие измерено и слишком мало, чтобы объяснить разрыв. Раздельные причинные эффекты пока не установлены.
3. **Воспроизведён ли baseline:** {reproduction} Один seed не доказывает устойчивость.
4. **Помогает ли длительность:** {growth} Продолжение того же trajectory контролирует остальные параметры внутри нового baseline; оно не изолирует Adam/AMP/sampling относительно Stage05.
5. **Лучший sampling:** качество альтернатив не измерено; подтверждены только exposure differences, обучаемые варианты готовы вручную.
6. **Лучший loss:** пока неизвестен; CE/weighted/focal заранее заданы и unit-tested, local ablations не запускались.
7. **Фрагментация во времени:** фактические class/case trajectories на рисунке и CSV; raw components без cleanup, HD95 missing не превращён в0. Островки и крупные ошибки необходимо анализировать совместно.
8. **Устойчивость seed:** локально один seed; seed SD не оценён. Три seeds предусмотрены, SDcases не заменяет SDseeds.
9. **Объём train:** N48 измеряется локально, остальные N отложены; тренд по объёму данных пока не подтверждён.
10. **Классы и дополнительные данные:** {bestclass}
11. **Недостаточность архитектуры:** оснований приписывать разрыв архитектуре нет; одинаковая архитектура обучалась по существенно разным pipelines. Capacity/bias ограничение не исключено, но не установлено.
12. **Следующий CHD68 v3:** сначала выполнить matched sampler/seed/loss/subset jobs и проверить обобщение; фиксировать preprocessing/metric/selection/budget; новую архитектуру проверять отдельным контролируемым экспериментом после стабильного baseline. Не менять production веса по reused-dev результатам.

## 8. Проверки и ограничения

Новые tests: historical evidence/budget, nested split и sealed guard, точный prepared plane, inverse-probability mapping/GT-free inputs, ignore/absent-class rules, atomic CPU resume, deterministic loader/samplers, train-only loss weights и игнорируемые gradients. Streamlit AppTest проверяет7 panels, Z/CT при смене immutable checkpoints, overlay/classes, source SHA mismatch и sealed gate **до** rawI/O. Real-browser verification сохраняется отдельно после480/3840 controls. SHA snapshots исходных116 train/dev NIfTI, v1/v2, frozen configs/split и всех артефактов01–05 проверяются отдельно в final_integrity.json; наличие фразы здесь не заменяет этот файл.

Данные — public release-cases; независимость пациентов и физические units не подтверждены. Development использовалась прежде и при выборе best, поэтому не независима. Seed composition только один; новые локальные seed SD/data/loss/sampling quality не доступны по решению о бюджете. Historical tails недоступны. Тяжёлые3D метрики только заранее объявленные checkpoints. Никаких новых фильтров/архитектур, GT inference-localization или sealed-test подбора. Старые отчёты и артефакты не пересчитываются и не перезаписываются.
'''
    destination=REPO/'TRAINING_DYNAMICS_GENERALIZATION_06.md' if final else OUT/'TRAINING_DYNAMICS_GENERALIZATION_06_IN_PROGRESS.md'
    destination.write_text(report,encoding='utf-8');write(OUT/'report_status.json',{'final':final,'baseline_complete':bool(result),'completed_runs':len(actual),'case_metric_rows':len(frame),'class_topology_rows':len(topo),'report':str(destination)})
    print('DYNAMICS REPORT',destination,'rows',len(frame),len(topo),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args();build(args.final)
