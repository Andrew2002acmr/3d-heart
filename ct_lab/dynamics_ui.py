"""Read-only training/checkpoint exploration with real measurements and pending-run labels."""
import pandas as pd
import nibabel as nib
import plotly.graph_objects as go
import streamlit as st
from ct_lab.dynamics_core import *
from ct_lab.dynamics_evaluate import load_checkpoint,predict_batch_native
from heart3d.ml.cardiac_data import index_transform,resize_plane,confusion_counts,scores_from_confusion
from ct_lab.roi_ui import png
os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache')

def read(path,default=None):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default

@st.cache_resource(max_entries=3)
def cached_model(path,sha):
    model,saved,pre=load_checkpoint(Path(path),'cpu');return model,saved,pre

@st.cache_data(max_entries=120)
def file_sha(path,mtime,size):return fingerprint(Path(path))

def native_directory(path,case):
    path=Path(path)
    if path.parent.name.startswith(('baseline_','sampling_','data_','loss_')):return path.parent/'evaluation'/path.stem/case
    return OUT/'references'/('CHD68_v1' if 'v1' in path.name else 'CHD68_v2')/case

@st.cache_data(max_entries=32)
def checkpoint_plane(path,sha,case,z,cache_version=None):
    source=nib.load(DATA/f'{case}_image.nii.gz')
    directory=native_directory(path,case);meta=read(directory/'metrics.json')
    if meta and meta['checkpoint_SHA256']==sha and (directory/'prediction.nii.gz').exists():
        image=nib.load(directory/'prediction.nii.gz');assert image.shape==source.shape and np.array_equal(image.affine,source.affine)
        return np.asarray(image.dataobj[:,:,z],np.uint8),'saved native full-volume prediction'
    model,saved,pre=cached_model(path,sha);t=index_transform(source.shape,pre['input_size'])
    low,high=pre['clip_source_intensity'];contexts=[]
    if pre['input_size']==256 and (OUT/'prepared256'/case/'image.npy').exists():
        image=np.load(OUT/'prepared256'/case/'image.npy',mmap_mode='r');indices=np.clip(z+np.array(pre['context_offsets_indices']),0,len(image)-1)
        _,native=predict_batch_native(model,np.array(image[indices],np.float32)[None],t)
        return native[0].T,'CPU single-slice inference; whole-volume metrics separate'
    for delta in pre['context_offsets_indices']:
        index=int(np.clip(z+delta,0,source.shape[2]-1));raw=np.asarray(source.dataobj[:,:,index],np.float32)
        contexts.append(np.clip((resize_plane(raw,t,1,0)-low)/(high-low),0,1)*2-1)
    _,native=predict_batch_native(model,np.stack(contexts)[None].astype(np.float32),t)
    return native[0].T,'CPU single-slice inference; whole-volume metrics separate'

def render(volume,mask,affine,case,image_path,mask_path):
    st.subheader('Training Dynamics Lab')
    st.caption('48 train / 10 reused development / sealed test закрыт. HU и физические единицы не подтверждены; HD95 в grid units, не мм.')
    if not (OUT/'protocol.json').exists():st.info('Протокол этапа6 ещё не подготовлен.');return
    split,records=cohort();allowed(case)
    for path,field in [(image_path,'image_SHA256'),(mask_path,'mask_SHA256')]:
        p=Path(path)
        if not p.exists() or file_sha(str(p),p.stat().st_mtime_ns,p.stat().st_size)!=records[case][field]:st.error('CT/GT не совпадают с frozen case; использовать сохранённые checkpoint predictions нельзя.');return
    target=cardiac_target(mask);st.caption(case+' · '+('train diagnostic' if case in split['train'] else 'reused development, не независимый тест'))
    audit=read(OUT/'training_audit.json');directories=sorted((OUT/'runs').glob('*/config.json'));runs={p.parent.name:p.parent for p in directories}
    if not runs:st.info('Подготовка baseline; интерфейс не запускает обучение.');return
    run=st.selectbox('Dynamics run',list(runs));directory=runs[run];cfg=read(directory/'config.json');history=read(directory/'history.json',[]);live=read(directory/'live.json',{})
    st.write(f'Optimizer steps: {live.get("step",history[-1]["step"] if history else 0)} / {cfg.get("stage06_job",{}).get("max_steps",23220)} · processed examples: {live.get("processed_examples",0)}')
    st.caption('Локальный бюджет: один полный baseline. Матрица33 запусков подготовлена для ручного RunPod; отсутствующие измерения не заполняются вымышленными значениями.')
    tabs=st.tabs(['Training Comparison','Learning Curves','Checkpoint Explorer','Sampling Analysis','Data Volume Analysis','Generalization','Topology'])
    with tabs[0]:
        st.dataframe(pd.DataFrame(audit['differences'],columns=['Фактор','CHD68 v1/v2','ROI05 Full FOV']).astype(str),hide_index=True)
        st.caption('v1 best checkpoint содержит23 эпохи истории, v2 —10; полные30-эпоховые хвосты не восстановлены по предположениям. Batch-Dice зависит от размера батча.')
    with tabs[1]:
        choice=st.selectbox('Dynamics curve',['Loss','Prepared Dice','Native Dice','Gap','Learning Rate','Foreground Recall'])
        fig=go.Figure()
        if choice in ('Loss','Prepared Dice'):
            for old in audit['historical']:
                fig.add_trace(go.Scatter(x=[h['epoch']*old['optimizer_steps_per_epoch'] for h in old['history']],y=[h['training_loss'] if choice=='Loss' else h['validation_mean_case_foreground_Dice'] for h in old['history']],name=old['variant']+' historical prepared',line={'dash':'dash'}))
        fields={'Loss':'training_loss','Prepared Dice':'prepared_development_Dice','Gap':'prepared_gap','Learning Rate':'learning_rate'}
        if choice in fields and history:fig.add_trace(go.Scatter(x=[r['step'] for r in history],y=[r[fields[choice]] for r in history],mode='lines+markers',name='new '+choice))
        evaluations=[]
        for path in (directory/'evaluation').glob('step*/summary.json'):
            value=read(path);step=int(path.parent.name[4:]);evaluations.append((step,value))
        for step,value in sorted(evaluations):
            rows=[r for r in value['records'] if r['role']=='development']
            if choice in ('Native Dice','Foreground Recall'):
                values=[r['macro_foreground_Dice'] if choice=='Native Dice' else float(np.array(r['native_confusion'])[1:,1:].sum()/max(1,np.array(r['native_confusion'])[1:,:].sum())) for r in rows]
                fig.add_trace(go.Scatter(x=[step],y=[float(np.mean(values))],mode='markers',name=f'native {step}'))
        st.plotly_chart(fig.update_layout(title=choice+' vs optimizer steps',xaxis_title='Optimizer steps',height=340),key='dynamics_curve')
        st.caption('Prepared и native показатели различаются и не смешиваются. Gap — clean prepared train-monitor8 минус prepared development10; не augmented minibatch Dice.')
        st.caption('Foreground Recall здесь — union foreground detection: GT foreground → любой foreground. Macro class recall отдельно учитывает правильный анатомический класс.')
    with tabs[2]:
        points={p.stem:p for p in sorted(directory.glob('step*.pt'))}
        if (directory/'selected.pt').exists():points['selected checkpoint']=directory/'selected.pt'
        points['CHD68 v1 historical']=WEIGHTS/'chd68_v1_best.pt';points['CHD68 v2 historical']=WEIGHTS/'chd68_v2_best.pt'
        selected=st.selectbox('Dynamics checkpoint',list(points),key=f'dynamics_checkpoint_{case}');path=points[selected];chosen_path=path;sha=fingerprint(path)
        z=st.slider('Dynamics native Z',0,volume.shape[2]-1,int(volume.shape[2]//2),key=f'dynamics_z_{case}')
        opacity=st.slider('Dynamics mask opacity',0.,1.,.45);classes=st.multiselect('Dynamics visible classes',list(range(1,8)),default=list(range(1,8)),format_func=lambda c:LABELS[c])
        cached=native_directory(path,case)/'metrics.json';version=cached.stat().st_mtime_ns if cached.exists() else None
        pred,prediction_scope=checkpoint_plane(str(path),sha,case,z,version);truth=target[:,:,z];valid=truth<8;error=np.zeros_like(pred)
        error[valid&(truth==0)&(pred>0)]=1;error[valid&(truth>0)&(pred==0)]=2;error[valid&(truth>0)&(pred>0)&(truth!=pred)]=3
        for col,(title,seg,palette,err) in zip(st.columns(3),[('GT',truth,classes,False),(selected,pred,classes,False),('FP / FN / class confusion',error,[1,2,3],True)]):
            col.image(png(volume[:,:,z].T,seg.T,palette,opacity,err),caption=f'{title} · native Z={z}')
        slice_scores=scores_from_confusion(confusion_counts(pred,truth));st.plotly_chart(go.Figure(go.Bar(x=LABELS[1:],y=[slice_scores['classes'][c]['Dice'] for c in LABELS[1:]])).update_layout(title='Dice только выбранного2D-среза, не всего объёма',height=250),key='dynamics_slice_class')
        st.caption('Один и тот же CT/Z сохраняется при переключении checkpoint. '+prediction_scope+'. Веса не изменяются; полнообъёмные метрики считаются отдельным pipeline.')
        st.write({'source_case':case,'source_Z':z,'checkpoint':selected,'checkpoint_SHA256':sha,'mask_classes_present':np.unique(pred).tolist()})
    with tabs[3]:
        rows=read(OUT/'sampling_analysis.json',[])
        if rows:
            st.dataframe(pd.DataFrame([{k:r[k] for k in ['sampler','steps','examples','unique_cases','unique_slices','foreground_slice_fraction','multiple_structure_fraction','repeat_mean']} for r in rows]),hide_index=True)
            fig=go.Figure()
            for r in rows:fig.add_trace(go.Scatter(x=(np.array(r['input_histogram_edges'][:-1])+np.array(r['input_histogram_edges'][1:]))/2,y=r['augmented_input_histogram'],mode='lines',name=r['sampler']))
            st.plotly_chart(fig.update_layout(title='Измеренные augmented train input histograms',height=300),key='dynamics_sampling')
            fig=go.Figure()
            for r in rows:
                counts=np.array(r['prepared_clean_class_voxel_counts']);fig.add_trace(go.Bar(x=LABELS,y=counts/counts.sum(),name=r['sampler']))
            st.plotly_chart(fig.update_layout(title='Классы sampled clean targets',yaxis_type='log',height=300),key='dynamics_sampling_classes')
        st.caption('Это simulation exposure и реальные augmented input samples, не результаты обучения четырёх моделей. Все sampling ablations для оценки Dice отложены на ручной RunPod.')
    with tabs[4]:
        rows=[]
        for config in directories:
            job=config.parent;settings=read(config);jobinfo=settings.get('stage06_job') or read(job/'run_manifest.json',{'family':'baseline','sampling':'historical','loss':'ce_dice'})
            if jobinfo.get('family') not in ('baseline','data') or jobinfo.get('sampling')!='historical' or jobinfo.get('loss')!='ce_dice':continue
            n=jobinfo.get('n_train',48)
            summary=read(job/'evaluation/step007740/summary.json')
            if summary:
                entry={'N':n,'seed':settings['seed'],'Dice':summary['by_role']['development']['native_Dice'],
                  'Train_monitor_Dice':summary['by_role'].get('train_monitor',{}).get('native_Dice')}
                entry['Gap']=entry['Train_monitor_Dice']-entry['Dice'] if entry['Train_monitor_Dice'] is not None else None
                development=[r for r in summary['records'] if r['role']=='development']
                entry.update({c:float(np.mean([r['classes'][c]['Dice'] for r in development if r['classes'][c]['target_voxels']])) for c in LABELS[1:]})
                rows.append(entry)
        volume_curve=st.selectbox('Data volume curve',['Development native Dice','Train-monitor native Dice','Gap','Class-wise native Dice','SD across initialization seeds'])
        fig=go.Figure()
        if rows:
            frame=pd.DataFrame(rows);group=frame.groupby('N')['Dice'].agg(['mean','std','count']).reset_index()
            columns=LABELS[1:] if volume_curve=='Class-wise native Dice' else [{'Development native Dice':'Dice','Train-monitor native Dice':'Train_monitor_Dice','Gap':'Gap','SD across initialization seeds':'Dice'}[volume_curve]]
            for column in columns:
                values=frame.groupby('N')[column].agg(['mean','std','count']).reset_index()
                y=values['std'].where(values['count']>=2) if volume_curve=='SD across initialization seeds' else values['mean']
                fig.add_trace(go.Scatter(x=values['N'],y=y,mode='markers' if len(values)==1 else 'lines+markers',name=column))
            st.dataframe(group,hide_index=True)
        st.plotly_chart(fig.update_layout(title=volume_curve+' · только реальные7740-step runs',xaxis={'tickvals':[8,16,24,32,48],'range':[0,52]},xaxis_title='Train release-cases, не число пациентов',height=300),key='dynamics_data')
        st.caption('Train здесь — один и тот же заранее заданный clean train-monitor8, а не полный train-N; Gap считает native train-monitor8 минус native development10. SD при менее двух seed отсутствует.')
        st.caption('Пять размеров/три seed подготовлены в RunPod matrix. SD по seed не оценивается по одному запуску; пустые точки не равны нулю. Не экстраполировать точное число необходимых пациентов.')
    with tabs[5]:
        summary_path=directory/'evaluation'/chosen_path.stem/'summary.json' if chosen_path.parent==directory else OUT/'references'/('CHD68_v1' if 'v1' in selected else 'CHD68_v2')/'summary.json'
        if summary_path.exists():
            values=[r for r in read(summary_path)['records'] if r['role']=='development']
            st.caption('Полнообъёмные показатели выбранного checkpoint: '+selected)
            st.dataframe(pd.DataFrame([{'case':r['case'],'native_Dice':r['macro_foreground_Dice'],'prepared_Dice':r['prepared_grid']['macro_foreground_Dice'],'recall':r['foreground_recall'],'BG_FP_voxels':r['BG_to_foreground_voxels']} for r in values]),hide_index=True)
            stats={r['case']:r for r in read(OUT/'case_statistics.json',[])}
            feature=st.selectbox('Generalization characteristic',['slice_count','foreground_fraction','source_sample_mean','source_sample_std','GT_bbox_FOV'])
            train_values=np.array([stats[c][feature] for c in split['train']]);mean=train_values.mean();sd=train_values.std()
            fig=go.Figure(go.Scatter(x=[(stats[r['case']][feature]-mean)/max(sd,1e-12) for r in values],y=[r['macro_foreground_Dice'] for r in values],mode='markers+text',text=[r['case'] for r in values],textposition='top center'))
            st.plotly_chart(fig.update_layout(title=feature+' vs native Dice',xaxis_title='Отклонение от train mean / train SD (diagnostic only)',height=350),key='dynamics_generalization')
        else:st.info('Полнообъёмные показатели этого checkpoint ещё не рассчитаны; выше показан только выбранный Z-срез.')
        st.caption('Case-level анализ10 development исследований. Связи с размерами/интенсивностями/GT геометрией описательные; не выводить причинность из корреляций.')
    with tabs[6]:
        rows=[]
        for path in (directory/'evaluation').glob('step*/*/topology.json'):
            data=read(path)
            if data['case']!=case:continue
            rows.append({'step':int(path.parent.parent.name[4:]),'false_components':sum(c['false_components_no_GT_overlap'] for c in data['topology'].values()),'false_volume_voxels':sum(c['false_component_voxels'] for c in data['topology'].values()),'Dice':data['macro_foreground_Dice']})
        if rows:
            frame=pd.DataFrame(rows).sort_values('step');st.dataframe(frame,hide_index=True)
            st.plotly_chart(go.Figure(go.Scatter(x=frame['step'],y=frame['false_components'],mode='lines+markers')).update_layout(title=case+' · ложные компоненты по шагам',height=300),key='dynamics_topology')
        topology_path=directory/'evaluation'/chosen_path.stem/case/'topology.json' if chosen_path.parent==directory else OUT/'references'/('CHD68_v1' if 'v1' in selected else 'CHD68_v2')/case/'topology.json'
        if topology_path.exists():
            measured=read(topology_path);st.dataframe(pd.DataFrame([{'class':LABELS[c],'HD95 index units':measured['topology'][str(c)]['HD95_index_units'],'components6':measured['topology'][str(c)]['components_6connected'],'largest_voxels':measured['topology'][str(c)]['largest_component_voxels'],'false_fraction':measured['topology'][str(c)]['false_component_fraction']} for c in range(1,8)]),hide_index=True)
            sizes=[comp['voxels'] for value in measured['topology'].values() for comp in value['components'] if comp['false_no_overlap']]
            if sizes:st.plotly_chart(go.Figure(go.Histogram(x=sizes)).update_layout(title='Размеры ложных островков выбранного checkpoint/case',xaxis_title='Voxels, не мм³',xaxis_type='log',height=280),key='dynamics_false_sizes')
        st.caption('Raw masks,6-connectivity, ignore исключён. Малые компоненты не удаляются. Меньше островков не доказывает лучшую анатомию; смотреть вместе с Dice/recall/HD95 в index units.')
