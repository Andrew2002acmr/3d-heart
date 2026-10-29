"""Eight-class extension inside the existing CT Lab and Three-plane Viewer."""
import base64
import io
import json
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from PIL import Image
import plotly.graph_objects as go
import streamlit as st
from ct_lab.multiclass_core import *
from ct_lab.multiclass_train import slug
from ct_lab.multiclass_projection import projection
from ct_lab.multiclass_dense import roi_bounds,infer_region,evaluate_region
from ct_lab.spatial_features import PLANES
from ct_lab.spatial_ui import VIEWER
os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache')
COLORS=['#9aa0a6','#ef476f','#118ab2','#9b5de5','#f15bb5','#06d6a0','#fb8500','#ffd166']


@st.cache_resource(max_entries=8)
def loaded(path):return load_model(path)


@st.cache_data(max_entries=10)
def projected(job,kind,representation):return projection(job,kind,representation)


@st.cache_data(max_entries=2)
def altered(volume,spec,seed):return augment(volume,spec,seed)


@st.cache_resource(max_entries=4)
def dense_arrays(path):
    p=Path(path)
    return {k:np.load(p/(k+'.npy'),mmap_mode='r') for k in ('prediction','probabilities','entropy')}


def picture(plane,mask=None,classes=(),alpha=.35,heat=None):
    gray=(np.clip(plane/4095,0,1)*255).astype(np.uint8);rgb=np.repeat(gray[:,:,None],3,2)
    if mask is not None:
        for c in classes:
            color=np.array(list(bytes.fromhex(COLORS[c][1:])));keep=mask==c;rgb[keep]=((1-alpha)*rgb[keep]+alpha*color).astype(np.uint8)
    if heat is not None:
        from matplotlib import colormaps
        keep=np.isfinite(heat);rgba=colormaps['inferno'](np.nan_to_num(np.clip(heat,0,1)))[:,:,:3]*255
        rgb[keep]=((1-alpha)*rgb[keep]+alpha*rgba[keep]).astype(np.uint8)
    stream=io.BytesIO();Image.fromarray(np.flipud(rgb)).save(stream,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


def render(volume,mask,affine,case):
    st.subheader('Multiclass Feature Lab')
    st.caption('8 классов · source intensity не HU · spacing не подтверждён как мм · development validation многократно использовалась.')
    if not (OUT/'results.json').exists():
        st.info('Подготовка данных / обучение выполняется. Эта вкладка не запускает обучение автоматически.');return
    result=json.loads((OUT/'results.json').read_text());plan=result['protocol']
    if case not in plan['train_cases']+plan['validation_cases']:st.info('Sealed test не открывается.');return
    controls=st.columns(3);feature=controls[0].selectbox('Multiclass feature set',list(SETS),index=4)
    target=cardiac_target(mask) if mask is not None else None
    candidates=[m for m in result['models'] if m['features']==feature]
    variants=list(dict.fromkeys((m['method'],m['augmentation']) for m in candidates))
    choice=controls[1].selectbox('Multiclass model',variants,format_func=lambda t:f'{t[0]} / {t[1]}',index=variants.index(('CE','combined')) if ('CE','combined') in variants else 0)
    seed=controls[2].selectbox('Multiclass seed',SEEDS)
    receipt=next(m for m in candidates if (m['method'],m['augmentation'])==choice and m['seed']==seed)
    job={k:receipt[k] for k in ('features','method','augmentation','seed')};model,saved=loaded(str(OUT/receipt['checkpoint']))
    st.caption('Оценка ниже — на фиксированных равномерных spatial IDs с естественными частотами; scatter использует отдельную диагностическую стратификацию.')
    a,b=st.columns(2);kind=a.selectbox('Multiclass projection',['PCA','UMAP']);representation=b.selectbox('Multiclass representation',['Embedding','Inputs before training'])
    chosen=st.multiselect('Отображаемые foreground классы',list(range(1,8)),default=list(range(1,8)),format_func=lambda c:LABELS[c])
    background=st.checkbox('Показывать background',value=False);classes=([0] if background else [])+chosen
    sample,info=projected(job,kind,'Embedding' if representation=='Embedding' else 'Inputs')
    visible=(sample['cases']==case)&np.isin(sample['labels'],classes)
    views=st.columns(2);selected_ids=[]
    for col,condition,title in [(views[0],'clean','Clean'),(views[1],'noise8' if case in plan['train_cases'] else 'noise12','CT Gaussian noise')]:
        fig=go.Figure();keep=visible&(sample['condition']==condition)
        for c in classes:
            take=keep&(sample['labels']==c);custom=np.column_stack([sample['xyz'][take],sample['ids'][take],sample['labels'][take]])
            fig.add_trace(go.Scattergl(x=sample['embedding'][take,0].tolist(),y=sample['embedding'][take,1].tolist(),customdata=custom.tolist(),
                                      mode='markers',name=LABELS[c],marker={'color':COLORS[c],'size':5},hovertemplate='XYZ %{customdata[0]},%{customdata[1]},%{customdata[2]}<br>ID %{customdata[3]}<extra></extra>'))
        fig.update_layout(title=f'{title} · {case} · {kind}',height=330,dragmode='lasso',clickmode='event+select')
        event=col.plotly_chart(fig,on_select='rerun',selection_mode=('points','box','lasso'),key=f'mc_scatter_{case}_{slug(job)}_{representation}_{kind}_{condition}')
        allowed=set(map(int,sample['ids'][keep]))
        selected_ids.extend(int(p['customdata'][3]) for p in event.get('selection',{}).get('points',[]) if len(p.get('customdata',[]))>=4 and int(p['customdata'][3]) in allowed)
    selected_ids=sorted(set(selected_ids));clean=(sample['cases']==case)&(sample['condition']=='clean');ids=sample['ids'][clean];coords=sample['xyz'][clean]
    key=f'multiclass_xyz_{case}'
    if key not in st.session_state:st.session_state[key]=np.rint(np.array(volume.shape)*plan['dense_ROI']['center_fraction']).astype(int).tolist()
    signature=(case,slug(job),kind,representation,tuple(selected_ids))
    if selected_ids and st.session_state.get('mc_selection')!=signature:
        st.session_state['mc_selection']=signature;st.session_state[key]=coords[np.flatnonzero(np.isin(ids,selected_ids))[0]].tolist()
    pending=st.session_state.pop('mc_pending',None)
    if pending and pending['case']==case:st.session_state[key]=pending['xyz']
    xyz=st.session_state[key];selected=coords[np.isin(ids,selected_ids)].tolist() if selected_ids else []
    st.caption('Clean/noisy используют общую проекцию, fit только на clean TRAIN. На train показан σ=8, на validation — новый σ=12 source counts. GT используется для окраски, не для reducer fit.')
    with st.expander('КТ, аугментации и dense overlay',expanded=True):
        spec_name=st.selectbox('CT исследовательское преобразование',list(VAL_AUG))
        processed=volume if spec_name=='clean' else altered(volume,VAL_AUG[spec_name],SEEDS[0]+int(case.split('_')[1])+900001+list(VAL_AUG).index(spec_name)*10007)
        compare=st.checkbox('Сравнить исходную и обработанную КТ',value=spec_name!='clean')
        if compare:
            first,second=st.columns(2);first.image(picture(volume[:,:,xyz[2]].T),caption='Исходная КТ');second.image(picture(processed[:,:,xyz[2]].T),caption=spec_name)
        layer=st.selectbox('Multiclass CT overlay',['GT','Source','Dense prediction','Errors','Entropy','Class probability'])
        opacity=st.slider('Multiclass opacity',0.,1.,.4);prob_class=st.selectbox('Probability class',CLASSES,format_func=lambda c:LABELS[c])
    scope=st.radio('Dense inference scope',['Fixed ROI','Full volume'],horizontal=True)
    dense_root=OUT/('dense_full' if scope=='Full volume' else 'dense')/case
    dense_path=dense_root/slug(job);available=[p for p in dense_root.glob('*') if (p/'metrics.json').exists()]
    if available:
        dense_choice=st.selectbox('Dense prediction model',[p.name for p in available],index=next((i for i,p in enumerate(available) if p==dense_path),0));dense_path=dense_root/dense_choice
    arrays=None;metric=None
    if (dense_path/'metrics.json').exists():
        metric=json.loads((dense_path/'metrics.json').read_text());lower=np.array(metric['lower_xyz']);upper=np.array(metric['upper_xyz']);arrays=dense_arrays(str(dense_path))
    elif layer not in ('GT','Source'):st.info('Для выбранной модели ROI ещё не рассчитана. Используйте кнопку ниже.');layer='GT'
    if spec_name!='clean' and layer not in ('GT','Source'):
        st.info('Dense overlay рассчитан на исходной КТ; для согласованных координат ниже показана исходная КТ. Преобразование выше — отдельный просмотр.')
    planes=[];spacing=check_affine(affine)
    for name,(a,b,c) in PLANES.items():
        plane=np.take(volume,xyz[c],axis=c).T;seg=np.take(mask,xyz[c],axis=c).T if mask is not None and layer=='GT' else None;heat=None
        if arrays is not None and layer not in ('GT','Source') and lower[c]<=xyz[c]<upper[c]:
            slices=[slice(None)]*3;slices[c]=int(xyz[c]-lower[c]);slices=tuple(slices)
            region=arrays['prediction'][slices].T;seg=np.full(plane.shape,255,np.int16)
            patch=(slice(lower[b],upper[b]),slice(lower[a],upper[a]));seg[patch]=region
            if layer=='Errors':
                truth=target[tuple(slice(int(lo),int(hi)) if ax!=c else xyz[c] for ax,(lo,hi) in enumerate(zip(lower,upper)))].T
                seg[patch]=np.where((truth<8)&(truth!=region),1,255)
            elif layer in ('Entropy','Class probability'):
                heat=np.full(plane.shape,np.nan,np.float32);heat[patch]=(arrays['entropy'][slices].T/np.log(8)) if layer=='Entropy' else arrays['probabilities'][slices+(prob_class,)].T;seg=None
        planes.append({'name':name,'axes':[a,b,c],'width':plane.shape[1],'height':plane.shape[0],'spacing':[float(spacing[a]),float(spacing[b])],
                       'source':picture(plane,seg,[1] if layer=='Errors' else classes,opacity,heat)})
    event=VIEWER(planes=planes,xyz=xyz,selected=selected,diagnostics=[],key=f'multiclass_viewer_{case}',default=None)
    if event and event['nonce']!=st.session_state.get('mc_nonce'):
        st.session_state['mc_nonce']=event['nonce'];st.session_state['mc_pending']={'case':case,'xyz':event['xyz']};st.rerun()
    mapped,valid=transformed_xyz(np.array([xyz]),VAL_AUG[spec_name],volume.shape);raw=extract_points(processed,mapped,affine)
    probability,z=infer(model,normalize(columns(raw,saved['features']),saved['scaler']))
    st.write({'Source XYZ':xyz,'Transformed XYZ':mapped[0].tolist(),'NIfTI coordinate (units unknown)':nib.affines.apply_affine(affine,xyz).tolist()})
    st.plotly_chart(go.Figure(go.Bar(x=LABELS,y=probability[0])).update_layout(title='Вероятности выбранного voxel · не клиническая калибровка',height=240))
    if not valid[0]:st.info('Source voxel после сдвига вышел за поле зрения; вероятность в clipped координате не соответствует исходной анатомии.')
    if target is not None and target[tuple(xyz)]==255:st.caption('Экспертная метка этого voxel — ignore=255: она исключена из метрик и не является background.')
    if arrays is not None and np.all(np.array(xyz)>=lower) and np.all(np.array(xyz)<upper):
        local=tuple(np.array(xyz)-lower);st.caption(f'Dense {dense_path.name}: class={LABELS[int(arrays["prediction"][local])]}, entropy={float(arrays["entropy"][local]):.3f}')
    role='train_natural' if case in plan['train_cases'] else 'validation';score=receipt[role]['per_case'][case]
    st.plotly_chart(go.Figure(go.Heatmap(z=score['confusion_matrix'],x=LABELS,y=LABELS,texttemplate='%{z}',colorscale='Blues')).update_layout(title=f'8×8 CM · {case} · natural sampling',height=360))
    st.plotly_chart(go.Figure(go.Bar(x=LABELS,y=[score['classwise'][str(c)]['f1'] for c in CLASSES])).update_layout(title='F1 по классам на natural spatial sample',height=260))
    with st.expander('Обучение, устойчивость и важность признаков'):
        cases=receipt['validation']['per_case']
        st.plotly_chart(go.Figure(go.Bar(x=list(cases),y=[r['macro_f1'] for r in cases.values()])).update_layout(title='Validation F1 по исследованиям · выбранный seed',height=260))
        history=json.loads((OUT/receipt['checkpoint']).with_name('history.json').read_text());fig=go.Figure()
        for field in ('train_balanced_case_f1','validation_natural_case_f1'):fig.add_trace(go.Scatter(x=[h['epoch'] for h in history],y=[h[field] for h in history],name=field))
        st.plotly_chart(fig.update_layout(title='F1 / epoch · early stopping development validation',height=260))
        st.plotly_chart(go.Figure(go.Scatter(x=[h['epoch'] for h in history],y=[h['loss'] for h in history])).update_layout(title='Training loss',height=230))
        if (OUT/'robustness.json').exists():
            robust=[r for r in json.loads((OUT/'robustness.json').read_text()) if all(r[k]==job[k] for k in job)]
            st.plotly_chart(go.Figure(go.Bar(x=[r['transformation'] for r in robust],y=[r['metrics']['case_macro']['macro_f1_mean'] for r in robust])).update_layout(title='Clean vs transformed CT · case F1',height=260))
            st.plotly_chart(go.Figure(go.Bar(x=[r['transformation'] for r in robust],y=[r['embedding_unit_RMSE'] for r in robust])).update_layout(title='Embedding stability · unit RMSE',height=250))
        if (OUT/'importance.json').exists():
            importance=[r for r in json.loads((OUT/'importance.json').read_text()) if r['method']==job['method'] and r['augmentation']==job['augmentation']]
            st.plotly_chart(go.Figure(go.Bar(x=[r['feature'] for r in importance],y=[r['F1_drop'] for r in importance])).update_layout(title='Permutation importance · fixed first seed, E inputs',height=260))
            st.caption('Перестановка внутри case нарушает корреляции; это диагностическая, не причинная важность.')
    if st.button('Вычислить фиксированную ROI для выбранной MLP'):
        directory=OUT/'dense'/case/slug(job);lower,upper=roi_bounds(volume.shape,plan['dense_ROI'])
        with st.spinner('Пакетный inference с halo; ROI выбрана без GT…'):
            pred=infer_region(volume,affine,model,saved['scaler'],saved['features'],lower,upper,directory,plan['dense_block_xyz'])
            source=nib.load(DEFAULT_DATA/f'{case}_image.nii.gz');evaluate_region(directory,pred,cardiac_target(mask),lower,upper,source,{'case':case,'model':slug(job)})
        st.rerun()
    if metric is not None:
        with st.expander('3D-анализ фрагментации и диагностический Dice / HD95'):
            fig=go.Figure();components=json.loads((dense_path/'components.json').read_text())
            for item in components:
                pts=np.array(item['xyz']);c=item['class']
                if c not in classes:continue
                fig.add_trace(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='markers',marker={'size':2,'color':COLORS[c]},name=f'{LABELS[c]} · {item["size"]} voxels'))
            st.plotly_chart(fig.update_layout(title='3D components: точки крупнейших компонент; counts по dense ROI',height=450,scene={'aspectmode':'data'}))
            st.dataframe([{'class':LABELS[c],**{k:v for k,v in metric['dense_classwise'][str(c)].items() if k!='component_sizes_desc'}} for c in CLASSES])
            st.caption('6-connectivity. Компоненты на краях ROI могут быть усечены. HD95 в индексной сетке, не мм; TP/FP оцениваются только на valid GT, ignore исключён.')
    st.caption('Dense MLP — локальный voxel classifier. ROI зависит только от формы CT; маска входит после inference для оценки/overlay. Dice здесь измерен отдельно на dense prediction, а не выведен из sampled F1.')
