"""Read-only encoder inspection, train-fit projections and source-coordinate links."""
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import streamlit as st
import plotly.graph_objects as go
from ct_lab.learnable_core import OUT,SETS,FEATURES,PAIRS,normalize,load_encoder,infer
from ct_lab.learnable_data import assemble
from ct_lab.learnable_diagnostics import selection,perturb,KINDS
from ct_lab.learnable_train import knn_probability
from ct_lab.spatial_features import extract,columns,PLANES,check_affine
from ct_lab.spatial_ui import png,VIEWER
from ct_lab.spatial_projection import project
from ct_lab.core import LABELS


@st.cache_resource(max_entries=16)
def loaded(path):return load_encoder(path)


@st.cache_data(max_entries=12)
def projected(pair,feature_set,method,dim,kind):
    result=json.loads((OUT/'results.json').read_text());plan=result['protocol'];p=list(PAIRS).index(pair)
    train=assemble(plan,'train',p,SETS[feature_set]);val=assemble(plan,'validation',p,SETS[feature_set]);ti=selection(train);vi=selection(val)
    sample={k:np.concatenate([train[k][ti],val[k][vi]]) for k in ('labels','cases','xyz','ids')};fold=np.arange(len(ti)+len(vi))<len(ti)
    base=next(b for b in result['baselines'] if b['pair']==pair and b['features']==feature_set)
    values=np.concatenate([normalize(train['features'][ti],base['scaler']),normalize(val['features'][vi],base['scaler'])])
    if method!='k-NN':
        receipt=next(m for m in result['models'] if m['pair']==pair and m['features']==feature_set and m['method']==method and m['embedding_dim']==dim)
        model,_=loaded(str(OUT/receipt['checkpoint']));_,values=infer(model,values)
    embedding,info=project(values,fold,kind)
    return {**sample,'embedding':embedding,'fold':fold,'projection_info':info}


@st.cache_data(max_entries=8)
def baseline_arrays(pair,feature_set):
    result=json.loads((OUT/'results.json').read_text());plan=result['protocol'];p=list(PAIRS).index(pair)
    data=assemble(plan,'train',p,SETS[feature_set]);base=next(b for b in result['baselines'] if b['pair']==pair and b['features']==feature_set)
    return normalize(data['features'],base['scaler']),data['labels'],base['scaler']


def render(volume,mask,affine,case):
    task=st.selectbox('Feature learning task',['Binary stage03','Multiclass stage04'])
    if task=='Multiclass stage04':
        from ct_lab.multiclass_ui import render as multiclass_render
        multiclass_render(volume,mask,affine,case);return
    st.subheader('Learnable Feature Explorer')
    st.caption('Небольшой feature encoder для бинарной классификации sampled voxels, не whole-volume сегментатор. Source intensity не HU; spacing не подтверждён как мм.')
    if not (OUT/'results.json').exists():st.info('Эксперимент ещё выполняется; завершённые checkpoint находятся в outputs/learnable_feature_03/models.');return
    result=json.loads((OUT/'results.json').read_text());plan=result['protocol'];check_affine(affine)
    controls=st.columns(4)
    pair=controls[0].selectbox('Learnable anatomy pair',list(PAIRS),format_func=lambda x:'/'.join(LABELS[c] for c in PAIRS[x]))
    feature_set=controls[1].selectbox('Learnable input features',plan['neural_inputs'])
    method=controls[2].selectbox('Learnable model',['k-NN',*plan['methods']])
    dim=controls[3].selectbox('Embedding size',[8,16,32],index=1,disabled=method=='k-NN')
    kind=st.selectbox('Learnable projection',['PCA','UMAP'])
    features=SETS[feature_set];classes=PAIRS[pair]
    receipt=next(b for b in result['baselines'] if b['pair']==pair and b['features']==feature_set) if method=='k-NN' else next(
        m for m in result['models'] if m['pair']==pair and m['features']==feature_set and m['method']==method and m['embedding_dim']==dim)
    if method=='k-NN':model=None;train,labels,scaler=baseline_arrays(pair,feature_set)
    else:model,saved=loaded(str(OUT/receipt['checkpoint']));scaler=saved['scaler']
    projected_data=projected(pair,feature_set,method,dim,kind)
    keep=projected_data['cases']==case
    if not keep.any():st.info('Нет sampled voxels выбранной пары в этом исследовании.');return
    sample={k:projected_data[k][keep] for k in ('labels','cases','xyz','ids','embedding')}
    fig=go.Figure()
    for cls,color in zip(classes,('#ef476f','#06d6a0')):
        chosen=sample['labels']==cls;custom=np.column_stack([sample['xyz'][chosen],sample['ids'][chosen],sample['labels'][chosen]])
        fig.add_trace(go.Scattergl(x=sample['embedding'][chosen,0].tolist(),y=sample['embedding'][chosen,1].tolist(),customdata=custom.tolist(),
                                  mode='markers',marker={'color':color,'size':6},name=LABELS[cls],
                                  hovertemplate='XYZ=%{customdata[0]},%{customdata[1]},%{customdata[2]}<br>ID=%{customdata[3]}<extra></extra>'))
    fig.update_layout(title=f'{kind} · {case} · {method} · train-fit basis',height=330,dragmode='lasso',clickmode='event+select')
    event=st.plotly_chart(fig,on_select='rerun',selection_mode=('points','box','lasso'),key=f'learnable_scatter_{case}_{pair}_{feature_set}_{method}_{dim}_{kind}')
    coordinate_key=f'learnable_xyz_{case}'
    if coordinate_key not in st.session_state:st.session_state[coordinate_key]=sample['xyz'][0].tolist()
    pending=st.session_state.pop('learnable_pending',None)
    if pending and pending['case']==case:st.session_state[coordinate_key]=pending['xyz']
    allowed=set(map(int,sample['ids']));ids=tuple(sorted({int(p['customdata'][3]) for p in event.get('selection',{}).get('points',[]) if len(p.get('customdata',[]))>=4 and int(p['customdata'][3]) in allowed}))
    signature=(case,pair,feature_set,method,dim,kind,ids)
    if ids and st.session_state.get('learnable_selection')!=signature:
        st.session_state['learnable_selection']=signature
        st.session_state[coordinate_key]=sample['xyz'][np.flatnonzero(np.isin(sample['ids'],ids))[0]].tolist()
    xyz=st.session_state[coordinate_key]
    with st.expander('Sampled voxel / CT overlay controls',False):
        index=st.selectbox('Inspect sampled voxel',range(len(sample['ids'])),format_func=lambda i:f"{int(sample['ids'][i])} · {LABELS[int(sample['labels'][i])]} · {sample['xyz'][i].tolist()}",key=f'learnable_sample_{case}_{pair}')
        if st.button('Перейти к sampled voxel'):st.session_state[coordinate_key]=sample['xyz'][index].tolist();st.rerun()
        alpha=st.slider('Learnable mask opacity',0.,1.,.3)
    selected=sample['xyz'][np.isin(sample['ids'],ids)].tolist() if ids else []
    spacing=np.diag(affine[:3,:3]);planes=[]
    for name,(a,b,c) in PLANES.items():
        plane=np.take(volume,xyz[c],axis=c).T;seg=np.take(mask,xyz[c],axis=c).T if mask is not None else None
        planes.append({'name':name,'axes':[a,b,c],'width':plane.shape[1],'height':plane.shape[0],'spacing':[float(spacing[a]),float(spacing[b])],
                       'source':png(plane,seg,alpha,classes)})
    clicked=VIEWER(planes=planes,xyz=xyz,selected=selected,diagnostics=[],key=f'learnable_viewer_{case}',default=None)
    if clicked and clicked['nonce']!=st.session_state.get('learnable_nonce'):
        st.session_state['learnable_nonce']=clicked['nonce'];st.session_state['learnable_pending']={'case':case,'xyz':clicked['xyz']};st.rerun()
    bank=extract(volume,np.array([xyz]),affine);raw=columns(bank,features);scaled=normalize(raw,scaler)
    if model is None:prob=knn_probability(train,labels,scaled,classes);trace=None
    else:
        with torch.inference_mode():trace={k:v.numpy() for k,v in model(torch.from_numpy(scaled),trace=True).items()}
        prob=trace['probabilities']
    st.write({'Source XYZ':xyz,'NIfTI RAS coordinate':nib.affines.apply_affine(affine,xyz).tolist(),'probabilities':dict(zip([LABELS[c] for c in classes],prob[0].tolist()))})
    if mask is not None and int(mask[tuple(xyz)]) not in classes:st.info('Выбранный voxel вне обученной пары. Эти две вероятности условны для выбранных классов и не являются multiclass сегментацией фона/сердца.')
    target=int(prob[0].argmax());rows=[]
    for i,name in enumerate(features):
        ablated=scaled.copy();ablated[0,i]=0
        if model is None:changed=knn_probability(train,labels,ablated,classes)
        else:changed,_=infer(model,ablated)
        rows.append({'feature':name,'source_value':float(raw[0,i]),'train_standardized_value':float(scaled[0,i]),
                     'predicted_class_probability_drop_at_train_mean':float(prob[0,target]-changed[0,target])})
    st.dataframe(rows)
    st.caption('Локальная ablation заменяет один канал training mean. Коррелированные признаки и нелинейные взаимодействия ограничивают интерпретацию; абсолютный вес Linear не является универсальной важностью.')
    if trace is not None:
        layer=st.selectbox('Embedding inspector layer',['linear1','relu1','linear2','relu2','embedding','probabilities'])
        st.plotly_chart(go.Figure(go.Bar(y=trace[layer][0])).update_layout(title=layer,height=280))
        history=json.loads((OUT/receipt['checkpoint']).with_name('history.json').read_text());history_plot=go.Figure()
        for key in ('train_case_f1','validation_case_f1'):history_plot.add_trace(go.Scatter(x=[h['epoch'] for h in history],y=[h[key] for h in history],name=key))
        st.plotly_chart(history_plot.update_layout(title=f'Training history; best epoch={receipt["best_epoch"]}',height=280))
    role='train' if case in plan['train_cases'] else 'validation'
    metric=receipt[role]['per_case'][case]
    st.dataframe([{'case':case,'role':role,'model':method,**metric}])
    st.plotly_chart(go.Figure(go.Heatmap(z=metric['confusion_matrix'],x=[LABELS[c] for c in classes],y=[LABELS[c] for c in classes],texttemplate='%{z}',colorscale='Blues')).update_layout(xaxis_title='Predicted',yaxis_title='GT',height=280))
    if method=='k-NN' and role=='train':st.caption('k-NN train scores — optimistic resubstitution с self-neighbors; validation используется для сравнения.')
    st.caption('Validation уже использовалась: ранняя остановка/выбор модели, не независимая финальная оценка. Проекции fit только train; одинаковые IDs между моделями. Красивые кластеры не означают рост Dice.')
    perturbation=st.selectbox('Selected voxel synthetic perturbation',KINDS)
    shifted=perturb(scaled,perturbation,features,[case],[np.ravel_multi_index(tuple(xyz),volume.shape)])
    if model is None:newprob=knn_probability(train,labels,shifted,classes)
    else:newprob,_=infer(model,shifted)
    st.write({'perturbed_probabilities':dict(zip([LABELS[c] for c in classes],newprob[0].tolist()))})
    st.caption('Это feature-space stress test в единицах training SD, а не моделирование реальных сканеров. При выключении канала используется train mean.')
    if (OUT/'importance.json').exists():
        importance=json.loads((OUT/'importance.json').read_text())
        st.dataframe([r for r in importance if r['pair']==pair and r['features']==feature_set and r['method']==method])
        if model is not None and dim!=16:st.caption('Global permutation table рассчитана для фиксированного e16; локальная ablation выше использует выбранный embedding size.')
