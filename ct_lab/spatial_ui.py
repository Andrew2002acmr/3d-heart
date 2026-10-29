"""Linked three-plane source viewer and train-only spatial feature projections."""
import base64
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import nibabel as nib
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from ct_lab.spatial_features import extract,columns,feature_sets,PLANES,check_affine
from ct_lab.spatial_run import OUT,PREVIOUS
from ct_lab.filtering import sample_voxels,projection,fit_scaler,standardize,SEED
from ct_lab.spatial_projection import project
from ct_lab.core import LABELS

VIEWER=components.declare_component('spatial_three_plane',path=str(Path(__file__).parent/'spatial_component'))
COLORS={1:'#ef476f',2:'#118ab2',3:'#9b5de5',4:'#f15bb5',5:'#06d6a0',6:'#fb8500',7:'#ffd166'}


def png(plane,mask=None,alpha=.3,labels=(),low=0.,high=4095.):
    gray=(np.clip((plane-low)/(high-low),0,1)*255).astype(np.uint8);rgb=np.repeat(gray[:,:,None],3,axis=2)
    if mask is not None:
        for cls in labels:
            color=tuple(bytes.fromhex(COLORS[cls][1:]))
            rgb[mask==cls]=((1-alpha)*rgb[mask==cls]+alpha*np.array(color)).astype(np.uint8)
    stream=io.BytesIO();Image.fromarray(np.flipud(rgb)).save(stream,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


@st.cache_data(max_entries=4,show_spinner='Извлечение только sampled spatial patches…')
def bank_for(source,xyz,affine):return extract(source,xyz,affine)


@st.cache_data(max_entries=8)
def project_local(values,fold,kind):return project(values,fold,kind)


def scatter(values,names,sample,pair,title):
    fig=go.Figure()
    for cls in pair:
        keep=sample['labels']==cls;custom=np.column_stack([sample['xyz'][keep],sample['voxel_ids'][keep],sample['labels'][keep]])
        fig.add_trace(go.Scattergl(x=values[keep,0].tolist(),y=values[keep,1].tolist(),customdata=custom.tolist(),mode='markers',
                                  name=LABELS[cls],marker=dict(size=4,color=COLORS[cls],opacity=.65),
                                  hovertemplate='XYZ=%{customdata[0]},%{customdata[1]},%{customdata[2]}<br>ID=%{customdata[3]}<extra>%{fullData.name}</extra>'))
    fig.update_layout(title=title,xaxis_title=names[0],yaxis_title=names[1],height=330,dragmode='lasso',clickmode='event+select',margin=dict(l=25,r=10,t=40,b=25))
    return fig


def render(source,mask,affine,case):
    st.subheader('Spatial Feature Explorer')
    spacing=check_affine(affine)
    st.caption('Один исходный XYZ volume и один affine. Source intensity; header grid units не подтверждены как мм. GT/расстояния — только диагностика, не признаки.')
    key=f'spatial_position_{case}'
    if key not in st.session_state:st.session_state[key]=[n//2 for n in source.shape]
    pending=st.session_state.pop('spatial_pending_position',None)
    if pending and pending['case']==case:
        st.session_state[key]=pending['xyz']
        for a in range(3):st.session_state[f'spatial_coord_{case}_{a}']=int(pending['xyz'][a])
    for a in range(3):
        if f'spatial_coord_{case}_{a}' not in st.session_state:st.session_state[f'spatial_coord_{case}_{a}']=int(st.session_state[key][a])
    with st.expander('Позиция, классы, маска и выборка',expanded=False):
        cols=st.columns(3)
        xyz=[int(cols[a].number_input(f'Source {name}',0,source.shape[a]-1,key=f'spatial_coord_{case}_{a}')) for a,name in enumerate('XYZ')]
        choice=st.selectbox('Пара структур',['LV/MYO','RV/PA','Произвольная foreground пара'])
        if choice=='Произвольная foreground пара':
            pair=tuple(st.multiselect('Выберите два foreground класса',list(range(1,8)),default=[1,5],format_func=lambda c:LABELS[c],max_selections=2))
        else:pair=(1,5) if choice=='LV/MYO' else (2,7)
        alpha=st.slider('Spatial mask opacity',0.,1.,.3)
        overlay=st.checkbox('Spatial expert mask',True)
        low=st.number_input('Spatial display low',value=0.);high=st.number_input('Spatial display high',value=4095.)
    if len(pair)!=2 or high<=low:st.info('Нужны два класса и корректная шкала.');return
    st.session_state[key]=xyz
    frozen=json.loads((Path(__file__).resolve().parents[1]/'configs/splits/cardiac_chd68_v1.json').read_text())
    validation=case in frozen['partitions']['validation']
    cached=OUT/f'case_{case}.npz'
    if cached.exists() and pair in ((1,5),(2,7)):
        p=0 if pair==(1,5) else 1
        with np.load(cached) as data:
            sample={k:data[f'p{p}_{k}'].copy() for k in ('xyz','labels','voxel_ids','train')}
            values=data[f'p{p}_values'].copy()
        info=json.loads(cached.with_suffix('.json').read_text());bank={'names':info['names'],'values':values}
        st.caption(f'Frozen sampled IDs этапа 1: {len(values)} voxels, три Z-позиции. Выборка одинакова A–H и не меняется при перемещении перекрестия.')
    elif validation:
        st.info('Для validation доступны только заранее извлечённые пары и фиксированные результаты; локальный fit отключён.');return
    else:
        if mask is None:st.info('Для стратифицированного анализа нужна экспертная маска.');return
        s=sample_voxels(mask[:,:,xyz[2]].T,pair,case,xyz[2],source.shape,per_class=400)
        sample={k:s[k] for k in ('xyz','labels','voxel_ids','train')}
        if len(sample['labels'])<10 or any(np.count_nonzero(sample['labels']==c)<5 for c in pair):st.info('В этом срезе мало вокселей выбранных классов.');return
        bank=bank_for(source,sample['xyz'],affine)
    sets=feature_sets();configuration=st.selectbox('Spatial feature set',list(sets),index=4)
    a,b=st.columns(2)
    with a:xname=st.selectbox('Spatial scatter X',bank['names'],index=bank['names'].index('Intensity'))
    with b:yname=st.selectbox('Spatial scatter Y',bank['names'],index=bank['names'].index('Z3 Std'))
    plot=scatter(columns(bank,[xname,yname]),[xname,yname],sample,pair,'Выделение связано с тремя плоскостями')
    event=st.plotly_chart(plot,key=f'spatial_scatter_{case}_{pair}',on_select='rerun',selection_mode=('points','box','lasso'))
    allowed=set(map(int,sample['voxel_ids']));points=event.get('selection',{}).get('points',[])
    ids=tuple(sorted({int(p['customdata'][3]) for p in points if len(p.get('customdata',[]))>=4 and int(p['customdata'][3]) in allowed}))
    signature=(case,pair,ids)
    if ids and st.session_state.get('spatial_last_selection')!=signature:
        st.session_state['spatial_last_selection']=signature
        chosen=sample['xyz'][np.flatnonzero(np.isin(sample['voxel_ids'],ids))[0]].tolist()
        st.session_state['spatial_pending_position']={'case':case,'xyz':chosen};st.rerun()
    selected=sample['xyz'][np.isin(sample['voxel_ids'],ids)].tolist() if ids else []
    diagnostics=[]
    error_file=OUT/f'errors_{case}.npz'
    if error_file.exists() and pair in ((1,5),(2,7)):
        mode=st.selectbox('Spatial error overlay',['None','Correct / wrong','Correction vs 2D B'])
        if mode!='None':
            p=0 if pair==(1,5) else 1;letter=configuration[0]
            with np.load(error_file) as data:
                truth=data[f'p{p}_truth'];base=data[f'p{p}_B'];pred=data[f'p{p}_{letter}'];positions=data[f'p{p}_xyz'];distance=data[f'p{p}_distance']
            for position,t,b,c in zip(positions,truth,base,pred):
                color=('#30cf70' if c==t else '#ef4444') if mode=='Correct / wrong' else ('#06d6ef' if b!=t and c==t else '#ec4899' if b==t and c!=t else '#30cf70' if c==t else '#ef4444')
                diagnostics.append({'xyz':position.tolist(),'color':color})
            st.caption('Error colors: green correct; red wrong; cyan corrected vs B; magenta new error vs B. Only evaluated sampled voxels. Boundary distance is diagnostic, never input. Train cases use optimistic voxel holdout; validation is development continuation.')
            st.dataframe([{'region':name,'n':int(keep.sum()),'baseline_wrong':int(np.count_nonzero((base!=truth)&keep)),
                           'corrected':int(np.count_nonzero((base!=truth)&(pred==truth)&keep)),
                           'new_wrong':int(np.count_nonzero((base==truth)&(pred!=truth)&keep))}
                          for name,keep in [('boundary ≤1',distance<=1),('transition (1,3]',(distance>1)&(distance<=3)),('interior >3',distance>3)]])
    planes=[]
    for name,(a,b,fixed) in PLANES.items():
        plane=np.take(source,xyz[fixed],axis=fixed).T
        seg=np.take(mask,xyz[fixed],axis=fixed).T if mask is not None else None
        planes.append({'name':name,'axes':[a,b,fixed],'width':plane.shape[1],'height':plane.shape[0],
                       'spacing':[float(spacing[a]),float(spacing[b])],'source':png(plane,seg,alpha,pair if overlay else (),low,high)})
    value=VIEWER(planes=planes,xyz=xyz,selected=selected,diagnostics=diagnostics,key=f'spatial_viewer_{case}',default=None)
    if value and value.get('nonce')!=st.session_state.get('spatial_click_nonce'):
        st.session_state['spatial_click_nonce']=value['nonce'];chosen=value['xyz']
        if all(0<=chosen[a]<source.shape[a] for a in range(3)):
            st.session_state['spatial_pending_position']={'case':case,'xyz':chosen};st.rerun()
    world=nib.affines.apply_affine(affine,xyz)
    st.write({'Source XYZ':xyz,'NIfTI RAS coordinate':world.round(4).tolist(),'source intensity':float(source[tuple(xyz)]),'selected voxels':len(selected)})
    radius=st.select_slider('Neighbor radius',options=[1,2,3],value=3)
    neighbors=st.columns(2*radius+1)
    for col,offset in zip(neighbors,range(-radius,radius+1)):
        z=int(np.clip(xyz[2]+offset,0,source.shape[2]-1))
        with col:
            fig=go.Figure(go.Heatmap(z=source[:,:,z].T,colorscale='Gray',zmin=low,zmax=high,showscale=False))
            fig.add_trace(go.Scatter(x=[xyz[0]],y=[xyz[1]],mode='markers',marker=dict(color='cyan',size=4)))
            fig.update_layout(title=f'{offset:+d} → Z={z}',height=180,margin=dict(l=0,r=0,t=30,b=0),yaxis=dict(scaleanchor='x'))
            st.plotly_chart(fig)
    profiles=go.Figure()
    for axis,name in enumerate('XYZ'):
        offsets=np.arange(-3,4);points=np.tile(xyz,(7,1));points[:,axis]=np.clip(points[:,axis]+offsets,0,source.shape[axis]-1)
        profiles.add_trace(go.Scatter(x=offsets*spacing[axis],y=source[tuple(points.T)],name=name,mode='lines+markers'))
    profiles.update_layout(title='Intensity profiles XYZ at the same source position',xaxis_title='Header grid displacement (not mm)',yaxis_title='Source intensity',height=300)
    st.plotly_chart(profiles)
    score_path=OUT/('results.json' if validation else 'train_development.json')
    if score_path.exists() and pair in ((1,5),(2,7)):
        saved=json.loads(score_path.read_text());saved=saved['scores'] if isinstance(saved,dict) else saved
        pair_label='/'.join(LABELS[c] for c in pair)
        matching=[s for s in saved if s['pair']==pair_label and s['configuration']==configuration and case in s['per_case']
                  and (not validation or s['scope']=='main unfiltered')]
        if matching:
            metric=matching[0]['per_case'][case]
            for col,name in zip(st.columns(4),('macro_f1','balanced_accuracy','silhouette','fisher_ratio')):
                value=metric[name];col.metric(name,'n/a' if value is None else f'{value:.4f}')
            st.caption('Fixed k-NN k=15; scaler from training only. '+('Previously used validation: development continuation.' if validation else 'Five-group train voxel holdout: optimistic, not group-independent.'))
            if st.checkbox('Spatial confusion matrix',False):
                st.plotly_chart(go.Figure(go.Heatmap(z=metric['confusion_matrix'],x=[LABELS[c] for c in pair],y=[LABELS[c] for c in pair],texttemplate='%{z}',colorscale='Blues'))
                                .update_layout(xaxis_title='Predicted',yaxis_title='Expert',height=300))
    if validation:
        st.info('Validation уже использовалась: только просмотр фиксированных метрик/ошибок, без fit scaler/PCA/UMAP по validation.')
        result=OUT/'results.json'
        if result.exists():
            rows=json.loads(result.read_text())['scores']
            pair_label='/'.join(LABELS[c] for c in pair)
            st.dataframe([{'configuration':s['configuration'],**s['per_case'][case]} for s in rows if s['scope']=='main unfiltered' and s['pair']==pair_label and case in s['per_case']])
        return
    projection_metadata={}
    if st.checkbox('Spatial PCA/UMAP (fit только train voxel fold)',True):
        kind=st.selectbox('Spatial projection',['PCA','UMAP'])
        before=columns(bank,sets['B 2D']);after=columns(bank,sets[configuration]);fold=sample['train']
        if np.count_nonzero(~fold)<2:st.info('Недостаточно holdout voxels');return
        for col,title,values in zip(st.columns(2),['2D baseline','Added spatial context'],[before,after]):
            embedding,info=project_local(values,fold,kind)
            projection_metadata[title]={'kind':kind,**info}
            with col:st.plotly_chart(scatter(embedding,[kind+' 1',kind+' 2'],sample,pair,title))
        if st.checkbox('Общая проекция пересечения признаков — контроль',False):
            common=[n for n in sets['B 2D'] if n in sets[configuration]]
            values=columns(bank,common);embedding,info=project_local(values,fold,kind)
            projection_metadata['shared_intersection']={'common_columns':common,'before_after_identical':True,'kind':kind,**info}
            st.plotly_chart(scatter(embedding,[kind+' 1',kind+' 2'],sample,pair,'Shared intersection: before = after'))
            st.caption('Одна basis/scaler fit по train. На пересечении неизменённых признаков before/after совпадают. Это контроль; новые оси исключены, поэтому он не измеряет пользу добавленных каналов. Нельзя напрямую применять одну basis к матрицам разной размерности.')
        st.caption('Независимые PCA/UMAP имеют разные basis; расстояния между их координатами не сопоставимы. Цвет не передаётся reducer, кластеры не доказывают рост Dice.')
    if st.button('Сохранить spatial просмотр и sampled IDs'):
        import uuid
        output=OUT/'interactive'/uuid.uuid4().hex;output.mkdir(parents=True)
        (output/'config.json').write_text(json.dumps({'case':case,'xyz':xyz,'world_RAS':world.tolist(),'pair':pair,'configuration':configuration,
                                                    'selected_ids':list(ids),'affine':affine.tolist(),'source_units':'unverified','seed':SEED,
                                                    'feature_names':sets[configuration],'projections':projection_metadata},indent=2),encoding='utf-8')
        np.savez_compressed(output/'sampled_features.npz',**sample,values=bank['values'],names=np.array(bank['names']))
        (output/'scatter.html').write_text(plot.to_html(include_plotlyjs=True),encoding='utf-8');st.success(str(output))
