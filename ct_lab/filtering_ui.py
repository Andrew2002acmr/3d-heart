"""Linked Streamlit experiment 01, using native source-grid axial coordinates."""
import io
import json
import zipfile
import uuid
from pathlib import Path
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import nibabel as nib
from ct_lab.filtering import (FEATURES,PAIRS,SEED,apply_filter,feature_maps,sample_voxels,matrix,
                              fixed_regions,measurements,projection,knn_score,selected_voxel_ids)
from ct_lab.core import LABELS,fingerprint

COLORS={1:'#ef476f',5:'#06d6a0',2:'#118ab2',7:'#ffd166'}


@st.cache_data(max_entries=12)
def calculate(source,config,spacing):
    after,provenance=apply_filter(source,config,spacing)
    return after,feature_maps(source,spacing),feature_maps(after,spacing),provenance


@st.cache_data(max_entries=8,show_spinner='Проекция: fit только по train-части sampled voxels…')
def projected(values,train,kind):
    return projection(values,train,kind)


@st.cache_data(max_entries=16)
def diagnostic_scores(before,after,labels,train):
    return [dict(stage=name,feature_set=feature_name,**knn_score(vals[train],labels[train],vals[~train],labels[~train],cols))
            for name,vals in [('Before',before),('After',after)]
            for feature_name,cols in [('Intensity only',[0]),('Intensity + texture',list(range(5)))]]


def point_plot(xy,sample,pair,title,xname,yname):
    fig=go.Figure()
    for c in pair:
        keep=sample['labels']==c
        custom=np.column_stack([np.full(keep.sum(),sample['z']),sample['rows'][keep],sample['cols'][keep],sample['voxel_ids'][keep],sample['labels'][keep]])
        fig.add_trace(go.Scattergl(x=xy[keep,0].tolist(),y=xy[keep,1].tolist(),mode='markers',name=LABELS[c],customdata=custom.tolist(),
                                  marker=dict(size=4,color=COLORS[c],opacity=.6),
                                  hovertemplate='x=%{x:.4g}, y=%{y:.4g}<br>Z=%{customdata[0]}, Y=%{customdata[1]}, X=%{customdata[2]}<br>voxel ID=%{customdata[3]}<extra>%{fullData.name}</extra>'))
    fig.update_layout(title=title,xaxis_title=xname,yaxis_title=yname,height=360,dragmode='lasso',clickmode='event+select',margin=dict(l=25,r=10,t=45,b=25))
    return fig


def ct_figure(image,title,roi,low,high,sample=None,selected=(),mask=None,pair=(),highlights=None):
    y0,y1,x0,x1=roi
    fig=go.Figure(go.Heatmap(z=image[y0:y1,x0:x1],x=np.arange(x0,x1),y=np.arange(y0,y1),
                               zmin=low,zmax=high,colorscale='Gray',showscale=False,
                               hovertemplate='source X=%{x}, Y=%{y}<br>value=%{z}<extra></extra>'))
    if mask is not None:
        for c in pair:
            overlay=np.where(mask[y0:y1,x0:x1]==c,1.,np.nan)
            fig.add_trace(go.Heatmap(z=overlay,x=np.arange(x0,x1),y=np.arange(y0,y1),colorscale=[[0,COLORS[c]],[1,COLORS[c]]],
                                    zmin=0,zmax=1,opacity=.4,showscale=False,name=LABELS[c],hoverinfo='skip'))
    if sample is not None and len(selected):
        keep=np.isin(sample['voxel_ids'],selected)
        fig.add_trace(go.Scattergl(x=sample['cols'][keep].tolist(),y=sample['rows'][keep].tolist(),mode='markers',name='Выбрано на графике',
                                  marker=dict(size=5,color='#ffe600',symbol='circle-open')))
    if highlights is not None:
        yy,xx=np.nonzero(highlights[y0:y1,x0:x1]);step=max(1,len(yy)//1500)
        fig.add_trace(go.Scattergl(x=xx[::step]+x0,y=yy[::step]+y0,mode='markers',name='Наибольшие изменения',
                                  marker=dict(size=2,color='cyan',opacity=.6)))
    fig.update_layout(title=title,height=350,yaxis=dict(scaleanchor='x'),xaxis_title='Source X index',yaxis_title='Source Y index',margin=dict(l=20,r=5,t=40,b=25))
    return fig


def render(x,y,affine,meta,case,source_path):
    st.subheader('Filtering & Feature Analysis')
    st.caption('Source intensity · source axial grid · без HU/mm · фильтры не меняют preprocessing CHD68. GT используется для анализа и оценки, а не как признак.')
    if y is None:st.info('Для этого эксперимента нужна экспертная маска.');return
    if not np.allclose(affine[:3,:3],np.diag(np.diag(affine[:3,:3]))) or np.any(np.diag(affine[:3,:3])<=0):
        st.error('Эксперимент требует осевой RAS source-сетки; oblique исходник нельзя называть аксиальным без reslicing.');return
    with st.expander('Z, ROI и параметры фильтра',expanded=False):
        col1,col2,col3=st.columns(3)
        with col1:
            z=st.slider('Source axial Z',0,x.shape[2]-1,x.shape[2]//2)
            pair_name=st.radio('Сравниваемые классы',list(PAIRS),horizontal=True)
        with col2:
            r0,r1=st.slider('ROI: source Y (включительно)',0,x.shape[1]-1,(0,x.shape[1]-1))
            c0,c1=st.slider('ROI: source X (включительно)',0,x.shape[0]-1,(0,x.shape[0]-1))
        with col3:
            method=st.selectbox('Метод обработки',['Baseline','Gaussian','Bilateral','CLAHE','Gradient Magnitude'])
            config={'name':method,'method':method}
            if method=='Gaussian':config['sigma']=st.select_slider('Gaussian sigma (grid units)',options=[.5,1.,2.],value=1.)
            elif method=='Bilateral':
                config['spatial_sigma']=st.slider('Bilateral spatial sigma (grid units)',.5,4.,1.,.5)
                config['intensity_sigma']=st.number_input('Bilateral intensity sigma (source intensity)',min_value=1.,value=100.)
            elif method=='CLAHE':
                config.update(clip_limit=st.number_input('CLAHE clip limit',.001,1.,.01,.001),
                              kernel_size=st.select_slider('CLAHE kernel size (pixels)',options=[8,16,32,64,128],value=32),input_range=[0.,4095.])
                st.caption('CLAHE: fixed [0,4095] → [0,1] → CLAHE → [0,4095]; без refit модели.')
            n=st.slider('Sampled voxels на класс',100,2000,600,100)
    source=np.array(x[:,:,z].T,np.float32,copy=True);mask=y[:,:,z].T
    pair=PAIRS[pair_name];roi=(r0,r1+1,c0,c1+1)
    spacing=nib.affines.voxel_sizes(affine)[[1,0]]
    after,before_maps,after_maps,provenance=calculate(source,config,tuple(spacing))
    sample=sample_voxels(mask,pair,case,z,x.shape,roi,n)
    if len(sample['labels'])<10 or any(np.count_nonzero(sample['labels']==c)<5 for c in pair):
        st.info('В выбранном срезе/ROI недостаточно вокселей обоих классов. Измените Z или ROI.');return
    values_before,values_after=matrix(before_maps,sample),matrix(after_maps,sample)
    st.caption(f"Стратифицировано {len(sample['labels'])} вокселей; seed={SEED}. Один набор source voxel IDs для всех фильтров и проекций. Фильтры/признаки считаются на полной плоскости до ROI, чтобы crop не создавал ложные границы.")
    with st.expander('Оси признаков и фиксированная шкала',expanded=False):
        a,b,c=st.columns(3)
        with a:
            xa=st.selectbox('Scatter X',FEATURES,index=0);ya=st.selectbox('Scatter Y',FEATURES,index=1)
        with b:scatter_source=st.radio('Scatter: версия признаков',['После обработки','Исходная'],horizontal=True)
        with c:
            display_low=st.number_input('Общая display scale: low',value=0.)
            display_high=st.number_input('Общая display scale: high',value=4095.)
            diff_high=st.number_input('Difference display scale: 0 …',min_value=.01,value=500.)
    values=values_after if scatter_source=='После обработки' else values_before
    scatter=point_plot(values[:,[FEATURES.index(xa),FEATURES.index(ya)]],sample,pair,'Выделение точек связано с КТ',xa,ya)
    left,right=st.columns([2,1])
    with right:
        event=st.plotly_chart(scatter,on_select='rerun',selection_mode=('points','box','lasso'),
                              key=f'feature_scatter_{case}_{z}_{pair_name}_{roi}',config={'displaylogo':False})
        selected=selected_voxel_ids(event,sample['voxel_ids'])
        st.caption('Нажмите на точку или выделите область рамкой/lasso. Жёлтые маркеры на КТ показывают выбранные sampled voxels.')
        if len(selected):
            keep=np.isin(sample['voxel_ids'],selected)
            st.dataframe([{'voxel_id':int(i),'source_xyz':xyz.tolist(),'world_RAS_grid_units':nib.affines.apply_affine(affine,xyz).tolist()}
                          for i,xyz in zip(sample['voxel_ids'][keep][:500],sample['xyz'][keep][:500])])
    if display_high<=display_low:st.error('Display high должен быть больше low');return
    difference=np.abs(after-source)
    top=float(np.percentile(difference[r0:r1+1,c0:c1+1],99))
    highlights=(difference>=top)&(difference>0)
    with left:
        for col,a,title in zip(st.columns(2),[source,after],['Исходная КТ','После обработки']):
            with col:st.plotly_chart(ct_figure(a,title,roi,display_low,display_high,sample,selected),config={'displaylogo':False})
        for col,fig in zip(st.columns(2),[ct_figure(difference,'|After − Before|',roi,0,diff_high,highlights=highlights),
                                        ct_figure(source,'Исходная КТ + экспертная маска',roi,display_low,display_high,mask=mask,pair=pair)]):
            with col:st.plotly_chart(fig,config={'displaylogo':False})
    st.caption(f'Изображения до/после имеют одну фиксированную шкалу [{display_low:g}, {display_high:g}]. Карта разности: [0,{diff_high:g}]. Cyan: верхний 1% изменений ROI, threshold={top:.4g}; display clipping не меняет данные.')
    if method=='Gradient Magnitude':
        st.plotly_chart(ct_figure(after_maps[:,:,1],'Отдельный Gradient Magnitude channel',roi,0,diff_high),config={'displaylogo':False})
    regions=fixed_regions(before_maps,mask,pair,roi)
    measures={'Before':measurements(before_maps,mask,pair,regions),
              'After':measurements(after_maps,mask,pair,regions,provenance['clipped_fraction'])}
    compact_keys=['homogeneous_local_variance_median','expert_interface_gradient_mean','contrast_abs_mean','fisher_ratio','histogram_overlap','clipped_fraction']
    st.dataframe([dict(stage=k,**{name:v.get(name) for name in compact_keys}) for k,v in measures.items()])
    st.caption(f"Фиксированные области: homogeneous={int(regions[2].sum())} voxels, interface={int(regions[1].sum())} voxels. Все измерения/параметры сохраняются в JSON.")
    with st.expander('Где измерены шум и границы'):
        st.plotly_chart(ct_figure(source,'Cyan: fixed homogeneous proxy regions',roi,display_low,display_high,highlights=regions[2]))
        st.plotly_chart(ct_figure(source,'Cyan: LV/MYO или RV/PA expert contact band',roi,display_low,display_high,highlights=regions[1]))
    st.caption('Шум: медиана локальной дисперсии в фиксированных эродированных участках с нижним квартилем baseline Local Std. Это proxy без бесшумного эталона. Граница: контакт двух выбранных GT-классов; если контакта нет, значение отсутствует. CLAHE может усиливать и границы, и локальные колебания.')
    projection_info={};embeddings={}
    kinds=['PCA']+(['UMAP'] if st.checkbox('Рассчитать UMAP (CPU; первый запуск компилирует numba)',value=False) else [])
    for kind in kinds:
        for col,name,vals in zip(st.columns(2),['Before','After'],[values_before,values_after]):
            embedding,info=projected(vals,sample['train'],kind);projection_info[kind+' '+name]=info;embeddings[kind+' '+name]=embedding
            with col:st.plotly_chart(point_plot(embedding,sample,pair,kind+' '+name,kind+' 1',kind+' 2'),config={'displaylogo':False})
    st.caption('StandardScaler и проекции fit только по 80% train sampled voxels; остальные transform. Это внутри одного train/development исследования, не независимая оценка. Отдельные UMAP до/после не имеют общей системы расстояний. Цвет — только GT-аннотация графика, labels не передаются PCA/UMAP.')
    scores=diagnostic_scores(values_before,values_after,sample['labels'],sample['train']) if st.checkbox('Диагностический k-NN: автоматически обновлять (k=15, voxel holdout 20%)',True) else []
    if scores:
        st.dataframe([{k:s[k] for k in ('stage','feature_set','macro_f1','train_samples','evaluation_samples')} for s in scores])
        selected_score=st.selectbox('Матрица ошибок k-NN',range(len(scores)),format_func=lambda i:scores[i]['stage']+' / '+scores[i]['feature_set'])
        score=scores[selected_score]
        st.plotly_chart(go.Figure(go.Heatmap(z=score['confusion_matrix'],x=[LABELS[c] for c in score['classes']],
                                            y=[LABELS[c] for c in score['classes']],texttemplate='%{z}',colorscale='Blues'))
                        .update_layout(xaxis_title='Predicted',yaxis_title='Expert',height=300))
    st.caption('Voxel holdout оптимистичен: соседние воксели зависимы. Frozen validation по release-case/group считается отдельной командой, со scaler только из train. Test не используется.')
    config_record={'case':case,'z_source':z,'roi_y0_y1_x0_x1_exclusive':list(roi),'pair':pair,'provenance':provenance,
                   'seed':SEED,'sampled_per_class_limit':n,'projection':projection_info,'measurements':measures,'diagnostic_knn':scores,
                   'display_scale':[display_low,display_high],'diff_scale':[0,diff_high],'sample_scope':'one source axial slice',
                   'source_intensity_not_HU':True,'physical_scale_verified':False}
    config_record.update(source_shape_xyz=list(x.shape),source_affine_ras=affine.tolist(),
                         split='frozen train; within-case train voxel fold for scaler and projections')
    if st.button('Сохранить эксперимент фильтрации и интерактивный отчёт'):
        output=Path(__file__).resolve().parents[1]/'outputs/filtering_feature_01/interactive'/uuid.uuid4().hex
        output.mkdir(parents=True)
        config_record['source_image_sha256']=fingerprint(source_path)
        (output/'config.json').write_text(json.dumps(config_record,ensure_ascii=False,indent=2),encoding='utf-8')
        np.save(output/'processed_plane.npy',after)
        np.save(output/'difference_plane.npy',difference)
        np.savez_compressed(output/'fixed_regions.npz',roi=regions[0],expert_interface=regions[1],homogeneous_proxy=regions[2])
        np.savez_compressed(output/'sampled_voxels.npz',**{k:v for k,v in sample.items() if isinstance(v,np.ndarray)},
                            features_before=values_before,features_after=values_after,**{k.replace(' ','_'):v for k,v in embeddings.items()})
        figures=[ct_figure(source,'Before',roi,display_low,display_high),ct_figure(after,'After',roi,display_low,display_high),
                 ct_figure(difference,'Absolute difference',roi,0,diff_high),scatter]
        for key,embedding in embeddings.items():figures.append(point_plot(embedding,sample,pair,key,'Dim 1','Dim 2'))
        (output/'linked_report.html').write_text('<meta charset="utf-8">'+''.join(f.to_html(full_html=False,include_plotlyjs=i==0) for i,f in enumerate(figures)),encoding='utf-8')
        st.success(str(output))
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in output.iterdir():archive.write(path,path.name)
        st.download_button('Скачать эксперимент ZIP',buffer.getvalue(),f'{case}_filtering.zip')
