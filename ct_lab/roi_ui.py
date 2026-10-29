"""Heart localization, matched ROI/full segmentation and native Heart3D surfaces."""
import base64
import io
import json
import numpy as np
import nibabel as nib
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from ct_lab.roi_core import *
from ct_lab.roi_data import case_box
from ct_lab.roi_localizer import fixed_box
from ct_lab.spatial_ui import VIEWER
from ct_lab.spatial_features import PLANES
from heart3d.labels import LABELS as HEART_LABELS
os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache')
COLORS={c:HEART_LABELS[c][2] for c in range(1,8)}


def png(plane,seg=None,classes=(),alpha=.35,error=False):
    gray=(np.clip(plane/4095,0,1)*255).astype(np.uint8);rgb=np.repeat(gray[:,:,None],3,2)
    palette={1:'#ff4d4f',2:'#ffb020',3:'#a855f7'} if error else COLORS
    if seg is not None:
        for c in classes:
            keep=seg==c;color=np.array(list(bytes.fromhex(palette[c][1:])));rgb[keep]=((1-alpha)*rgb[keep]+alpha*color).astype(np.uint8)
    stream=io.BytesIO();Image.fromarray(np.flipud(rgb)).save(stream,format='PNG');return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


@st.cache_resource(max_entries=8)
def predictions(path):return np.load(path,mmap_mode='r')


@st.cache_data(max_entries=16)
def mesh_data(path):
    with np.load(path) as f:return f['vertices'],f['faces']


@st.cache_data(max_entries=120)
def source_fingerprint(path,mtime_ns,size):return fingerprint(Path(path))


def wire(fig,box,affine,name,color):
    b=np.array(box,float)-.5;v=np.array([[b[i,0],b[j,1],b[k,2]] for i in (0,1) for j in (0,1) for k in (0,1)]);v=nib.affines.apply_affine(affine,v)
    edges=[(i,j) for i in range(8) for j in range(i+1,8) if (i^j) in (1,2,4)];x=[];y=[];z=[]
    for a,b in edges:x.extend([v[a,0],v[b,0],None]);y.extend([v[a,1],v[b,1],None]);z.extend([v[a,2],v[b,2],None])
    fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',line={'color':color,'width':4},name=name))


def render(volume,mask,affine,case,image_path,mask_path):
    st.subheader('Heart Localization & ROI Segmentation')
    st.caption('Source intensity не HU · расстояния/HD95 в индексной сетке, не мм · development validation многократно использовалась.')
    if not (OUT/'predicted_boxes.json').exists():st.info('Подготовка локализатора выполняется; обучение из интерфейса не запускается.');return
    plan=json.loads((OUT/'protocol.json').read_text());selection=json.loads((OUT/'selection.json').read_text());data=json.loads((OUT/'predicted_boxes.json').read_text())
    if case not in plan['train_cases']+plan['validation_cases']:st.info('Sealed test не открывается.');return
    st.caption(case+' · '+('train: диагностический пример, не независимая оценка' if case in plan['train_cases'] else 'development: повторно использованная validation, не финальная оценка'))
    # Cached study predictions may only be displayed with their actual source/GT.
    _,records=cohort()
    for path,field in ((image_path,'image_SHA256'),(mask_path,'mask_SHA256')):
        p=Path(path)
        if not p.is_file() or source_fingerprint(str(p),p.stat().st_mtime_ns,p.stat().st_size)!=records[case][field]:
            st.error('Файлы не совпадают с frozen ImageCHD68 case. Сохранённые bbox/маски этого исследования нельзя накладывать на другой CT или GT с тем же ID.');return
    target=cardiac_target(mask) if mask is not None else None;prediction=data[case];automatic=np.array(prediction['bbox']);fixed=fixed_box(selection['fixed_roi'],volume.shape)
    st.write(f'Localizer confidence: {prediction["confidence"]:.3f} · margin={selection["margin"]:.2f} bbox + 8 grid indices · fallback={prediction["fallback_fixed_ROI"]}')
    st.caption('Confidence — средняя вероятность coarse foreground, не калиброванная гарантия покрытия. Автоматический bbox получен только из CT.')
    oracle_allowed=case in plan['train_cases']
    if not oracle_allowed:oracle_allowed=st.checkbox('Oracle validation diagnostic — uses GT',value=False)
    if oracle_allowed:
        oracle=expand_box(bbox((target>0)&(target<8)),volume.shape,selection['margin'],8)
        if case in plan['validation_cases']:st.warning('Oracle diagnostic использует экспертный bbox для исследования потенциальной пользы ROI. Это отдельный диагностический результат.')
    choices=['Predicted ROI','Fixed train ROI']+(['Oracle GT ROI (diagnostic)'] if oracle_allowed else [])
    box_choice=st.multiselect('ROI boxes',choices,default=['Predicted ROI'],key=f'roi_boxes_{case}_{oracle_allowed}')
    boxes=[]
    for name,box,color in [('Predicted ROI',automatic,'#00ff9d'),('Fixed train ROI',fixed,'#ffb020')]+([('Oracle GT ROI (diagnostic)',oracle,'#c084fc')] if oracle_allowed else []):
        if name in box_choice:boxes.append({'name':name,'bbox':np.array(box).tolist(),'color':color})
    controls=st.columns(3)
    classes=controls[0].multiselect('ROI foreground classes',list(range(1,8)),default=list(range(1,8)),format_func=lambda c:LABELS[c])
    alpha=controls[1].slider('ROI mask opacity',0.,1.,.35)
    options=['Source','GT'];available={}
    for directory in (OUT/'evaluation'/'all_train').glob('*'):
        path=directory/case/'prediction.npy'
        if not path.exists() or not (directory/case/'scores.json').exists():continue
        if directory.name.startswith('C_oracle') and not oracle_allowed:continue
        options.append(directory.name);available[directory.name]=path
    for receipt in (OUT/'segmenters'/'all_train').glob('*/result.json'):
        name=receipt.parent.name
        if name.startswith('C_oracle') and not oracle_allowed:continue
        if name not in options:options.append(name)
        path=OUT/'interactive'/name/case/'prediction.npy'
        if name not in available and path.exists():available[name]=path
    for name in ('CHD68_v1','CHD68_v2'):
        if name not in options:options.append(name)
        path=OUT/'interactive'/name/case/'prediction.npy'
        if name not in available and path.exists():available[name]=path
    default='D_predicted_256' if 'D_predicted_256' in options else 'GT'
    method=controls[2].selectbox('ROI segmentation model',options,index=options.index(default),key=f'roi_model_{case}_{oracle_allowed}')
    pred=predictions(str(available[method])) if method in available else None
    if method not in ('Source','GT') and pred is None:
        st.info('Для этого CT готовой маски нет. Inference использует уже обученную модель и не запускает обучение.')
        if st.button('Выполнить inference выбранной модели'):
            from ct_lab.roi_evaluate import predict_with_box
            from ct_lab.roi_train import load_segmenter
            from heart3d.ml.infer_cardiac import load_recovered,prepare_volume,infer_volume,save_prediction
            from heart3d.ml.cardiac_data import confusion_counts,scores_from_confusion
            device='cuda' if torch.cuda.is_available() else 'cpu';source=nib.load(DEFAULT_DATA/f'{case}_image.nii.gz')
            with st.spinner('Восстановление полной исходной сетки…'):
                if method.startswith('CHD68'):
                    variant=method[-2:];network,_,_,pre=load_recovered(variant,checkpoint_path(variant),device)
                    prepared,t=prepare_volume(volume,pre);out=infer_volume(network,prepared,t,pre,1,device);box=np.array([[0,0,0],volume.shape]);t=transform(volume.shape,box,pre['input_size'])
                else:
                    network,saved=load_segmenter(OUT/'segmenters'/'all_train'/method/'best.pt',device);job=saved['job'];box=case_box(case,job['mode'],oracle_diagnostic=oracle_allowed)
                    out,t=predict_with_box(volume,affine,network,job,box,device,OUT/'interactive_core'/case/str(job['size']))
                directory=OUT/'interactive'/method/case;directory.mkdir(parents=True,exist_ok=True);np.save(directory/'prediction.npy',out);save_prediction(out,source,directory/'prediction.nii.gz')
                cm=confusion_counts(out,target);write(directory/'scores.json',{'case':case,'method':method,'transform':t,'coverage':coverage(target,box),**scores_from_confusion(cm),
                    'scope':'interactive exploratory result, not part of frozen study comparison','oracle_GT_ROI':method.startswith('C_oracle')})
                del network
                if device=='cuda':torch.cuda.empty_cache()
            st.rerun()
    error_mode=st.selectbox('ROI error view',['Segmentation','False positives','False negatives','Class confusion'])
    if error_mode!='Segmentation':st.caption('FP: background → foreground; FN: foreground → background; class confusion: неверный foreground-класс. Ignore исключён.')
    key=f'roi_xyz_{case}'
    if key not in st.session_state:st.session_state[key]=((automatic[0]+automatic[1]-1)//2).tolist()
    pending=st.session_state.pop('roi_pending',None)
    if pending and pending['case']==case:st.session_state[key]=pending['xyz']
    xyz=st.session_state[key];planes=[];spacing=np.diag(affine[:3,:3])
    for name,(a,b,c) in PLANES.items():
        image=np.take(volume,xyz[c],axis=c).T;gt=np.take(target,xyz[c],axis=c).T if target is not None else None
        seg=np.take(pred,xyz[c],axis=c).T if pred is not None else gt if method=='GT' else None;error=error_mode!='Segmentation' and pred is not None
        if error:
            valid=gt<8
            if error_mode=='False positives':seg=np.where(valid&(gt==0)&np.isin(seg,classes),1,0)
            elif error_mode=='False negatives':seg=np.where(valid&np.isin(gt,classes)&(seg==0),1,0)
            else:seg=np.where(valid&(gt>0)&(seg>0)&(gt!=seg)&(np.isin(gt,classes)|np.isin(seg,classes)),1,0)
        planes.append({'name':name,'axes':[a,b,c],'width':image.shape[1],'height':image.shape[0],'spacing':[float(spacing[a]),float(spacing[b])],
                       'source':png(image,seg,[1] if error else classes,alpha,error)})
    clicked=VIEWER(planes=planes,xyz=xyz,selected=[],boxes=boxes,diagnostics=[],key=f'roi_viewer_{case}',default=None)
    if clicked and clicked['nonce']!=st.session_state.get('roi_nonce'):
        st.session_state['roi_nonce']=clicked['nonce'];st.session_state['roi_pending']={'case':case,'xyz':clicked['xyz']};st.rerun()
    position=nib.affines.apply_affine(affine,xyz).tolist()
    st.caption(f'Source XYZ: {xyz} · NIfTI: {position}, единицы не подтверждены · Predicted ROI [lo, hi): {automatic.tolist()}')
    with st.expander('Full FOV / Oracle / Predicted ROI',expanded=False):
        for col,(name,box) in zip(st.columns(3),[('Full FOV',np.array([[0,0,0],volume.shape])),('Predicted ROI',automatic)]+([('Oracle GT ROI — diagnostic',oracle)] if oracle_allowed else [('Oracle disabled',None)])):
            if box is None:col.info('Для validation oracle включается только отдельным диагностическим переключателем.');continue
            if not box[0,2]<=xyz[2]<box[1,2]:col.info(name+': выбранный исходный срез вне ROI.');continue
            cropped=volume[slice(box[0,0],box[1,0]),slice(box[0,1],box[1,1]),xyz[2]].T
            col.image(png(cropped),caption=name)
    if pred is not None:
        scores=json.loads(available[method].with_name('scores.json').read_text());st.write(f'Native-grid foreground Dice: {scores["macro_foreground_Dice"]:.3f} · ROI coverage: {scores["coverage"]["foreground_coverage"]:.4f}')
        st.plotly_chart(go.Figure(go.Bar(x=list(scores['classes']),y=[v['Dice'] for v in scores['classes'].values()])).update_layout(title='Dice по классам · исходная сетка',height=260))
        topology=available[method].with_name('topology.json')
        if topology.exists():
            stats=json.loads(topology.read_text())
            with st.expander('HD95, false components и потеря разметки из-за ROI'):
                st.dataframe([{'class':LABELS[c],'HD95 grid':stats['topology'][str(c)]['HD95_index_units'],'components 6':stats['topology'][str(c)]['components_6connected'],
                               'false components':stats['topology'][str(c)]['false_components_no_GT_overlap'],'false volume voxels':stats['topology'][str(c)]['false_component_voxels'],
                               'largest component voxels':stats['topology'][str(c)]['largest_component_voxels'],'GT coverage':scores['coverage']['per_class_coverage'][str(c)]} for c in range(1,8)])
                st.caption('Малые компоненты не удалялись. Меньшее количество компонент само по себе не доказывает лучшую анатомию. HD95 — grid units, не mm.')
    with st.expander('3D: bbox и Heart3D surfaces',expanded=False):
        figure=go.Figure()
        for item in boxes:wire(figure,item['bbox'],affine,item['name'],item['color'])
        mesh_method='GT' if method in ('GT','Source') else method;index=OUT/'surfaces'/case/mesh_method/'index.json'
        if index.exists():
            surface_alpha=st.slider('ROI surface opacity',.05,1.,.5)
            for item in json.loads(index.read_text()):
                c=item['class']
                if c not in classes:continue
                vertices,faces=mesh_data(str(index.parent/item['display_npz']))
                figure.add_trace(go.Mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color=COLORS[c],opacity=surface_alpha,name=LABELS[c],showlegend=True))
        else:
            st.caption('Для GT/CHD68/B/C/D256 поверхности предрассчитаны в ct_1001 и ct_1027. Другую сохранённую маску можно построить ниже.')
            if (pred is not None or method in ('Source','GT')) and st.button('Построить Heart3D поверхности этой маски'):
                from ct_lab.roi_topology import export_meshes
                with st.spinner('Marching cubes с исходным affine…'):export_meshes(case,mesh_method,target if method in ('GT','Source') else pred)
                st.rerun()
        position=nib.affines.apply_affine(affine,xyz);figure.add_trace(go.Scatter3d(x=[position[0]],y=[position[1]],z=[position[2]],mode='markers',marker={'color':'#00e5ff','size':5},name='Selected voxel'))
        st.plotly_chart(figure.update_layout(height=480,scene={'aspectmode':'data'},title='RAS header coordinates · units unknown; selected native voxel'))
        st.caption('Поверхности построены существующим heart3d.surfaces. Сырые VTP сохраняют все компоненты; display mesh упрощён только для браузера и не участвует в метриках.')
    with st.expander('Training / CV / protocol'):
        path=OUT/'segmenters'/'all_train'/method/'history.json'
        if path.exists():
            h=json.loads(path.read_text());fig=go.Figure()
            for field in ('loss','train_augmented_grid_foreground_Dice'):fig.add_trace(go.Scatter(x=[r['epoch'] for r in h],y=[r[field] for r in h],name=field))
            st.plotly_chart(fig.update_layout(title='Train grid curves, не native validation quality',height=280))
        st.caption('48 train / 10 reused development / sealed test закрыт. Margin выбран по train-only OOF. B/C/D имеют одинаковые параметры и budget; oracle всегда явно помечен. Обучение из GUI не запускается.')
