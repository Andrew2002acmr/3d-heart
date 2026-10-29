"""CT Preprocessing Lab. Run with ct_lab/launch.py from any working directory."""
import io
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
import streamlit as st
import plotly.graph_objects as go

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ct_lab.core import (discover, load_pair, metadata, orthogonal_pair, stats, plane, window, normalize,
                         filter_plane, LABELS, mask_analysis, line_profile, crop_pad, resize_dimensions, fingerprint, cardiac_target)

st.set_page_config(page_title='CT Preprocessing Lab', layout='wide')
st.title('CT Preprocessing Lab · ImageCHD68')
st.caption('Исследовательский анализ · исходные файлы не изменяются · CPU')
DEFAULT_DATA = REPO.parent/'laboratories/data/chd68/pairs'
OUTPUT = REPO/'outputs/ct_preprocessing_lab'


@st.cache_data(max_entries=2, show_spinner='Чтение и проверка геометрии…')
def cached_pair(image_path, mask_path, image_mtime, mask_mtime):
    image, mask = load_pair({'image': image_path, 'mask': mask_path})
    target=np.asanyarray(mask.dataobj).astype(np.int16) if mask is not None else None
    meta=metadata(image)
    meta['source_mask_labels']=np.unique(target).tolist() if target is not None else []
    return (image.get_fdata(dtype=np.float32), image.affine, image.header.copy(),
            target, meta)


@st.cache_data(max_entries=1, hash_funcs={nib.nifti1.Nifti1Header: lambda h: h.binaryblock}, show_spinner='Построение ортогональной сетки RAS…')
def aligned(x, affine, header, y):
    image = nib.Nifti1Image(x, affine, header)
    mask = nib.Nifti1Image(y, affine, header) if y is not None else None
    image, mask = orthogonal_pair(image, mask)
    return image.get_fdata(dtype=np.float32), image.affine, np.asanyarray(mask.dataobj) if mask is not None else None


def show(fig, key=None):
    st.plotly_chart(fig, key=key,
                    config={'displaylogo': False, 'toImageButtonOptions': {'format': 'svg'}})


def image_figure(a, title, spacing=(1,1), mask=None, selected=(), alpha=.35):
    fig = go.Figure(go.Heatmap(z=a, x=np.arange(a.shape[1])*spacing[1],
                               y=np.arange(a.shape[0])*spacing[0], colorscale='Gray', showscale=False,
                               hovertemplate='u=%{x:.3f}, v=%{y:.3f}<br>intensity=%{z}<extra></extra>'))
    if mask is not None and selected:
        overlay = np.where(np.isin(mask, selected), mask, np.nan)
        fig.add_trace(go.Heatmap(z=overlay, x=np.arange(a.shape[1])*spacing[1], y=np.arange(a.shape[0])*spacing[0],
                                colorscale='Turbo', zmin=0, zmax=max(7,max(selected)), opacity=alpha, showscale=False))
    fig.update_layout(title=title, height=370, margin=dict(l=5,r=5,t=35,b=20),
                      yaxis=dict(scaleanchor='x'), xaxis_title='Grid distance; see declared units')
    return fig


def histogram(series, bins=100, bounds=None, density=False):
    fig = go.Figure()
    for name, a in series.items():
        a = np.asarray(a)
        a = a[np.isfinite(a)]
        if not a.size: continue
        counts, edges = np.histogram(a, bins=bins, range=bounds, density=density)
        fig.add_trace(go.Scatter(x=(edges[1:]+edges[:-1])/2, y=counts, mode='lines', name=name))
    fig.update_layout(xaxis_title='Intensity (source scale or selected transform)', yaxis_title='Density' if density else 'Voxel count', height=380)
    return fig


root = st.sidebar.text_input('Каталог NIfTI pairs', str(DEFAULT_DATA))
rows = discover(root) if Path(root).is_dir() else []
if not rows:
    st.info('Не найдены *_image.nii[.gz]. Укажите каталог реальных исследований; маски ожидаются как *_label.nii[.gz].')
    st.stop()
section = st.sidebar.radio('Раздел', ['Dataset Overview','CT Explorer','HU Analysis','Preprocessing',
                                     'Spatial Context','Mask Anatomy','Model Input Inspector','Model Diagnostics','Experiments','Filtering & Feature Analysis','Spatial Feature Explorer','Learnable Feature Explorer','Multiclass Feature Lab','Heart Localization & ROI Segmentation','Training Dynamics Lab'])
row = rows[st.sidebar.selectbox('Исследование', range(len(rows)), format_func=lambda i: rows[i]['id'])]
if section in ('Learnable Feature Explorer','Multiclass Feature Lab','Heart Localization & ROI Segmentation','Training Dynamics Lab'):
    frozen=json.loads((REPO/'configs/splits/cardiac_chd68_v1.json').read_text(encoding='utf-8'))
    if row['id'] not in frozen['partitions']['train']+frozen['partitions']['validation']:
        st.info('Исследовательский режим: sealed test и неизвестные split cases не открываются.');st.stop()
if section == 'Spatial Feature Explorer':
    frozen=json.loads((REPO/'configs/splits/cardiac_chd68_v1.json').read_text(encoding='utf-8'))
    if row['id'] not in frozen['partitions']['train']+frozen['partitions']['validation']:
        st.info('Spatial experiment: sealed test и неизвестные split cases не открываются. Выберите train/development ct_1001.');st.stop()
if section == 'Filtering & Feature Analysis':
    frozen=json.loads((REPO/'configs/splits/cardiac_chd68_v1.json').read_text(encoding='utf-8'))
    if row['id'] not in frozen['partitions']['train']:
        # Gate before cached_pair: this experiment never reads sealed test CT,
        # nor fits a projection/classifier to a selected validation case.
        if row['id'] in frozen['partitions']['validation']:
            st.subheader('Filtering & Feature Analysis · frozen validation')
            st.info('Для validation показываются результаты заранее зафиксированных конфигураций. Для интерактивного fit/изменения параметров выберите train/development ct_1001.')
            results_path=REPO/'outputs/filtering_feature_01/results.json'
            if results_path.exists():
                fixed=json.loads(results_path.read_text(encoding='utf-8'))
                st.dataframe([{'case':row['id'],'pair':s['pair'],'config':s['config'],'feature_set':s['feature_set'],
                               **s['case_metrics'][row['id']]} for s in fixed['scores']
                              if s.get('role')=='held-out frozen validation' and row['id'] in s.get('case_metrics',{})])
            st.caption('CT не загружается в этом режиме; train statistics, PCA/UMAP и k-NN не fit по validation. Source intensity, не HU.')
        else:
            st.info('Этот эксперимент доступен для frozen train/development. Sealed test и неизвестные split cases здесь не открываются. Выберите ct_1001.')
        st.stop()
mask_path = Path(row['mask'])
try:
    x, source_affine, header, y, meta = cached_pair(row['image'], row['mask'], Path(row['image']).stat().st_mtime_ns,
                                                  mask_path.stat().st_mtime_ns if mask_path.exists() else 0)
except Exception as exc:
    st.error(str(exc)); st.stop()
st.sidebar.write(f"{len(rows)} исследований · {meta['orientation']} · {meta['shape_xyz']}")
st.warning('HU и физический масштаб ImageCHD68 не подтверждены. Числа spacing из заголовка не считаются миллиметрами; исходные интенсивности не считаются HU.')

if section == 'Training Dynamics Lab':
    from ct_lab.dynamics_ui import render
    render(x,y,source_affine,row['id'],row['image'],row['mask']);st.stop()

if section == 'Heart Localization & ROI Segmentation':
    from ct_lab.roi_ui import render
    render(x,y,source_affine,row['id'],row['image'],row['mask']);st.stop()

if section == 'Multiclass Feature Lab':
    from ct_lab.multiclass_ui import render
    render(x,y,source_affine,row['id']);st.stop()

if section == 'Learnable Feature Explorer':
    from ct_lab.learnable_ui import render
    render(x,y,source_affine,row['id']);st.stop()

if section == 'Spatial Feature Explorer':
    from ct_lab.spatial_ui import render
    render(x,y,source_affine,row['id']);st.stop()

if section == 'Filtering & Feature Analysis':
    import ct_lab.launch  # optional feature dependencies for a running pre-existing server
    from ct_lab.filtering_ui import render
    render(x,y,source_affine,meta,row['id'],row['image'])
    st.stop()

if section == 'Dataset Overview':
    st.json(meta)
    st.dataframe([stats(x)], width='stretch')
    if y is not None:
        labels, counts = np.unique(y, return_counts=True)
        st.write('Уникальных меток:', len(labels))
        st.dataframe([{'label': int(l), 'name': LABELS.get(int(l), 'UNKNOWN'), 'voxels': int(n), 'fraction': n/y.size} for l,n in zip(labels, counts)])
    if (OUTPUT/'dataset.json').exists():
        dataset = json.loads((OUTPUT/'dataset.json').read_text(encoding='utf-8'))
        st.write({k:v for k,v in dataset.items() if k != 'records'})
        st.dataframe([{'id': r['id'], **r.get('image_statistics', {}), 'anomalies': ', '.join(r['anomalies'])} for r in dataset['records']])
        st.download_button('Скачать dataset.json', (OUTPUT/'dataset.json').read_bytes(), 'dataset.json')
    st.code(f'..\\repository\\.venv\\Scripts\\python.exe ct_lab\\launch.py batch --data "{root}" --output "{OUTPUT}"')
    st.caption('Пакетный анализ последователен, возобновляем; --components включает дорогие 2D/3D операции. Глобальные параметры обучения не вычисляются.')
    st.stop()

if section in ('Model Input Inspector', 'Model Diagnostics'):
    from ct_lab.model_inspector import inspect, predict_slice, diagnose_volume
    from heart3d.ml.infer_cardiac import protocol
    st.caption('Исходная сетка модели XYZ; это отдельный путь от ортогонального исследовательского viewer.')
    variant = st.selectbox('Модель', ['v1','v2'])
    center = st.slider('Центральный Z index исходной КТ', 0, x.shape[2]-1, x.shape[2]//2)
    configs = {v: protocol(v)[2] for v in ('v1','v2')}
    st.dataframe([{'variant':v, **p} for v,p in configs.items()])
    if st.button('Получить фактический preprocessing / тензор'):
        try:
            result = inspect(nib.Nifti1Image(x, source_affine, header), variant, center)
            st.session_state['inspection'] = (row['id'], variant, center, result)
        except Exception as exc: st.error(str(exc))
    saved = st.session_state.get('inspection')
    if saved and saved[:3] == (row['id'],variant,center):
        result = saved[3]
        st.write('Shape BCHW:', result['tensor_shape'], 'Source Z:', result['source_indices'], 'Edge replicated:', result['replicated_edge'])
        st.json(result['tensor_statistics']); st.dataframe(result['channel_statistics'])
        stage = st.selectbox('Стадия — в фактическом порядке', list(result['stages']))
        channel = st.slider('Канал', 0, 4, 2)
        show(image_figure(result['stages'][stage][channel], stage))
        show(histogram({name: a[channel] for name,a in result['stages'].items()}))
        st.caption(result['source']+'. До clipping выполняется resize. Identity orientation; отдельного mm resampling нет. Исследовательские фильтры сюда не подмешиваются.')
        st.download_button('Tensor .npy', (lambda b: (np.save(b, result['tensor'].numpy()), b.getvalue())[1])(io.BytesIO()), f'{row["id"]}_{variant}_input.npy')
        checkpoint = st.text_input('Checkpoint (проверяется исторический SHA256)', str(Path.home()/f'heart-model-recovery/chd68_{variant}_best.pt'))
        if st.button('CPU forward: захват входа и диагностика центрального среза'):
            try:
                probs, captured, entropy = predict_slice(result, variant, checkpoint, x.shape)
                st.session_state['slice_diagnostics']=(row['id'],variant,center,probs,entropy,list(captured.shape))
            except Exception as exc: st.error(str(exc))
        diagnostic=st.session_state.get('slice_diagnostics')
        if diagnostic and diagnostic[:3]==(row['id'],variant,center):
            probs,entropy,captured_shape=diagnostic[3:]
            prediction=probs.argmax(0)
            st.success(f'forward_pre_hook: вход совпал побитово, shape {captured_shape}')
            cols = st.columns(3)
            with cols[0]: show(image_figure(x[:,:,center].T, 'Source / prediction', mask=prediction.T, selected=list(range(1,8))))
            with cols[1]: show(image_figure(entropy.T, 'Entropy, natural log'))
            with cols[2]: show(image_figure(probs[st.selectbox('Probability class', list(LABELS))].T, 'Class probability'))
            if y is not None:
                from heart3d.ml.cardiac_data import confusion_counts, scores_from_confusion
                st.json(scores_from_confusion(confusion_counts(prediction, cardiac_target(y[:,:,center]))))
                st.caption('Source labels вне 0..7 исключены из метрик как ignore=255, согласно исходной оценке; в viewer они сохраняются.')
                show(image_figure(x[:,:,center].T, 'Expert mask', mask=y[:,:,center].T, selected=list(range(1,8))))
            st.dataframe([{'class': LABELS[c], 'components_2d_8': ndi.label(prediction==c, np.ones((3,3)))[1]} for c in range(1,8)])
            st.caption('Метрики относятся только к выбранному 2D срезу; это не Dice всего пациента и не оценка test. Полная 3D диагностика запускается отдельно.')
        if section == 'Model Diagnostics':
            components = st.checkbox('Полный 3D/2D component analysis (дорогая операция)',False)
            if st.button('Запустить полный CPU inference выбранного объёма'):
                try:
                    with st.spinner('Full-volume inference и восстановление исходной сетки…'):
                        image=nib.Nifti1Image(x,source_affine,header)
                        mask=nib.Nifti1Image(y,source_affine,header) if y is not None else None
                        pred,report=diagnose_volume(image,mask,variant,checkpoint,components)
                    st.session_state['volume_diagnostics']=(row['id'],variant,pred,report)
                except Exception as exc: st.error(str(exc))
            volume=st.session_state.get('volume_diagnostics')
            if volume and volume[:2]==(row['id'],variant):
                pred,report=volume[2:]
                st.json(report.get('metrics',{'Ground Truth':'absent; Dice not computed'}))
                show(image_figure(x[:,:,center].T,'Full-volume prediction: source Z',mask=pred[:,:,center].T,selected=list(range(1,8))))
                fig=go.Figure()
                for c in range(1,8):
                    fig.add_trace(go.Scatter(y=report['prediction_anatomy']['slice_area_grid_units2'][str(c)]['2'],name=f'{LABELS[c]} prediction')) if str(c) in report['prediction_anatomy']['slice_area_grid_units2'] else None
                    if 'expert_anatomy' in report and str(c) in report['expert_anatomy']['slice_area_grid_units2']:
                        fig.add_trace(go.Scatter(y=report['expert_anatomy']['slice_area_grid_units2'][str(c)]['2'],name=f'{LABELS[c]} expert',line_dash='dot'))
                fig.update_layout(xaxis_title='Source Z index (physical units unverified)',yaxis_title='Area in grid units²')
                show(fig)
                st.dataframe(report['prediction_anatomy']['classes'])
                if 'fragment_gt_comparison' in report:
                    st.dataframe(report['fragment_gt_comparison'])
                    st.caption('Largest=false — satellite component. GT overlap показывает метки, с которыми он пересекается; отдельный компонент не автоматически ошибка.')
                st.download_button('Full-volume diagnostics JSON',json.dumps(report,ensure_ascii=False,indent=2),'volume_diagnostics.json')
                if st.button('Сохранить prediction NIfTI в derived output'):
                    import uuid
                    from heart3d.ml.infer_cardiac import save_prediction
                    destination=OUTPUT/f'{row["id"]}_{variant}_{uuid.uuid4().hex}_prediction.nii.gz'
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    save_prediction(pred.astype(np.uint8),nib.Nifti1Image(x,source_affine,header),destination)
                    st.success(str(destination))
    st.stop()

ox, affine, oy = aligned(x, source_affine, header, y)
spacing = nib.affines.voxel_sizes(affine)
unit = meta['spatial_unit']
positions = [st.sidebar.slider(f'{name} index', 0, ox.shape[a]-1, ox.shape[a]//2, key=f'pos_{row["id"]}_{a}') for a,name in enumerate(['R / X','A / Y','S / Z'])]
world = nib.affines.apply_affine(affine, positions)
source_index = nib.affines.apply_affine(np.linalg.inv(source_affine), world)
st.caption(f'RAS coordinate {world.round(3).tolist()} ({unit}; scale unverified) · source XYZ {source_index.round(3).tolist()} · intensity {ox[tuple(positions)]:.5g}')
available_labels=sorted(set(LABELS) | set(map(int,meta['source_mask_labels']))) if oy is not None else []
selected = st.sidebar.multiselect('Метки overlay', available_labels, default=list(range(1,8)), format_func=lambda i:f'{i} {LABELS.get(i,"source extra / anatomy unverified")}') if oy is not None else []
alpha = st.sidebar.slider('Прозрачность маски', 0., 1., .35)
axis_name = st.sidebar.selectbox('Плоскость анализа', ['Axial','Coronal','Sagittal'])
axis = {'Axial':2, 'Coronal':1, 'Sagittal':0}[axis_name]
idx = positions[axis]
p = plane(ox, axis, idx)
mp = plane(oy, axis, idx) if oy is not None else None
other = [a for a in range(3) if a!=axis]
spacing2 = spacing[other][::-1]

if section == 'CT Explorer':
    for col, a, title in zip(st.columns(3), [2,1,0], ['Axial · R/A','Coronal · R/S','Sagittal · A/S']):
        with col:
            others = [i for i in range(3) if i != a]
            fig = image_figure(plane(ox,a,positions[a]), title, spacing[others][::-1],
                               plane(oy,a,positions[a]) if oy is not None else None, selected, alpha)
            fig.add_vline(x=positions[others[0]]*spacing[others[0]], line_color='cyan')
            fig.add_hline(y=positions[others[1]]*spacing[others[1]], line_color='cyan')
            show(fig)
    st.caption('Позиция общая для трёх проекций. Plotly: wheel/box zoom, pan, hover интенсивности; reset двойным нажатием. Геометрия сначала приведена к осевой RAS сетке.')

elif section == 'HU Analysis':
    scope = st.selectbox('Область распределения', ['Вся КТ','Текущий срез','ROI box','Heart foreground','Background','По классам'])
    bins = st.slider('Число интервалов', 10, 500, 100)
    xmin = st.number_input('X min', value=float(np.nanmin(ox))); xmax = st.number_input('X max', value=float(np.nanmax(ox)))
    if xmax <= xmin: st.error('X max должен быть больше X min'); st.stop()
    series = {}
    if scope == 'Вся КТ': series[row['id']] = ox
    elif scope == 'Текущий срез': series[row['id']+' slice'] = p
    elif scope == 'ROI box':
        roi = [st.slider(f'ROI {a}', 0, ox.shape[a]-1, (0,ox.shape[a]-1)) for a in range(3)]
        series['ROI (orthogonal grid)'] = ox[tuple(slice(lo,hi+1) for lo,hi in roi)]
    elif oy is None: st.info('Требуется геометрически корректная экспертная маска'); st.stop()
    elif scope == 'Heart foreground': series['Heart labels 1–7'] = ox[np.isin(oy, list(range(1,8)))]
    elif scope == 'Background': series['Background label 0'] = ox[oy==0]
    else:
        include_bg = st.checkbox('Включить фон', False)
        series = {LABELS.get(int(c),f'Unknown {c}'): ox[oy==c] for c in np.unique(oy) if c or include_bg}
    show(histogram(series,bins,(xmin,xmax),st.checkbox('Density', False)))
    st.dataframe([dict(region=k, **stats(v)) for k,v in series.items()])
    kind = st.selectbox('Сравнение распределений', ['None','Box','Violin'])
    if kind != 'None':
        fig = go.Figure()
        for name, a in series.items():
            a = a.ravel(); a = a[np.isfinite(a)]
            sample = a[np.linspace(0,len(a)-1,min(5000,len(a)),dtype=int)] if a.size else a
            fig.add_trace(go.Box(y=sample,name=name,boxpoints=False) if kind=='Box' else go.Violin(y=sample,name=name,points=False))
        show(fig); st.caption('Box/violin: до 5000 детерминированных значений на область; таблица и гистограммы используют все значения. Перекрытие распределений не доказывает разделимость классов.')
    comparisons = st.multiselect('Сравнить пациентов (исходная сетка, вся КТ)', [r['id'] for r in rows if r!=row], max_selections=3)
    if comparisons and st.button('Построить сравнение пациентов'):
        compare = {row['id']: x.ravel()[::max(1,x.size//100000)]}
        for r in rows:
            if r['id'] in comparisons:
                im,_ = load_pair(r); values=im.get_fdata(dtype=np.float32).ravel()
                compare[r['id']]=values[::max(1,values.size//100000)].copy()
        show(histogram(compare,bins,(xmin,xmax),True)); st.caption('Сравнение: равные по порядку объёмы выборок до ~100000/пациент; исходная сетка, вся КТ. Не используется для fitting.')
    joint = st.selectbox('2D histogram (текущий срез, вся плоскость)', ['None','Intensity × Gradient','Intensity × Local Variance'])
    if joint != 'None':
        feature = np.hypot(*np.gradient(p, *spacing2)) if joint.endswith('Gradient') else np.maximum(ndi.uniform_filter(p*p,5)-ndi.uniform_filter(p,5)**2,0)
        show(go.Figure(go.Histogram2d(x=p.ravel(),y=feature.ravel(),nbinsx=bins,nbinsy=bins)))

elif section in ('Preprocessing','Experiments'):
    st.caption('Экспериментальные преобразования не изменяют pipeline обученных моделей. Сдвиг входного распределения может ухудшить предсказания.')
    preset = st.selectbox('Стартовый исследовательский пресет (не клинический стандарт)', ['Source scale','Soft tissue hypothesis','Heart hypothesis','Vessel hypothesis'])
    presets = {'Source scale':(0.,2015.),'Soft tissue hypothesis':(-160.,240.),'Heart hypothesis':(-100.,500.),'Vessel hypothesis':(0.,800.)}
    st.caption('Пресеты с отрицательными границами применимы только после независимого подтверждения HU; здесь это числовой эксперимент на исходной шкале.')
    low = st.number_input('Нижняя граница', value=presets[preset][0], key='low_'+preset)
    high = st.number_input('Верхняя граница', value=presets[preset][1], key='high_'+preset)
    if st.checkbox('Управлять через center / width'):
        wc=st.number_input('Window center',value=(low+high)/2)
        ww=st.number_input('Window width',min_value=.001,value=max(.001,high-low))
        low,high=wc-ww/2,wc+ww/2
    st.write('Center:', (low+high)/2, 'Width:', high-low)
    method = st.selectbox('Normalization (по текущему срезу, если не saved train)', ['None','Min-Max','Z-score','Percentile','Fixed-window','Saved train window'])
    filt = st.selectbox('2D filter / исследовательская карта', ['None','Gaussian','Median','Bilateral','Non-local Means','CLAHE','Gradient Magnitude','Sobel','Laplacian','Canny'])
    strength = st.slider('Filter strength (pixels)', .5, 4., 1.)
    st.caption('Bilateral/NLM/CLAHE работают в [0,1]; Canny и карты границ — визуализация. Фильтры ограничены выбранной плоскостью.')
    try:
        clipped, fractions = window(p,low,high)
        filtered = filter_plane(clipped,filt,strength)
        after = normalize(filtered,method,low=0 if method=='Saved train window' else low, high=2015 if method=='Saved train window' else high)
    except ValueError as exc: st.error(str(exc)); st.stop()
    st.write(fractions)
    for col,a,title in zip(st.columns(2),[p,after],['Before','After']):
        with col: show(image_figure(a,title,spacing2,mp,selected,alpha))
    show(histogram({'Source':p,'Clipping':clipped,'Filter':filtered,'Normalization':after}))
    st.dataframe([dict(stage=k,**stats(v)) for k,v in {'Source':p,'Clipping':clipped,'Filter':filtered,'Normalization':after}.items()])
    multi=None
    if st.checkbox('Экспериментальное multi-window представление (3 канала; не вход текущей модели)'):
        bounds=[]
        for c,defaults in enumerate([(0.,1000.),(500.,1500.),(1000.,2015.)]):
            bounds.append([st.number_input(f'Channel {c} low',value=defaults[0]),st.number_input(f'Channel {c} high',value=defaults[1])])
        try:
            multi=np.stack([normalize(p,'Fixed-window',low=a,high=b) for a,b in bounds])
        except ValueError as exc:st.error(str(exc));st.stop()
        for col,c in zip(st.columns(3),range(3)):
            with col:show(image_figure(multi[c],f'Window {bounds[c]}',spacing2))
        st.dataframe([dict(channel=c,**stats(v)) for c,v in enumerate(multi)])
    xx=np.linspace(min(float(p.min()),low),max(float(p.max()),high),512)
    show(go.Figure(go.Scatter(x=xx,y=np.clip(xx,low,high),name='Clipping transfer')))
    show(go.Figure(go.Histogram2d(x=p.ravel(),y=after.ravel(),nbinsx=80,nbinsy=80)))
    if filt != 'None':
        before_gradient=np.hypot(*np.gradient((normalize(p,'Min-Max')+1)/2,*spacing2))
        after_gradient=np.hypot(*np.gradient((normalize(after,'Min-Max')+1)/2,*spacing2))
        st.write({'mean_gradient_before':float(before_gradient.mean()),'mean_gradient_after':float(after_gradient.mean()),
                  'edge_analysis':'Each image independently scaled to [0,1]; exploratory, not segmentation improvement'})
        with st.expander('Карты границ before / after'):
            for col,a,title in zip(st.columns(2),[before_gradient,after_gradient],['Before gradient','After gradient']):
                with col: show(image_figure(a,title,spacing2))
    with st.expander('Пространственные преобразования (по запросу)'):
        sp = [st.number_input(f'Target spacing {a} ({unit})', min_value=.1,value=float(spacing[a])) for a in range(3)]
        target_size = [st.number_input(f'Crop/pad size {a}',min_value=1,max_value=2048,value=int(ox.shape[a])) for a in range(3)]
        start = [st.number_input(f'Crop start {a} (ROI; negative = padding)', value=0, step=1) for a in range(3)]
        resize=st.checkbox('Изменить размер сетки с сохранением physical endpoints')
        resize_size=[st.number_input(f'Resize samples {a}',min_value=2,max_value=2048,value=max(2,int(ox.shape[a]))) for a in range(3)] if resize else None
        if st.button('Вычислить resampling + ROI crop/pad'):
            try:
                image=nib.Nifti1Image(ox,affine,header); mask=nib.Nifti1Image(oy,affine,header) if oy is not None else None
                res,seg=orthogonal_pair(image,mask,sp)
                if resize_size is not None: res,seg=resize_dimensions(res,resize_size,seg)
                res=crop_pad(res,start,target_size)
                if seg is not None: seg=crop_pad(seg,start,target_size)
                st.json({'before':metadata(image),'resampled_cropped':metadata(res),'image_interpolation':'linear','mask_interpolation':'nearest'})
                show(image_figure(np.asanyarray(res.dataobj)[:,:,res.shape[2]//2].T,'Resampled/cropped preview'))
                if seg is not None: st.write('Result mask labels:',np.unique(np.asanyarray(seg.dataobj)))
            except ValueError as exc: st.error(str(exc))
    config = {'case':row['id'],'source_image':row['image'],'axis':axis,'index':idx,'world_ras':world.tolist(),
              'geometry':meta,'sequence':['clip','filter2d','normalize'],'low':low,'high':high,'normalization':method,
              'filter':filt,'strength':strength,'scope':'one orthogonal slice','statistics':stats(after)}
    config['software']={'numpy':np.__version__,'nibabel':nib.__version__,'streamlit':st.__version__,'pipeline':'ct_lab/core.py'}
    if multi is not None: config['multi_window_bounds']=bounds
    name=st.text_input('Название эксперимента','Baseline preprocessing')
    config['name']=name
    if st.button('Сохранить в сравнение'):
        st.session_state.setdefault('experiments',[]).append({'config':config,'image':after.copy()})
        import uuid
        directory=OUTPUT/'experiments'/uuid.uuid4().hex
        directory.mkdir(parents=True)
        config['source_image_sha256']=fingerprint(row['image'])
        (directory/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        np.save(directory/'processed.npy',after)
        (directory/'report.html').write_text(histogram({'Before':p,'After':after}).to_html(include_plotlyjs=True),encoding='utf-8')
        st.success(f'Сохранено: {directory}')
    saved=st.session_state.get('experiments',[])
    st.dataframe([s['config'] for s in saved])
    if saved:
        comparable=[s for s in saved if s['config']['case']==row['id'] and s['config']['axis']==axis and s['config']['index']==idx]
        show(histogram({f'{i}: '+s['config']['name']:s['image'] for i,s in enumerate(comparable)}))
        st.caption('На общем графике только эксперименты того же пациента и той же плоскости/позиции.')
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('experiment.json',json.dumps(config,ensure_ascii=False,indent=2))
        if multi is not None:
            b=io.BytesIO();np.save(b,multi);archive.writestr('multi_window.npy',b.getvalue())
        archive.writestr('report.html',histogram({'Before':p,'After':after}).to_html(include_plotlyjs=True))
        array_buffer=io.BytesIO();np.save(array_buffer,after);archive.writestr('processed.npy',array_buffer.getvalue())
        archive.writestr('comparison.json',json.dumps([s['config'] for s in saved],ensure_ascii=False,indent=2))
        # Matplotlib export works on CPU without a browser or external Chrome.
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].imshow(p,cmap='gray',origin='lower');axes[0].set_title('Before')
        axes[1].imshow(after,cmap='gray',origin='lower');axes[1].set_title('After')
        for ext in ('png','svg'):
            b=io.BytesIO();fig.savefig(b,format=ext,bbox_inches='tight');archive.writestr('before_after.'+ext,b.getvalue())
        plt.close(fig)
        fig,ax=plt.subplots();ax.hist(p.ravel(),bins=100,histtype='step',label='Before');ax.hist(after.ravel(),bins=100,histtype='step',label='After');ax.legend()
        for ext in ('png','svg'):
            b=io.BytesIO();fig.savefig(b,format=ext);archive.writestr('histogram.'+ext,b.getvalue())
        plt.close(fig)
        archive.writestr('statistics.csv','stage,count,mean,std,min,max\n'+'\n'.join(','.join(map(str,[k,*[stats(v)[s] for s in ('count','mean','std','min','max')]])) for k,v in {'before':p,'after':after}.items()))
    st.download_button('Экспорт эксперимента ZIP · HTML / JSON / CSV / NPY / PNG / SVG',buffer.getvalue(),f'{row["id"]}_experiment.zip')

elif section == 'Spatial Context':
    st.write(f'Шаг между плоскостями: {spacing[axis]:g} {unit}; миллиметры не подтверждены')
    columns=st.columns(4)
    context_stats=[]
    for j,offset in enumerate(range(-3,4)):
        z=idx+offset
        with columns[j%4]:
            if 0<=z<ox.shape[axis]:
                context_stats.append(dict(offset=offset,distance_grid_units=offset*spacing[axis],**stats(plane(ox,axis,z))))
                show(image_figure(plane(ox,axis,z),f'{offset:+d} · coordinate {affine[axis,3]+z*spacing[axis]:.3f}',spacing2,
                                  plane(oy,axis,z) if oy is not None else None,selected,alpha))
            else: st.info(f'{offset:+d}: вне объёма; без подмены крайним срезом')
    st.dataframe(context_stats)
    difference_offset=st.selectbox('Абсолютная разность относительно центра',[-3,-2,-1,1,2,3],index=3)
    if 0<=idx+difference_offset<ox.shape[axis]:
        show(image_figure(np.abs(plane(ox,axis,idx+difference_offset)-p),f'Absolute difference: {difference_offset:+d} vs center',spacing2))
    profile=np.take(ox,positions[other[0]],axis=other[0])
    values=np.array([ox[tuple(i if a==axis else positions[a] for a in range(3))] for i in range(ox.shape[axis])])
    show(go.Figure(go.Scatter(x=affine[axis,3]+np.arange(len(values))*spacing[axis],y=values,name='Crosshair intensity along axis')))
    st.write('Линия на выбранном срезе (row / column endpoints)')
    p0=[st.number_input('Start row',0,p.shape[0]-1,0),st.number_input('Start column',0,p.shape[1]-1,0)]
    p1=[st.number_input('End row',0,p.shape[0]-1,p.shape[0]-1),st.number_input('End column',0,p.shape[1]-1,p.shape[1]-1)]
    sigma=st.slider('Gaussian comparison sigma (pixels)',.1,4.,1.)
    line=line_profile(p,ndi.gaussian_filter(p,sigma),mp,p0,p1,spacing2)
    location=image_figure(p,'Line location',spacing2,mp,selected,alpha)
    location.add_trace(go.Scatter(x=np.array([p0[1],p1[1]])*spacing2[1],y=np.array([p0[0],p1[0]])*spacing2[0],mode='lines+markers',line_color='cyan'))
    show(location)
    fig=go.Figure()
    for name in ('original','processed','gradient'): fig.add_trace(go.Scatter(x=line['distance'],y=line[name],name=name))
    for i in np.flatnonzero(np.diff(line['label'])!=0): fig.add_vline(x=line['distance'][i],line_dash='dot')
    show(fig)
    st.caption('Расстояние — в единицах сетки; пунктир — переходы экспертных меток. Gaussian comparison использует текущую плоскость.')

elif section == 'Mask Anatomy':
    if oy is None: st.info('Нет экспертной маски'); st.stop()
    st.caption('26 соседей в 3D; 8 соседей в 2D. Объёмы и площади в единицах заголовка, не в подтверждённых мм³/мм².')
    labels,counts=np.unique(oy,return_counts=True)
    show(go.Figure(go.Bar(x=[LABELS.get(int(c),str(c)) for c in labels],y=counts,name='Class voxels')))
    if st.button('Рассчитать 3D/2D связность и геометрию (объёмная операция)'):
        with st.spinner('Анализ компонентов…'):
            st.session_state['anatomy']=(row['id'],mask_analysis(oy,affine,unit))
    saved=st.session_state.get('anatomy')
    if saved and saved[0]==row['id']:
        result=saved[1];st.dataframe(result['classes'])
        fig=go.Figure()
        for c in selected:
            area=result['slice_area_grid_units2'].get(str(c),{}).get(str(axis),[])
            fig.add_trace(go.Scatter(x=affine[axis,3]+np.arange(len(area))*spacing[axis],y=area,name=LABELS.get(c,f'Source extra {c}')))
        show(fig)
        cls=st.selectbox('Component sizes class', [r['label'] for r in result['classes'] if r['label']])
        details=next(r for r in result['classes'] if r['label']==cls)
        st.json(details)
        show(go.Figure(go.Bar(y=details.get('component_sizes_voxels',[]),name='3D component size in voxels')))
        area=np.asarray(result['slice_area_grid_units2'][str(cls)][str(axis)])
        show(go.Figure(go.Scatter(y=np.diff(area),name='Adjacent slice area change')))
        st.download_button('Mask anatomy JSON',json.dumps(result,ensure_ascii=False,indent=2),'mask_anatomy.json')
