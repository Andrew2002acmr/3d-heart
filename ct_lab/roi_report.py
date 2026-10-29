"""Measured localization/ROI study, case statistics, geometry audit and compact package."""
import csv
import importlib.metadata
import json
import platform
import zipfile
import xml.etree.ElementTree as ET
import nibabel as nib
import numpy as np
import ct_lab.launch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ct_lab.roi_core import *


def table(headers,rows):
    def fmt(v):
        if isinstance(v,(list,tuple)):return '['+', '.join(fmt(x) for x in v)+']'
        return f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(fmt,row))+' |' for row in rows]])


def csv_file(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def bootstrap(values):
    values=np.asarray(values,float);rng=np.random.default_rng(SEED);resampled=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
    return {'mean':float(values.mean()),'case_SD':float(values.std()),'case_CI95':np.quantile(resampled,[.025,.975]).tolist(),'cases':len(values)}


def run():
    plan,records=freeze();geometry=json.loads((OUT/'train_geometry.json').read_text());local=json.loads((OUT/'localization_results.json').read_text())
    ui=json.loads((OUT/'ui_verification.json').read_text());browser=json.loads((OUT/'browser_verification.json').read_text())
    assert ui and all(not c['exceptions'] for c in ui) and any(c['check']=='Mismatched source/GT rejected before cached prediction access' for c in ui)
    assert not browser['page_errors'] and len(browser['plane_clicks'])==3 and all(c['expected']==c['actual'] for c in browser['plane_clicks']) and browser['Heart3D_meshes']
    suites=list(ET.parse(OUT/'tests.xml').getroot().iter('testsuite'));test_count=sum(int(s.get('tests',0)) for s in suites)
    assert test_count>=69 and all(int(s.get('errors',0))==0 and int(s.get('failures',0))==0 for s in suites)
    learned=json.loads((OUT/'segmentation_results.json').read_text());baseline=json.loads((OUT/'baseline_results.json').read_text());topology=json.loads((OUT/'topology_results.json').read_text())
    resampling=json.loads((OUT/'resolution_roundtrip.json').read_text());selection=json.loads((OUT/'segmentation_CV_selection.json').read_text());smoke=json.loads((OUT/'smoke.json').read_text())
    methods={};training={}
    for item in learned['main']:
        name=item['training']['job']['name'];methods[name]=item['evaluation']['records'];training[name]=item['training']
    for variant in ('CHD68_v1','CHD68_v2'):methods[variant]=[r for r in baseline if r['method']==variant]
    topo_lookup={(r['method'],r['case']):r for r in topology}
    for name,rows in methods.items():
        for r in rows:
            measured=topo_lookup[name,r['case']]
            assert abs(measured['macro_foreground_Dice']-r['macro_foreground_Dice'])<1e-12
            if 'confusion_matrix' in r:assert measured['confusion_matrix']==r['confusion_matrix']
    summaries=[];pairs=[]
    for name,rows in methods.items():
        val=[r for r in rows if r['case'] in plan['validation_cases']];values=[r['macro_foreground_Dice'] for r in val]
        assert len(val)==10 and set(r['case'] for r in val)==set(plan['validation_cases']) and 'ct_1001' not in plan['validation_cases']
        summaries.append({'method':name,'role':'reused development validation','case_ids':[r['case'] for r in val],**bootstrap(values),'train_ct1001_Dice':next(r['macro_foreground_Dice'] for r in rows if r['case']=='ct_1001'),
                          'per_class_Dice':{LABELS[c]:float(np.mean([r['classes'][LABELS[c]]['Dice'] for r in val if r['classes'][LABELS[c]]['target_voxels']>0])) for c in range(1,8)}})
    for a,b in [('D_predicted_256','B_full_256'),('D_predicted_384','B_full_384'),('C_oracle_256','D_predicted_256'),('D_core_256','D_predicted_256'),('B_full_384','B_full_256'),('D_predicted_384','D_predicted_256'),('C_oracle_256','B_full_256')]:
        left={r['case']:r['macro_foreground_Dice'] for r in methods[a]};right={r['case']:r['macro_foreground_Dice'] for r in methods[b]}
        diff={c:left[c]-right[c] for c in plan['validation_cases']};pairs.append({'comparison':a+' − '+b,**bootstrap(list(diff.values())),'paired_case_deltas':diff})
    write(OUT/'study_summary.json',{'methods':summaries,'paired':pairs,'primary_train_CV_selection':selection})
    charts=[]
    def chart(name,fig,explanation):
        directory=OUT/'charts';directory.mkdir(exist_ok=True);fig.update_layout(template='plotly_white')
        (directory/(name+'.html')).write_text('<meta charset="utf-8"><p>'+explanation+'</p>'+fig.to_html(full_html=False,include_plotlyjs=True),encoding='utf-8');charts.append((name,explanation))
    centers=np.array([r['normalized_center'] for r in geometry]);ratios=[r['bbox_to_FOV'] for r in geometry]
    chart('train_heart_centers',go.Figure(go.Scatter3d(x=centers[:,0],y=centers[:,1],z=centers[:,2],mode='markers',text=[r['case'] for r in geometry],marker={'color':ratios,'colorscale':'Viridis','size':5,'showscale':True})).update_layout(title='48 train: normalized foreground centers',scene={'xaxis_title':'X/FOV','yaxis_title':'Y/FOV','zaxis_title':'Z/FOV','aspectmode':'cube'}), 'Центр масс valid foreground, не bbox center. Цвет — доля GT-bbox от полного FOV. Только48 train; координаты нормализованы по shape, не мм.')
    fig=go.Figure()
    from ct_lab.roi_ui import wire
    for r in geometry:wire(fig,np.array(r['normalized_bbox'])+.5,np.eye(4),r['case'],'rgba(40,100,180,.25)')
    chart('all_train_boxes_normalized',fig.update_layout(title='48 GT bounding boxes in one normalized FOV',showlegend=False,scene={'aspectmode':'cube'}),'Все train GT-bbox в общей нормализованной системе координат. Это исследовательская визуализация, не автоматические ROI.')
    fig=make_subplots(rows=1,cols=3,subplot_titles=['X','Y','Z'])
    for a in range(3):
        fig.add_trace(go.Histogram(x=[r['GT_bbox_size'][a] for r in geometry],name='GT bbox',showlegend=a==0),1,a+1)
        fig.add_trace(go.Histogram(x=[r['shape_xyz'][a] for r in geometry],name='Source FOV',showlegend=a==0),1,a+1)
    chart('train_dimensions',fig.update_layout(barmode='overlay',title='Размеры FOV и GT bbox · grid indices'),'Исходные объёмы и bbox различаются по форме. Сравниваются количества вокселей вдоль осей, не физические длины.')
    chart('bbox_FOV_fraction',go.Figure(go.Histogram(x=ratios)).update_layout(title='Train GT bbox / full FOV volume'), 'Минимальный axis-aligned bbox всех классов1–7. Внутри него остаётся фон; bbox volume не равен foreground volume.')
    fig=go.Figure()
    for role in ('train','validation'):
        for method in ('fixed','predicted'):
            rows=[r for r in local['metrics'] if r['role']==role and r['method']==method]
            fig.add_trace(go.Scatter(x=[r['ROI_to_full_FOV'] for r in rows],y=[r['foreground_coverage'] for r in rows],mode='markers',text=[r['case'] for r in rows],name=role+'/'+method))
    chart('coverage_vs_ROI_volume',fig.update_layout(title='Foreground coverage vs ROI/FOV',xaxis_title='ROI/FOV',yaxis_title='Foreground coverage'),'Coverage — главная метрика. Более крупная ROI может снижать bbox IoU и при этом лучше сохранять разметку; margin выбран только по train OOF.')
    oof=json.loads((OUT/'localization_OOF.json').read_text());margin_rows=[]
    for margin in plan['localizer']['margin_candidates']:
        rows=[r['metrics'] for r in oof if r['margin']==margin]
        margin_rows.append({'margin':margin,'OOF_mean_coverage':float(np.mean([r['foreground_coverage'] for r in rows])),'OOF_worst_case_coverage':min(r['foreground_coverage'] for r in rows),'OOF_mean_ROI_FOV':float(np.mean([r['ROI_to_full_FOV'] for r in rows]))})
    chart('train_OOF_margin',go.Figure([go.Scatter(x=[r['margin'] for r in margin_rows],y=[r['OOF_mean_coverage'] for r in margin_rows],mode='lines+markers',name='Mean coverage'),go.Scatter(x=[r['margin'] for r in margin_rows],y=[r['OOF_worst_case_coverage'] for r in margin_rows],mode='lines+markers',name='Worst case coverage')]).update_layout(title='Margin и coverage: только train OOF',xaxis_title='Margin fraction + 8 native indices'), 'Margin выбран до development оценки. Критерий выбора основан на среднем coverage и минимальном среднем по классам; это не гарантия худшего случая.')
    valpred=[r for r in local['metrics'] if r['role']=='validation' and r['method']=='predicted']
    chart('localization_class_coverage',go.Figure(go.Heatmap(z=[[r['per_class_coverage'][str(c)] for c in range(1,8)] for r in valpred],x=LABELS[1:],y=[r['case'] for r in valpred],zmin=0,zmax=1,colorscale='Viridis',texttemplate='%{z:.3f}')).update_layout(title='Development: coverage per structure'), 'GT используется после автоматического предсказания для диагностики покрытия. Никакой коррекции validation ROI по этой таблице не выполнялось.')
    fig=go.Figure()
    for method in ('fixed','predicted'):
        rows=[r for r in local['metrics'] if r['role']=='validation' and r['method']==method]
        fig.add_trace(go.Scatter(x=[r['bbox_IoU'] for r in rows],y=[r['foreground_coverage'] for r in rows],mode='markers',text=[r['case'] for r in rows],name=method,marker={'size':9}))
    chart('IoU_vs_coverage',fig.update_layout(title='BBox IoU vs target coverage'),'IoU не заменяет coverage: включение контекстного margin меняет IoU даже при полном сохранении foreground.')
    fig=go.Figure()
    for case in plan['resize_diagnostic_cases']:
        rows=[r for r in resampling if r['case']==case];fig.add_trace(go.Bar(x=[r['mode']+' '+str(r['size']) for r in rows],y=[r['roundtrip_fg_Dice_mean'] for r in rows],name=case))
    chart('resampling_mask_only_Dice',fig.update_layout(title='GT nearest resize/inverse: geometry-only Dice'),'Ошибки только от преобразования масок, без нейросети. Oracle здесь диагностический train bbox +20%+8 indices; main C использует выбранный train-only margin.')
    fig=go.Figure()
    for case in plan['resize_diagnostic_cases']:
        rows=[r for r in resampling if r['case']==case];fig.add_trace(go.Scatter(x=[r['mode']+' '+str(r['size']) for r in rows],y=[r['native_grid_gradient_preservation_ratio'] for r in rows],mode='lines+markers',name=case))
    chart('native_gradient_preservation',fig.update_layout(title='Gradient preservation after image resize/inverse'),'Градиенты сравниваются после обратного восстановления на одинаковой исходной области каждой строки; разные FOV содержат разные ткани, поэтому raw gradient levels между ROI/full напрямую не сравниваются.')
    fig=go.Figure()
    for item in learned['CV']:
        result=item['training'];fig.add_trace(go.Bar(x=[result['context']],y=[item['evaluation']['mean_case_foreground_Dice']],name=result['job']['name'],legendgroup=result['job']['name'],showlegend=result['context']=='fold0'))
    chart('train_only_group_CV',fig.update_layout(title='Train-only group CV: B vs D256'),'Три holdout groups внутри48train, source-grid Dice. CV budget extended symmetrically6→18 epochs after train-only background-only pilot; main jobs unchanged. Не nested unbiased evaluation: localizer best epoch использует train-fold holdout.')
    chart('native_Dice_methods',go.Figure(go.Bar(x=[r['method'] for r in summaries],y=[r['mean'] for r in summaries],error_y={'type':'data','array':[r['case_SD'] for r in summaries]},marker_color=['#b58adb' if r['method'].startswith('C_oracle') else '#2c9e94' if r['method'].startswith('CHD68') else '#4c78a8' for r in summaries])).update_layout(title='Development native mean foreground Dice'), 'Среднее по10reused development cases; error bars — case SD, не voxel confidence interval. Oracle отдельно помечен; CHD68 historical budget differs, causal matched comparison — new B/C/D.')
    fig=go.Figure()
    for name,rows in methods.items():
        vals={r['case']:r['macro_foreground_Dice'] for r in rows};fig.add_trace(go.Scatter(x=plan['validation_cases'],y=[vals[c] for c in plan['validation_cases']],mode='lines+markers',name=name))
    chart('Dice_by_case',fig.update_layout(title='Native Dice по development cases'),'Показывает межслучайный разброс, который нельзя заменить числом срезов или sampled voxels.')
    chart('Dice_by_class',go.Figure(go.Heatmap(z=[[r['per_class_Dice'][LABELS[c]] for c in range(1,8)] for r in summaries],x=LABELS[1:],y=[r['method'] for r in summaries],zmin=0,zmax=1,colorscale='Viridis',texttemplate='%{z:.3f}')).update_layout(title='Native Dice по foreground классам'),'Среднее только по случаям, где данный GT класс присутствует; false positives для отсутствующих классов сохранены отдельно.')
    fig=go.Figure()
    for p in pairs:fig.add_trace(go.Scatter(x=list(p['paired_case_deltas']),y=list(p['paired_case_deltas'].values()),mode='lines+markers',name=p['comparison']))
    chart('paired_Dice_differences',fig.update_layout(title='Парные изменения native Dice'),'Одни и те же10cases. Bootstrap CI вычисляется по cases, не по voxels; reused development и multiple comparisons ограничивают выводы.')
    fig=go.Figure()
    for size in (256,384):
        full={r['case']:r for r in methods[f'B_full_{size}']};roi={r['case']:r for r in methods[f'D_predicted_{size}']}
        cases=plan['validation_cases'];gains=[]
        for case in cases:
            a=full[case]['transform'];b=roi[case]['transform'];fa=np.array(a['resize']['resized_xy'])/np.array(a['source_shape_xyz'][:2]);rb=np.array(b['resize']['resized_xy'])/(np.array(b['bbox_xyz'])[1,:2]-np.array(b['bbox_xyz'])[0,:2])
            gains.append(float(np.sqrt(np.prod(rb/fa))))
        fig.add_trace(go.Scatter(x=gains,y=[roi[c]['macro_foreground_Dice']-full[c]['macro_foreground_Dice'] for c in cases],mode='markers',name=str(size),text=cases,marker={'size':9}))
    chart('XY_scale_vs_Dice_change',fig.update_layout(title='Изменение XY-масштаба и Dice: ROI против Full',xaxis_title='Отношение XY pixels/source index (geometric mean)',yaxis_title='Парное Δ Dice'), 'Описательная связь на десяти случаях. ROI одновременно меняет контекст и ограничивает область предсказания; график не доказывает причинную роль детализации и не используется для выбора параметров.')
    for measure,title in [('false_components_no_GT_overlap','False component count'),('false_component_voxels','False component volume (voxels)'),('largest_component_voxels','Largest component size')]:
        fig=go.Figure()
        for name in methods:
            rows=[r for r in topology if r['method']==name and r['case'] in plan['validation_cases']]
            fig.add_trace(go.Bar(x=LABELS[1:],y=[float(np.mean([r['topology'][str(c)][measure] for r in rows])) for c in range(1,8)],name=name))
        chart(measure,fig.update_layout(title=title),'6-connectivity, source valid grid, без удаления малых компонент. False component не пересекает GT своего класса; это не обязательно только фоновый false positive. Меньшее число компонент не доказывает лучшую анатомию.')
    fig=go.Figure()
    for name in methods:
        values=[[r['topology'][str(c)]['HD95_index_units'] for r in topology if r['method']==name and r['case'] in plan['validation_cases'] and r['topology'][str(c)]['HD95_index_units'] is not None] for c in range(1,8)]
        fig.add_trace(go.Bar(x=LABELS[1:],y=[float(np.mean(v)) if v else None for v in values],name=name))
    chart('HD95_by_class',fig.update_layout(title='HD95 по классам · единицы сетки, не мм'),'Среднее по случаям с двумя непустыми поверхностями. Пропуски поверхности не превращаются в нулевой HD95: значения None сохранены в таблицах; оценивать вместе с recall/Dice.')
    for field in ('loss','train_augmented_grid_foreground_Dice'):
        fig=go.Figure()
        for name,result in training.items():
            h=json.loads((OUT/result['checkpoint']).with_name('history.json').read_text());fig.add_trace(go.Scatter(x=[r['epoch'] for r in h],y=[r[field] for r in h],name=name))
        chart('training_'+field,fig.update_layout(title=field+' / epoch'), 'Это train augmented/resampled grid. Native development Dice считается отдельно после полного восстановления; best.pt у main — fixed last epoch, не outer-selected.')
    # Real source images, resampling differences, and source-grid predictions.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from heart3d.labels import LABELS as HL
    cmap=ListedColormap(['#000000',*[HL[c][2] for c in range(1,8)]])
    for case in plan['resize_diagnostic_cases']:
        with np.load(OUT/'resolution'/f'{case}.npz') as f:
            keys=['Full_FOV_256','Full_FOV_384','Oracle_ROI_256','Oracle_ROI_384'];fig,axes=plt.subplots(2,5,figsize=(18,9))
            axes[0,0].imshow(f['source'].T,cmap='gray',origin='lower',vmin=0,vmax=4095);axes[0,0].set_title('Исходная КТ');axes[1,0].imshow(np.ma.masked_where(f['GT'].T>7,f['GT'].T),cmap=cmap,vmin=0,vmax=7,origin='lower');axes[1,0].set_title('GT')
            for i,key in enumerate(keys,1):
                axes[0,i].imshow(f[key+'_image'],cmap='gray',origin='lower',vmin=0,vmax=4095);axes[0,i].set_title(key.replace('_',' '))
                axes[1,i].imshow(f[key+'_error'].T,cmap='Reds',origin='lower');axes[1,i].set_title('Ошибки только resampling')
            for ax in axes.flat:ax.axis('off')
            fig.suptitle(case+' · train diagnostic, oracle не automatic');fig.tight_layout(rect=(0,0,1,.94),h_pad=2.5);fig.savefig(OUT/f'resolution_{case}.png',dpi=130);plt.close(fig)
    for case in ('ct_1001',plan['validation_cases'][0]):
        image=nib.load(DEFAULT_DATA/f'{case}_image.nii.gz');x=image.get_fdata(dtype=np.float32);gt=cardiac_target(np.asanyarray(nib.load(DEFAULT_DATA/f'{case}_label.nii.gz').dataobj));box=np.array(json.loads((OUT/'predicted_boxes.json').read_text())[case]['bbox']);z=int((box[0,2]+box[1,2]-1)//2)
        names=['GT','CHD68_v1','B_full_256','C_oracle_256','D_predicted_256'];fig,axes=plt.subplots(2,len(names)+1,figsize=(18,8))
        for row in range(2):axes[row,0].imshow(x[:,:,z].T,cmap='gray',origin='lower',vmin=0,vmax=4095);axes[row,0].set_title('Исходная КТ')
        for i,name in enumerate(names,1):
            pred=gt if name=='GT' else np.load(OUT/'evaluation'/'all_train'/name/case/'prediction.npy',mmap_mode='r');seg=pred[:,:,z].T;truth=gt[:,:,z].T;raw=x[:,:,z].T
            axes[0,i].imshow(raw,cmap='gray',origin='lower',vmin=0,vmax=4095);axes[0,i].imshow(np.ma.masked_where((seg==0)|(seg==255),seg),cmap=cmap,vmin=0,vmax=7,origin='lower',alpha=.5);axes[0,i].set_title(name)
            error=np.zeros(seg.shape,np.uint8);valid=truth<8;error[valid&(truth==0)&(seg>0)]=1;error[valid&(truth>0)&(seg==0)]=2;error[valid&(truth>0)&(seg>0)&(truth!=seg)]=3
            axes[1,i].imshow(raw,cmap='gray',origin='lower',vmin=0,vmax=4095);axes[1,i].imshow(np.ma.masked_where(error==0,error),cmap=ListedColormap(['#000000','#ffb020','#ff4d4f','#a855f7']),vmin=0,vmax=3,origin='lower',alpha=.8);axes[1,i].set_title('FP / FN / class error')
        for ax in axes.flat:ax.axis('off')
        fig.legend(handles=[Patch(color=HL[c][2],label=LABELS[c]) for c in range(1,8)],loc='lower center',ncol=7,frameon=False);fig.suptitle(case+f' · native Z={z}; C=Oracle diagnostic');fig.tight_layout(rect=(0,.05,1,.96));fig.savefig(OUT/f'native_comparison_{case}.png',dpi=130);plt.close(fig)
    # Export tables: preserve per-case/native metrics and complete false-component locations.
    csv_file('localization',[{k:v for k,v in r.items() if k!='per_class_coverage'}|{'coverage_'+LABELS[c]:r['per_class_coverage'][str(c)] for c in range(1,8)} for r in local['metrics']])
    csv_file('train_geometry',[{k:v for k,v in r.items() if k not in ('affine','class_bboxes')} for r in geometry]);csv_file('resampling',[{k:v for k,v in r.items() if k not in ('transform','class_areas')} for r in resampling])
    csv_file('segmentation_cases',[{'method':name,'case':r['case'],'role':'train example' if r['case']=='ct_1001' else 'development','mean_fg_Dice':r['macro_foreground_Dice'],'coverage':r['coverage']['foreground_coverage']} for name,rows in methods.items() for r in rows])
    csv_file('segmentation_classes',[{'method':name,'case':r['case'],'class':label,**score} for name,rows in methods.items() for r in rows for label,score in r['classes'].items()])
    csv_file('topology_classes',[{'case':r['case'],'method':r['method'],'class':LABELS[c],**{k:v for k,v in r['topology'][str(c)].items() if k not in ('components','component_sizes_desc')}} for r in topology for c in range(1,8)])
    csv_file('component_locations',[{'case':r['case'],'method':r['method'],'class':LABELS[c],**component} for r in topology for c in range(1,8) for component in r['topology'][str(c)]['components']])
    # Provenance: no original input/checkpoint/preprocessing or prior artifact mutation.
    actual_prior=prior_hashes()
    assert actual_prior==json.loads((OUT/'prior_files.json').read_text()),'Prior experiment output changed'
    assert protected_hashes()==json.loads((OUT/'protected_files.json').read_text()),'Historical weights/split/preprocessing changed'
    source_hashes={}
    for case in plan['train_cases']+plan['validation_cases']:
        for suffix,field in [('image','image_SHA256'),('label','mask_SHA256')]:
            path=DEFAULT_DATA/f'{case}_{suffix}.nii.gz';sha=fingerprint(path);assert sha==records[case][field];source_hashes[str(path)]=sha
    from ct_lab.roi_train import load_segmenter
    from ct_lab.roi_localizer import load_localizer
    checked=[]
    for item in learned['CV']+learned['main']:
        r=item['training'];load_segmenter(OUT/r['checkpoint']);assert fingerprint(OUT/r['checkpoint'])==r['checkpoint_sha256'];checked.append(r['checkpoint'])
    for r in local['CV']+[local['final']]:load_localizer(OUT/r['checkpoint']);assert fingerprint(OUT/r['checkpoint'])==r['checkpoint_sha256'];checked.append(r['checkpoint'])
    masks=[]
    for name,rows in methods.items():
        for r in rows:
            directory=OUT/'evaluation'/'all_train'/name/r['case'];pred=np.load(directory/'prediction.npy',mmap_mode='r');image=nib.load(DEFAULT_DATA/f'{r["case"]}_image.nii.gz');saved=nib.load(directory/'prediction.nii.gz')
            assert pred.shape==image.shape and saved.shape==image.shape;np.testing.assert_allclose(saved.affine,image.affine,atol=1e-7);assert np.array_equal(np.asanyarray(saved.dataobj),pred)
            masks.append({'method':name,'case':r['case'],'source_grid_roundtrip_exact':True})
    code_paths=list((REPO/'ct_lab').glob('roi_*.py'))+list((REPO/'ct_lab').glob('roi_*.cjs'))+[REPO/'ct_lab/roi_execution_gate.json',REPO/'scripts/run_roi05_runpod.sh',REPO/'ct_lab/app.py',REPO/'ct_lab/spatial_component/index.html',REPO/'tests/test_roi_lab.py']
    write(OUT/'integrity.json',{'prior_files_unchanged':len(actual_prior),'protected_unchanged':True,'source_hashes':source_hashes,'checkpoints_verified':checked,'native_masks_verified':masks,'sealed_test_read':False,'tests_passed':test_count,'UI_checks_passed':len(ui),'browser_verified':True,'stage05_code_SHA256':{str(p.relative_to(REPO)):fingerprint(p) for p in code_paths}})
    versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','nibabel','plotly','streamlit','scikit-image','pyvista','filelock')};versions.update(python=platform.python_version(),GPU=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',CUDA=torch.version.cuda);write(OUT/'versions.json',versions)
    def metric(name):return next(s for s in summaries if s['method']==name)
    p256=pairs[0];p384=pairs[1];branch=pairs[3];vg=[r for r in local['metrics'] if r['role']=='validation' and r['method']=='predicted']
    cvgate=json.loads((OUT/'training_gate.json').read_text());loss_classes={LABELS[c]:float(np.mean([r['per_class_coverage'][str(c)] for r in vg if r['per_class_coverage'][str(c)] is not None])) for c in range(1,8)}
    completely_lost=[(r['case'],LABELS[c]) for r in vg for c in range(1,8) if r['per_class_coverage'][str(c)]==0]
    worst_class=min((v,r['case'],LABELS[int(c)]) for r in vg for c,v in r['per_class_coverage'].items() if v is not None)
    predicted_boxes=json.loads((OUT/'predicted_boxes.json').read_text());fallbacks=sum(predicted_boxes[r['case']]['fallback_fixed_ROI'] for r in vg)
    class_gains={LABELS[c]:metric('D_predicted_256')['per_class_Dice'][LABELS[c]]-metric('B_full_256')['per_class_Dice'][LABELS[c]] for c in range(1,8)}
    class_gains_384={LABELS[c]:metric('D_predicted_384')['per_class_Dice'][LABELS[c]]-metric('B_full_384')['per_class_Dice'][LABELS[c]] for c in range(1,8)}
    def fragmentation(method,field):
        rows=[r for r in topology if r['method']==method and r['case'] in plan['validation_cases']]
        return float(np.mean([sum(r['topology'][str(c)][field] for c in range(1,8)) for r in rows]))
    def valid_recall(method):
        rows=[r for r in methods[method] if r['case'] in plan['validation_cases']]
        return float(np.mean([np.mean([s['recall'] for s in r['classes'].values() if s['target_voxels']>0]) for r in rows]))
    resize_text='; '.join(case+': '+', '.join('{mode}{size} Dice={roundtrip_fg_Dice_mean:.4f}, gradient={native_grid_gradient_preservation_ratio:.3f}'.format(**r) for r in resampling if r['case']==case) for case in plan['resize_diagnostic_cases'])
    answers=[
        f'Нормализованные центры масс train различаются по X/Y/Z: min={np.round(centers.min(0),3).tolist()}, max={np.round(centers.max(0),3).tolist()}. Размеры сосудистых bbox заметно варьируют; фиксированная ROI по train statistics почти равна всему FOV.',
        f'GT-bbox занимает в среднем {np.mean(ratios):.3f} FOV, диапазон [{min(ratios):.3f},{max(ratios):.3f}]. Предсказанная development ROI занимает в среднем {np.mean([r["ROI_to_full_FOV"] for r in vg]):.3f}; этого нельзя считать минимально необходимым объёмом или гарантией безопасности.',
        'Потеря деталей измерена после прямого и обратного преобразования изображения и масок на трёх train КТ. '+resize_text+'. Это ошибки resampling без нейросети; визуальное увеличение само по себе не доказывает пользу для сегментации.',
        f'Локализатор: development bbox IoU={np.mean([r["bbox_IoU"] for r in vg]):.4f}, center distance={np.mean([r["center_distance_index"] for r in vg]):.2f} grid indices; основной показатель coverage приведён отдельно.',
        f'Foreground coverage development={np.mean([r["foreground_coverage"] for r in vg]):.6f}. Это среднее по случаям; confidence coarse foreground не калиброван как вероятность такого покрытия.',
        f'Полное сохранение всех foreground voxels: {sum(r["all_structures_fully_covered"] for r in vg)}/10. Полностью исчезнувшие из ROI классы: {completely_lost or "нет"}. Худшее покрытие класса: {worst_class[0]:.6f} ({worst_class[1]}, {worst_class[2]}). Fallback к fixed ROI: {fallbacks}/10. Частичное обрезание нужно учитывать независимо от высокого среднего coverage.',
        f'Predicted ROI−Full FOV: native Dice Δ256={p256["mean"]:+.4f}, CI95={np.round(p256["case_CI95"],4).tolist()}; Δ384={p384["mean"]:+.4f}, CI95={np.round(p384["case_CI95"],4).tolist()}. При 256 результаты близки, при 384 Predicted ROI хуже Full FOV в данном бюджете. Это не устанавливает результат более длительного обучения.',
        f'Oracle256 Dice={metric("C_oracle_256")["mean"]:.4f}, automatic D256={metric("D_predicted_256")["mean"]:.4f}. Oracle−Full256: Δ={pairs[6]["mean"]:+.4f}, CI95={np.round(pairs[6]["case_CI95"],4).tolist()}. Oracle явно использует GT и не является автоматическим результатом или математически гарантированной верхней границей обученной модели. Разницу Oracle/Predicted нельзя объяснить только потерей GT-вокселей: отличаются bbox size, контекст и training distribution.',
        f'В среднем по development число ложных 6-связных foreground-компонент: Full256={fragmentation("B_full_256","false_components_no_GT_overlap"):.1f}, Predicted256={fragmentation("D_predicted_256","false_components_no_GT_overlap"):.1f}; их объём: {fragmentation("B_full_256","false_component_voxels"):.0f} → {fragmentation("D_predicted_256","false_component_voxels"):.0f} вокселей. Средний recall по присутствующим классам: {valid_recall("B_full_256"):.4f} → {valid_recall("D_predicted_256"):.4f}. Меньше компонент само по себе не означает лучшую анатомию: одновременно нужно учитывать пропуски. Компоненты не удалялись.',
        f'Dice на исходной сетке: B256={metric("B_full_256")["mean"]:.4f}, D256={metric("D_predicted_256")["mean"]:.4f}, B384={metric("B_full_384")["mean"]:.4f}, D384={metric("D_predicted_384")["mean"]:.4f}. Исторические CHD68 v1/v2 — отдельный reference с иным training budget.',
        'Изменения среднего class Dice D256−B256: '+', '.join(f'{c}={v:+.4f}' for c,v in class_gains.items())+'. При 384: '+', '.join(f'{c}={v:+.4f}' for c,v in class_gains_384.items())+'. Это описательные результаты одного seed, а не подтверждённое ранжирование пользы для классов.',
        f'Дополнительная 3D-core ветвь: Δ native Dice={branch["mean"]:+.4f}, CI95={np.round(branch["case_CI95"],4).tolist()}. Ветвь отключаемая, без Gaussian2 и старой MLP; число параметров увеличено на 896. Преимущество ветви в этом пилоте не подтверждено; одного seed недостаточно для устойчивого вывода.',
        'Для клинических КТ не подтверждены HU/physical spacing mapping, scanner generalization и coverage guarantee; cohort содержит release-cases с непроверенной пациентной уникальностью. Sealed test не использован, development reused; независимого клинического подтверждения нет.',
        'Сохранить: coverage-oriented localizer, train-only margin/CV, точное source-grid восстановление вероятностей, oracle labels, matched B/C/D budget, raw topology/error maps, CPU/GPU gates и epoch resume. Более длительное обучение/несколько seed и внешняя проверка — отдельный следующий этап, а не вывод из этого пилота.']
    lines=['# HEART_LOCALIZATION_ROI_SEGMENTATION_05','Дата: 2026-10-29. В существующем Heart3D выполнено новое ограниченное обучение локализатора и 2.5D ROI-сегментаторов. CHD68 v1/v2 только читались. Development уже использовалась ранее; это продолжение разработки, не независимая финальная оценка.',
           f'Основное сравнение на 10 development cases: Full256 Dice={metric("B_full_256")["mean"]:.4f}, Predicted256={metric("D_predicted_256")["mean"]:.4f}; Δ={p256["mean"]:+.4f}, case bootstrap CI95={np.round(p256["case_CI95"],4).tolist()}. При 384 Δ={p384["mean"]:+.4f}. Локализатор сохраняет в среднем {np.mean([r["foreground_coverage"] for r in vg])*100:.3f}% foreground в {np.mean([r["ROI_to_full_FOV"] for r in vg])*100:.1f}% FOV, но полностью все foreground voxels — только в {sum(r["all_structures_fully_covered"] for r in vg)}/10 случаях. Основной вариант, выбранный предварительной train-only CV: {selection["primary_pipeline"]}. Это ограниченный пилот с одним seed; высокая coverage не гарантирует выигрыш Dice.',
           '## Ответы на 14 исследовательских вопросов','\n\n'.join(f'{i+1}. {a}' for i,a in enumerate(answers)),
           '## Протокол, CV и вычислительный бюджет',
           'Frozen split: 48 train / 10 development / 10 sealed test. Test КТ не открывались. Исходные значения 0–4095 не интерпретируются как HU. Affine учитывается при преобразовании координат и поверхностей; расстояния указаны в единицах индексной сетки. Метки: 0 Background, 1 LV, 2 RV, 3 LA, 4 RA, 5 MYO, 6 AO, 7 PA; остальные → ignore=255, не background. Исходные файлы, веса и production preprocessing сохранены.',
           'Исходная предрегистрация: protocol.json; последующее предварительное решение только по train: training_gate.json. CV по 6 эпох × 48 шагов предсказывала только background. После проверки targets, геометрии и обновлений optimizer бюджет симметрично продлён до 18 × 48 только внутри train. Начальные результаты сохранены. Основные модели: 20 эпох × 192 шага × batch 2; модель, loss, margin и матрица экспериментов не менялись по внешней оценке сегментации. Train-only selection: '+json.dumps(selection,ensure_ascii=False),
           'Локализатор: три группы CV внутри train; лучшая эпоха по binary Dice на holdout своей группы, итоговое число эпох на всех 48 train — медиана этих трёх эпох. Margin выбран по train OOF coverage. Порог 0.4, крупнейшая foreground-компонента; пустой результат или средняя confidence < 0.5 → fixed train ROI. Fixed ROI: 2-й и 98-й процентили нормализованных train bbox плюс 0.05 FOV. Development ROI не исправляются по GT. В CV сегментатора используется локализатор своего fold и заранее заданный консервативный margin 0.3 + 8 индексов; main использует выбранный OOF margin 0.1 + 8. Это CV для разработки с выбором checkpoint и параметров, не независимая nested оценка.',
           table(['Fold','Pipeline','Native holdout Dice','Cases'],[[r['training']['context'],r['training']['job']['name'],r['evaluation']['mean_case_foreground_Dice'],len(r['evaluation']['cases'])] for r in learned['CV']]),
           'Бюджет CV — 864 optimizer steps и 32 train cases на fold; основной запуск — 3840 steps и 48 train cases. Абсолютные CV/development Dice поэтому не сопоставляются как оценка generalization gap. Предварительное ранжирование при малом бюджете также не гарантирует того же ранжирования при длительном обучении.',
           'Исторический clip 0–2015 заранее зафиксирован по 48 train в прежнем протоколе; на внешней development он не пересчитывается. Внутренняя CV наследует этот clip и поэтому также не является полностью независимой nested оценкой.',
           table(['Model','Parameters','Epochs','Steps/epoch','Batch','GPU peak MiB'],[[name,r['parameters'],r['epochs'],r['steps_per_epoch'],r['batch_size'],r['peak_GPU_bytes']/2**20 if r['peak_GPU_bytes'] else None] for name,r in training.items()]),
           'Архитектура HeartUNet25D: каналы 16/32/64/128, 489112 параметров; пять согласованных axial-срезов [-2, -1, 0, +1, +2] → 8 logits. Для B/C/D одного разрешения одинаковы параметры, loss, optimizer, augmentation, выбранные случаи и Z-позиции, число обновлений. При 384 увеличиваются FLOPs; равен бюджет обновлений и примеров, не время. Ветвь добавляет 4 native scalar maps и fusion на первом encoder-блоке, без Gaussian и прежних MLP weights. AdamW: lr=0.001, weight_decay=0.0001. Loss CE+Dice — float32; CUDA forward — AMP float16 с GradScaler, CPU — float32. Main best.pt содержит фиксированную последнюю эпоху; development не выбирает checkpoint. Исторические модели A имеют другой бюджет и FP32 inference; контролируемое сравнение проводится между новыми B/C/D.',
           '3D-core использует прежний CT-only extractor: Mean/Std по окну 5³; Gradient — центральные разности, на source edges односторонние; LoG — сумма вторых производных Gaussian с sigma=1 в индексной сетке этого набора, truncate=4. Отдельный Gaussian2 preprocessing/channel не добавляется. Source reflect и halo 8 индексов согласуют блоки; evaluation считает по 8 центральных Z за раз. Ограниченные patch intermediates прежнего extractor имеют float64 для устойчивости variance, результат — float32; кэш входов сети — float16. Маски — uint8/ignore255. Кэш 3D-core проверяет хэш исходных значений и геометрию, записывается атомарно и защищён file lock.',
           'Core рассчитывается на исходных source counts, без clip 2015: Mean / 2015 × 2 − 1, остальные каналы / 512. Центральные CT-каналы B/C/D используют общий исторический clip 0–2015. Поэтому branch-абляция проверяет весь дополнительный pipeline: 3D-сводки, сведения о высоких интенсивностях и 896 новых параметров одновременно. Её возможный выигрыш нельзя приписать только 3D-контексту; clip-matched и parameter-matched controls остаются следующим исследованием.',
           'Каждая main модель получила 7680 обучающих примеров с циклическим равным отбором случаев. На случай: 12 равномерных Z-позиций внутри объединённого train GT bbox и до 4 фоновых позиций снаружи. Центры одинаковы во всех режимах; GT применяется только к отбору train. Сдвиг интенсивности равномерный ±0.03 нормализованной шкалы (±30.225 source counts), Gaussian noise std=0.01 (10.075 counts). Одна фиксированная реализация на случай общая всем конфигурациям; вероятность выбора clean/aug — 50/50. Native core пересчитан на той же изменённой CT. Далее выполняется общий пространственный warp каналов и nearest targets: ±5°, масштаб ±5%, сдвиг ±3%, без flips и elastic. После 2D warp карты core интерполируются совместно; это не новое точное физическое 3D-измерение.',
           'Для согласованного training sampling глобальные Z-центры не отбрасываются даже вне predicted bbox по Z: применяется один XY crop и тот же набор центров B/C/D. При inference ограничивается также диапазон выходных центральных Z, а остальная source mask заполняется background. Это вариант 2.5D XY-ROI training с 3D ROI output gate, а не обучение только на строго вырезанных Z-центрах. Различие training/inference sampling раскрывается как ограничение протокола.',
           table(['Smoke device','Size','Batch','Mean s/step','Loss before','Loss after'],[[r['device'],r['size'],r['batch'],r['mean_seconds_step_after_warmup'],r['loss_before'],r['loss_after']] for r in smoke]),
           'Seed фиксирован; CUDA loss/interpolation предупреждают об отсутствии bitwise deterministic реализации. CPU seed/optimizer reproducibility протестирована. Main только один seed; train-only folds не заменяют повторные final runs.',
           '## Геометрия, coverage и resampling',
           table(['Role','Localizer','Coverage mean','BBox IoU','Center distance index','ROI/FOV','All foreground fully covered'],[[role,method,float(np.mean([r['foreground_coverage'] for r in local['metrics'] if r['role']==role and r['method']==method])),float(np.mean([r['bbox_IoU'] for r in local['metrics'] if r['role']==role and r['method']==method])),float(np.mean([r['center_distance_index'] for r in local['metrics'] if r['role']==role and r['method']==method])),float(np.mean([r['ROI_to_full_FOV'] for r in local['metrics'] if r['role']==role and r['method']==method])),sum(r['all_structures_fully_covered'] for r in local['metrics'] if r['role']==role and r['method']==method)] for role in ('train','validation') for method in ('fixed','predicted')]),
           table(['Class','Development mean coverage'],list(loss_classes.items())),
           table(['Train OOF margin','Mean coverage','Worst case coverage','Mean ROI/FOV'],[[r['margin'],r['OOF_mean_coverage'],r['OOF_worst_case_coverage'],r['OOF_mean_ROI_FOV']] for r in margin_rows]),
           'Координаты, размеры, центры, расстояния до границ и class bbox: train_geometry.json/csv. Margin — доля bbox плюс 8 индексов, не мм. Bbox полуоткрытый [lo, hi). Coarse cube 64³ используется только для локализации с последующим обратным mapping по исходному shape; это нормализованная сетка, не физически изотропное изображение. Z-контекст сегментатора сохраняет исходную сетку. При inference центральные срезы ограничены predicted bbox по Z, но соседние ±2 берутся с halo из исходного объёма. На границе source повторяется крайний срез.',
           table(['Case','Mode','Size','XY scale','Mask-only roundtrip Dice','Native gradient ratio','Image MAE counts','Thin MYO recall'],[[r['case'],r['mode'],r['size'],r['XY_scale'],r['roundtrip_fg_Dice_mean'],r['native_grid_gradient_preservation_ratio'],r['roundtrip_image_MAE_source_counts'],r['thin_MYO_recall_width_le4_index']] for r in resampling]),
           'Thin MYO: диагностические GT pixels с 2 × distance-to-background ≤ 4 индексов; это не клиническая толщина стенки. Ошибки resampling и изменения площадей сохранены отдельно. После inverse ignore остаётся ignore; valid → ignore учитывается как unclassified/FN в геометрическом Dice. Основные маски получают argmax после обратной интерполяции вероятностей. B против D одновременно меняет детализацию, анатомический контекст и ограничение FOV. Контроль full-model + ROI-gating отдельно не выполнялся, поэтому эффект нельзя приписать только визуальной детализации.',
           '## Сегментация на исходной сетке',
           'Все итоговые mean/SD/CI в таблицах ниже относятся ровно к 10 development cases. Train ct_1001 показан отдельной колонкой. Сырые evaluation summary содержат mean по всем запрошенным 11 случаям; это общий технический агрегат и не development-метрика. Для выводов используются case_ids и role в study_summary.json, segmentation_cases.csv и таблицы ниже.',
           table(['Method','Train ct1001 Dice','Development mean Dice','Case SD','Case CI95'],[[r['method'],r['train_ct1001_Dice'],r['mean'],r['case_SD'],r['case_CI95']] for r in summaries]),
           table(['Method',*LABELS[1:]],[[r['method'],*[r['per_class_Dice'][LABELS[c]] for c in range(1,8)]] for r in summaries]),
           table(['Paired comparison','Mean ΔDice','Case CI95'],[[r['comparison'],r['mean'],r['case_CI95']] for r in pairs]),
           'Case bootstrap: 2000 выборок из 10 development случаев; воксели и срезы не считаются независимыми наблюдениями. Case SD описывает эти десять случаев (ddof=0). CI описательные, без поправки на повторное использование development и множественные сравнения. Foreground-классы без GT исключены из macro; empty-empty не получает награду, а FP-объёмы сохраняются. Все методы оцениваются на одинаковых исходных CT/GT и valid 0–7 после восстановления исходной сетки. Oracle C использует GT bbox и всегда выделен отдельно.',
           '## Фрагментация, HD95 и 3D',
           table(['Method','Case','FG macro Dice','False FG components','False component voxels','BG→FG voxels'],[[r['method'],r['case'],r['macro_foreground_Dice'],sum(r['topology'][str(c)]['false_components_no_GT_overlap'] for c in range(1,8)),sum(r['topology'][str(c)]['false_component_voxels'] for c in range(1,8)),r['BG_to_foreground_voxels']] for r in topology]),
           'HD95, precision, recall, крупнейшая компонента и доля ложных компонентов по классам: topology_classes.csv и source scores. Все размеры, bbox и положения компонентов: topology.json и component_locations.csv. HD95 вычисляется точно через EDT/KD в единицах сетки; поверхности у source/ignore границ исключены из дистанционной оценки. При отсутствии одной поверхности HD95=None; Dice и recall отражают пропуск. Малые компоненты не удаляются, GT cleanup не применяется. Ложная компонента не пересекает GT своего класса, включая ошибки между структурами; это отличается от объёма BG → FG. Raw VTP Heart3D сохраняет все поверхности. Упрощение до 30 тысяч граней на класс применяется только для браузера. Основные метрики считаются по неизменённой маске в valid domain с 6-связностью. Raw VTP сохраняет также предсказания в ignored GT regions: экспертная маска не используется для очистки поверхности. Дополнительные данные surface helper с 26-связностью и полным raw domain приведены отдельно; их нельзя смешивать с primary counts.',
           'Marching cubes замыкает структуру у границ маски. Heart3D metadata отмечает artificial_boundary_caps при касании исходной границы скана; обрезание ROI также может создавать искусственные окончания сосудов. Замкнутая поверхность не доказывает анатомическую полноту; её нужно рассматривать вместе с coverage и картами ошибок.',
           'Для производительности используется максимум два независимых metric worker; при старте с доступной RAM < 4 GiB или отсутствии psutil — один. VTK экспорт последователен. KD-tree и EDT реализуют одну симметричную Euclidean HD95: на ct_1127 все семь классов совпали с допуском 1e−8 (HD95_exact_algorithms_7classes.json). KD применяется к разреженным поверхностям; EDT остаётся для crop > 5 миллионов вокселей и > 2 миллионов surface points. Меняется алгоритм вычисления, не определение метрики или маски.',
           '## Интерфейс, ограничения и следующий этап',
           'В существующий CT Lab добавлен раздел Heart Localization & ROI Segmentation: три исходные проекции, predicted/fixed/oracle bbox, связанные XYZ, прозрачность и выбор классов, сравнение crop, native predictions, карты FP/FN и ошибок классов, поверхности Heart3D и история обучения. Oracle на validation выключен по умолчанию; для него требуется явный диагностический переключатель. Автоматический inference не принимает маску. Результаты AppTest и браузерных проверок сохранены рядом с отчётом.',
           'Ограничения: ограниченный бюджет обучения с нуля на RTX 2060 и один основной seed; исторические CHD68 обучались с другим бюджетом. Срезы train отбирались по GT. Внутренняя CV наследует ранее выбранный intensity clip. Coarse bbox может частично обрезать тонкие и удалённые структуры; высокое среднее coverage не гарантирует покрытия каждого случая. HU и физические единицы не подтверждены; внешние клинические данные и sealed test не оценивались. Следующий этап: большее число train-only запусков, несколько seed и отдельная внешняя оценка после фиксации методики. Test не открывался и не использовался для подбора.',
           '## Графики и воспроизведение']
    for name,explanation in charts:lines.append(f'- [{name}](outputs/heart_localization_roi_05/charts/{name}.html) — {explanation}')
    for case in plan['resize_diagnostic_cases']:lines.append(f'![Resolution {case}](outputs/heart_localization_roi_05/resolution_{case}.png)')
    for case in ('ct_1001',plan['validation_cases'][0]):lines.append(f'![Native comparison {case}](outputs/heart_localization_roi_05/native_comparison_{case}.png)')
    lines+=['Команды из repository (PowerShell). Для независимого воспроизведения нужна отдельная копия с новым stage05 outputs; прежние результаты не удалять. Завершённые receipts пропускаются, last.pt продолжает обучение с границы эпохи. Перед browser запустить roi_launch в отдельном терминале. Для уже выделенного RTX 4090: ROI05_DATA_ROOT и ROI05_WEIGHTS_ROOT, затем bash scripts/run_roi05_runpod.sh; платные ресурсы этот скрипт не создаёт.\n```powershell\n.venv/Scripts/python.exe -m ct_lab.roi_analysis\n.venv/Scripts/python.exe -m ct_lab.roi_smoke\n.venv/Scripts/python.exe -m ct_lab.roi_localizer\n.venv/Scripts/python.exe -m ct_lab.roi_train\n.venv/Scripts/python.exe -m ct_lab.roi_baselines\n.venv/Scripts/python.exe -m ct_lab.roi_topology\n.venv/Scripts/python.exe -m ct_lab.roi_analysis\n.venv/Scripts/python.exe -m ct_lab.roi_verify_ui\nnode ct_lab/roi_browser.cjs\n.venv/Scripts/python.exe -m pytest -p ct_lab.roi_pytest tests/test_roi_lab.py tests/test_multiclass_lab.py tests/test_learnable_features.py tests/test_spatial_features.py tests/test_filtering_features.py tests/test_ct_lab.py tests/test_cardiac_inference.py -q\n.venv/Scripts/python.exe -m ct_lab.roi_report\n.venv/Scripts/python.exe -m ct_lab.roi_launch\n```',
            f'Целостность: проверены 116 исходных файлов; {len(actual_prior)} прежних артефактов не изменены побайтно. SHA256 весов v1/v2, split и preprocessing совпали. Проверены {len(checked)} новых checkpoints и {len(masks)} native NIfTI/array масок. Автотесты: {test_count}; проверки UI: {len(ui)} — все успешно. Браузерные результаты сохранены. Production training regression test не запускался.',table(['Library','Version'],versions.items())]
    (REPO/'HEART_LOCALIZATION_ROI_SEGMENTATION_05.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8');write(OUT/'chart_index.json',charts)
    (OUT/'charts.html').write_text('<meta charset="utf-8"><h1>Heart Localization & ROI Segmentation</h1>'+''.join(f'<p><a href="charts/{n}.html">{n}</a>: {e}</p>' for n,e in charts),encoding='utf-8')
    package_outputs()
    print('STAGE05 REPORT COMPLETE',len(methods),'methods',len(masks),'native masks',len(checked),'new checkpoints',flush=True)


def package_outputs():
    with zipfile.ZipFile(OUT/'heart_localization_roi_05.zip','w',zipfile.ZIP_DEFLATED) as z:
        paths=[p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ('.zip','.npy','.vtp') and p.name!='last.pt' and not any(k in p.parts for k in ('cache','eval_core','numba_cache'))]
        paths+=list((REPO/'ct_lab').glob('roi_*.py'))+list((REPO/'ct_lab').glob('roi_*.cjs'))+[REPO/'ct_lab/roi_execution_gate.json',REPO/'scripts/run_roi05_runpod.sh',REPO/'ct_lab/app.py',REPO/'ct_lab/spatial_component/index.html',REPO/'ct_lab/ROI_README.md',REPO/'tests/test_roi_lab.py',REPO/'HEART_LOCALIZATION_ROI_SEGMENTATION_05.md']
        for p in sorted(set(paths)):z.write(p,str(p.relative_to(REPO)))
        assert len(z.namelist())==len(set(z.namelist()))


if __name__=='__main__':run()
