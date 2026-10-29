"""Measured spatial charts, projections and GT-distance error diagnostics."""
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import ct_lab.spatial_features  # dependency paths and isolated stage-02 numba cache
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from ct_lab.spatial_run import OUT,DEFAULT_DATA,score,predict,metrics
from ct_lab.spatial_features import columns,feature_sets
from ct_lab.spatial_projection import project
from ct_lab.filtering import fit_scaler,standardize

COLORS=['#ef476f','#06d6a0']


def plot_projection(embeddings,labels,pair,title):
    fig=make_subplots(rows=1,cols=2,subplot_titles=('B: 2D baseline','Added spatial feature set'))
    for col,embedding in enumerate(embeddings,1):
        for c,color in zip(np.unique(labels),COLORS):
            keep=labels==c;fig.add_trace(go.Scattergl(x=embedding[keep,0],y=embedding[keep,1],mode='markers',marker=dict(size=4,opacity=.6,color=color),name=str(c),showlegend=col==1),1,col)
    fig.update_layout(title=title+' · '+pair,height=400,margin=dict(l=20,r=15,t=65,b=25))
    return fig


def run():
    result=json.loads((OUT/'results.json').read_text());plan=result['protocol']
    all_cases=plan['train_cases']+plan['validation_cases'];cache={};info={}
    for case in all_cases:
        with np.load(OUT/f'case_{case}.npz') as f:cache[case]={k:f[k] for k in f.files}
        info[case]=json.loads((OUT/f'case_{case}.json').read_text())
        np.savez_compressed(OUT/f'context_flags_{case}.npz',**{f'p{p}':((cache[case][f'p{p}_xyz'][:,2,None]+np.arange(-3,4)<0)|
                                        (cache[case][f'p{p}_xyz'][:,2,None]+np.arange(-3,4)>=info[case]['shape_xyz'][2])) for p in (0,1)})
    with np.load(OUT/'validation_predictions.npz') as f:vp={k:f[k] for k in f.files}
    development=[];diagnostics=[];projection_info={}
    for p,pair in enumerate(('LV/MYO','RV/PA')):
        # Five-case development voxel fold, reported separately and optimistic.
        truth=np.concatenate([cache[c][f'p{p}_labels'][~cache[c][f'p{p}_train']] for c in plan['train_cases']])
        groups=np.concatenate([np.repeat(c,np.count_nonzero(~cache[c][f'p{p}_train'])) for c in plan['train_cases']])
        dev_predictions={}
        for configuration,names in feature_sets().items():
            train=np.concatenate([columns({'names':info[c]['names'],'values':cache[c][f'p{p}_values']},names)[cache[c][f'p{p}_train']] for c in plan['train_cases']])
            labels=np.concatenate([cache[c][f'p{p}_labels'][cache[c][f'p{p}_train']] for c in plan['train_cases']])
            query=np.concatenate([columns({'names':info[c]['names'],'values':cache[c][f'p{p}_values']},names)[~cache[c][f'p{p}_train']] for c in plan['train_cases']])
            scored,pred=score(train,labels,query,truth,groups,np.unique(labels));dev_predictions[configuration[0]]=pred
            development.append({'pair':pair,'configuration':configuration,'scope':'five train groups voxel holdout; optimistic',**scored})
        for role,cases in [('train',plan['train_cases']),('validation',plan['validation_cases'])]:
            for case in cases:
                data=cache[case]
                if role=='train':
                    keep=~data[f'p{p}_train'];select=groups==case
                    arrays={f'p{p}_{letter}':pred[select] for letter,pred in dev_predictions.items()}
                    positions=data[f'p{p}_xyz'][keep];target=data[f'p{p}_labels'][keep];distance=data[f'p{p}_boundary_distance'][keep];ids=data[f'p{p}_voxel_ids'][keep]
                else:
                    select=vp[f'p{p}_cases']==case;arrays={f'p{p}_{letter}':vp[f'p{p}_{letter}'][select] for letter in 'ABCDEFGH'}
                    positions=vp[f'p{p}_xyz'][select];target=vp[f'p{p}_truth'][select];distance=vp[f'p{p}_boundary_distance'][select];ids=vp[f'p{p}_voxel_ids'][select]
                path=OUT/f'errors_{case}.npz'
                if path.exists():
                    with np.load(path) as f:merged={k:f[k] for k in f.files}
                else:merged={}
                merged.update(arrays);merged.update({f'p{p}_xyz':positions,f'p{p}_truth':target,f'p{p}_distance':distance,f'p{p}_voxel_ids':ids})
                np.savez_compressed(path,**merged)
                base=arrays[f'p{p}_B']
                for letter in 'CDEFGH':
                    pred=arrays[f'p{p}_{letter}']
                    for region,keep in [('boundary ≤1',distance<=1),('transition (1,3]',(distance>1)&(distance<=3)),('interior >3',distance>3)]:
                        wrong=(base!=target)&keep;fixed=wrong&(pred==target);new=(base==target)&(pred!=target)&keep
                        diagnostics.append({'case':case,'role':role,'pair':pair,'configuration':letter,'region':region,'samples':int(keep.sum()),
                                            'baseline_wrong':int(wrong.sum()),'corrected':int(fixed.sum()),'new_wrong':int(new.sum()),
                                            'correction_fraction':float(fixed.sum()/wrong.sum()) if wrong.any() else None})
        # Identical sampled points, train fold only; independent bases and shared-intersection controls.
        case='ct_1001';data=cache[case];bank={'names':info[case]['names'],'values':data[f'p{p}_values']};fold=data[f'p{p}_train'];labels=data[f'p{p}_labels']
        for kind in ('PCA','UMAP'):
            measured={}
            for configuration,names in feature_sets().items():
                path=OUT/f'projection_p{p}_{configuration[0]}_{kind}.npz';metadata=path.with_suffix('.json')
                if path.exists():
                    with np.load(path) as f:embedding=f['embedding']
                    params=json.loads(metadata.read_text())
                else:
                    embedding,params=project(columns(bank,names),fold,kind)
                    np.savez_compressed(path,embedding=embedding,voxel_ids=data[f'p{p}_voxel_ids'],xyz=data[f'p{p}_xyz'],labels=labels,train=fold)
                    metadata.write_text(json.dumps({'features':names,**params},indent=2),encoding='utf-8')
                measured[configuration]=embedding;projection_info[f'{pair} {configuration} {kind}']=params
                print('PROJECTION',pair,configuration,kind,flush=True)
            names=list(feature_sets());fig=plot_projection((measured[names[1]],measured[names[4]]),labels,pair,kind+' independent (not a common distance frame)')
            # Two baseline traces remain fixed; swap the two context traces.
            fig.update_layout(updatemenus=[{'buttons':[{'label':name,'method':'restyle','args':[{
                'x':[measured[name][labels==c,0].tolist() for c in np.unique(labels)],
                'y':[measured[name][labels==c,1].tolist() for c in np.unique(labels)]},[2,3]]} for name in names]}])
            (OUT/f'{kind.lower()}_{p}.html').write_text(fig.to_html(include_plotlyjs=True),encoding='utf-8')
            # Common basis on the measured feature intersection B∩H: exactly B, no synthetic padding.
            common=[n for n in feature_sets()[names[1]] if n in feature_sets()[names[-1]]]
            shared,params=project(columns(bank,common),fold,kind)
            projection_info[f'{pair} shared intersection {kind}']={'columns':common,'before_after_identical':True,**params}
            (OUT/f'{kind.lower()}_shared_{p}.html').write_text(plot_projection((shared,shared),labels,pair,'Shared intersection control: before = after').to_html(include_plotlyjs=True),encoding='utf-8')
    (OUT/'train_development.json').write_text(json.dumps(development,indent=2),encoding='utf-8')
    (OUT/'error_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    (OUT/'projection_manifest.json').write_text(json.dumps(projection_info,indent=2),encoding='utf-8')
    charts=[];main=[s for s in result['scores'] if s['scope']=='main unfiltered']
    fig=go.Figure()
    for pair in ('LV/MYO','RV/PA'):
        rows=[s for s in main if s['pair']==pair]
        fig.add_trace(go.Scatter(x=[s['configuration'] for s in rows],y=[s['case_macro']['macro_f1_mean'] for s in rows],
                                error_y={'type':'data','array':[s['case_macro']['macro_f1_std'] for s in rows]},name=pair,mode='lines+markers'))
    fig.update_layout(title='Macro F1 vs Feature Set · equal-case mean ± case SD (not CI)',yaxis_title='Case-macro F1');charts.append(('feature_sets',fig))
    for p,pair in enumerate(('LV/MYO','RV/PA')):
        fig=go.Figure()
        for s in main:
            if s['pair']==pair:fig.add_trace(go.Scatter(x=list(s['per_case']),y=[m['macro_f1'] for m in s['per_case'].values()],name=s['configuration'],mode='lines+markers'))
        fig.update_layout(title='Macro F1 by Case · '+pair);charts.append((f'by_case_{p}',fig))
        fig=go.Figure()
        for s in main:
            if s['pair']==pair and s['configuration'][0] in 'BCDE':
                radius={'B':0,'C':1,'D':2,'E':3}[s['configuration'][0]]
                fig.add_trace(go.Scatter(x=[radius],y=[s['case_macro']['macro_f1_mean']],name=s['configuration'],mode='markers',marker={'size':12}))
        fig.update_layout(title='Neighbor Distance vs F1 · '+pair,xaxis_title='Z radius in source index steps (not mm)');charts.append((f'neighbors_{p}',fig))
        fig=go.Figure()
        selected=[s for s in main if s['pair']==pair and s['configuration'][0] in 'BEFGH']
        fig.add_trace(go.Bar(x=[s['configuration'] for s in selected],y=[s['case_macro']['macro_f1_mean'] for s in selected]))
        fig.update_layout(title='2D vs 2.5D vs Tri-planar vs 3D · '+pair);charts.append((f'families_{p}',fig))
        fig=go.Figure()
        for i,s in enumerate([s for s in main if s['pair']==pair]):
            fig.add_trace(go.Heatmap(z=s['pooled_confusion_matrix'],x=pair.split('/'),y=pair.split('/'),visible=i==1,texttemplate='%{z}',colorscale='Blues'))
        fig.update_layout(title='Confusion Matrix · '+pair+' · rows=true / cols=predicted',updatemenus=[{'buttons':[
            {'label':s['configuration'],'method':'update','args':[{'visible':[j==i for j in range(8)]}]} for i,s in enumerate([s for s in main if s['pair']==pair])]}]);charts.append((f'confusion_{p}',fig))
        corr=json.loads((OUT/'correlations.json').read_text())[pair]
        used=list(dict.fromkeys(n for names in feature_sets().values() for n in names));idx=[corr['names'].index(n) for n in used]
        fig=go.Figure(go.Heatmap(z=np.array(corr['correlation'])[np.ix_(idx,idx)],x=used,y=used,zmin=-1,zmax=1,colorscale='RdBu'))
        fig.update_layout(title='Feature Correlation Matrix · train only · '+pair,height=800);charts.append((f'correlation_{p}',fig))
    # Error maps: predetermined primary development example + first validation case.
    for case in ('ct_1001',plan['validation_cases'][0]):
        image=nib.load(DEFAULT_DATA/f'{case}_image.nii.gz');volume=image.get_fdata(dtype=np.float32)
        with np.load(OUT/f'errors_{case}.npz') as f:errors={k:f[k] for k in f.files}
        for p,pair in enumerate(('LV/MYO','RV/PA')):
            positions=errors[f'p{p}_xyz'];zs=np.unique(positions[:,2]);z=int(zs[np.argmin(np.abs(zs-(volume.shape[2]-1)/2))]);keep=positions[:,2]==z
            fig=make_subplots(rows=1,cols=4,subplot_titles=['B baseline','E Z±3 vs B','F tri-planar vs B','G 3D vs B'])
            truth=errors[f'p{p}_truth'];base=errors[f'p{p}_B']
            for col,letter in enumerate('BEFG',1):
                fig.add_trace(go.Heatmap(z=volume[:,:,z].T,zmin=0,zmax=4095,colorscale='Gray',showscale=False),1,col)
                pred=errors[f'p{p}_{letter}'];colors=np.where(base!=truth,np.where(pred==truth,'#00ddff','#ef4444'),np.where(pred!=truth,'#ec4899','#22cc66'))
                fig.add_trace(go.Scattergl(x=positions[keep,0],y=positions[keep,1],mode='markers',marker={'color':colors[keep].tolist(),'size':4},showlegend=False),1,col)
            fig.update_layout(title=f'Spatial Error Maps · {case} · {pair} · Z={z} · green correct/red wrong/cyan fixed/magenta new',height=450)
            charts.append((f'errors_{case}_{p}',fig))
        point=errors['p0_xyz'][0];fig=go.Figure()
        for axis,name in enumerate('XYZ'):
            ids=np.arange(volume.shape[axis]);coordinates=np.tile(point,(len(ids),1));coordinates[:,axis]=ids
            fig.add_trace(go.Scatter(x=ids,y=volume[tuple(coordinates.T)],name=name))
        fig.update_layout(title=f'Intensity Profiles XYZ · {case} · source point {point.tolist()}',xaxis_title='Source index coordinate',yaxis_title='Source intensity');charts.append((f'profiles_{case}',fig))
    explanations={'feature_sets':'Среднее Macro F1 по release-case; error bars — SD между случаями, не доверительный интервал.',
                  'by_case':'Каждая точка относится к отдельному validation release-case; это продолжение разработки на ранее использованной validation.',
                  'neighbors':'Z-radius измерен в индексах. Более широкий контекст не обязательно полезнее; число sampled voxels неизменно.',
                  'families':'Различаются схема и размерность признаков. G включает три дополнительных Gaussian 3D канала; это не сравнение нейросетей.',
                  'confusion':'Матрица по sampled voxels, не по всем вокселям объёма. Порядок классов указан по осям.',
                  'correlation':'Pearson correlation только по train features. Корреляция не доказывает полезность и не заменяет ablation.',
                  'errors':'Только проверенные sampled voxels: зелёный — верно, красный — ошибка, cyan — исправлено относительно B, magenta — новая ошибка. Это не сегментация.',
                  'profiles':'Профили вдоль исходных XYZ через один voxel; расстояния — индексы/единицы заголовка, не мм.'}
    for name,fig in charts:
        explanation=explanations.get(name.split('_')[0],explanations.get(name,'Реальные данные зафиксированного протокола.'))
        (OUT/f'{name}.html').write_text('<meta charset="utf-8"><p>'+explanation+'</p>'+fig.to_html(full_html=False,include_plotlyjs=True),encoding='utf-8')
    (OUT/'charts.html').write_text('<meta charset="utf-8"><h1>Spatial Feature Analysis</h1><p>Продолжение разработки; не независимая финальная оценка. Все ошибки — sampled voxel k-NN, не сегментация. Source spacing не считается мм.</p>'+''.join(f'<p><a href="{name}.html">{name}</a></p>' for name,_ in charts)+''.join(f'<p><a href="{kind}_{p}.html">{kind.upper()} before/after · pair {p}</a></p>' for kind in ('pca','umap') for p in (0,1)),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(11,4))
    for pair in ('LV/MYO','RV/PA'):
        rows=[s for s in main if s['pair']==pair];ax.plot([s['configuration'] for s in rows],[s['case_macro']['macro_f1_mean'] for s in rows],marker='o',label=pair)
    ax.set_ylabel('Mean case Macro F1');ax.legend();ax.tick_params(axis='x',rotation=20);fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(OUT/f'feature_sets.{ext}',bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':run()
