"""Train-fit representation projections on fixed source IDs and compact charts."""
import json
import numpy as np
from ct_lab.learnable_core import OUT,SETS,PAIRS,normalize,load_encoder,infer
from ct_lab.learnable_data import assemble
from ct_lab.learnable_diagnostics import selection
from ct_lab.spatial_projection import project
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def run():
    result=json.loads((OUT/'results.json').read_text());plan=result['protocol'];manifest={};figures=[]
    for p,(pair,classes) in enumerate(PAIRS.items()):
        for feature_set in plan['neural_inputs']:
            train=assemble(plan,'train',p,SETS[feature_set]);val=assemble(plan,'validation',p,SETS[feature_set]);ti=selection(train);vi=selection(val)
            sampled={key:np.concatenate([train[key][ti],val[key][vi]]) for key in ('labels','cases','xyz','ids')}
            fold=np.arange(len(ti)+len(vi))<len(ti)
            base=next(b for b in result['baselines'] if b['pair']==pair and b['features']==feature_set)
            x=np.concatenate([normalize(train['features'][ti],base['scaler']),normalize(val['features'][vi],base['scaler'])])
            representations={'Before':x}
            for receipt in result['models']:
                if receipt['pair']==pair and receipt['features']==feature_set and receipt['embedding_dim']==16:
                    model,_=load_encoder(OUT/receipt['checkpoint'],SETS[feature_set]);_,embedding=infer(model,x);representations[receipt['method']]=embedding
            for kind in ('PCA','UMAP'):
                fig=make_subplots(rows=2,cols=4,subplot_titles=[f'{name} · {role}' for role in ('train','validation') for name in representations])
                for col,(name,values) in enumerate(representations.items(),1):
                    path=OUT/f'projection_{pair}_{feature_set[0]}_{name.replace("+","_")}_{kind}.npz'
                    meta=path.with_suffix('.json')
                    if path.exists():
                        with np.load(path) as f:embedding=f['embedding']
                        info=json.loads(meta.read_text())
                    else:
                        embedding,info=project(values,fold,kind)
                        np.savez_compressed(path,embedding=embedding,representation=values,fold=fold,**sampled)
                        meta.write_text(json.dumps(info,indent=2),encoding='utf-8')
                    manifest[path.name]={'pair':pair,'feature_set':feature_set,'method':name,'kind':kind,'samples':len(fold),'train_samples':int(fold.sum()),'fit':'TRAIN only','metadata':info}
                    for row,trainflag in ((1,True),(2,False)):
                        for cls,color in zip(classes,('#ef476f','#06d6a0')):
                            keep=(fold==trainflag)&(sampled['labels']==cls)
                            custom=np.column_stack([sampled['xyz'][keep],sampled['ids'][keep],sampled['labels'][keep],sampled['cases'][keep]])
                            fig.add_trace(go.Scattergl(x=embedding[keep,0],y=embedding[keep,1],customdata=custom.tolist(),mode='markers',
                                                      marker={'size':3,'opacity':.6,'color':color},name=str(cls),showlegend=col==1 and row==1,
                                                      hovertemplate='XYZ=%{customdata[0]},%{customdata[1]},%{customdata[2]}<br>ID=%{customdata[3]}, case=%{customdata[5]}<extra></extra>'),row,col)
                    print('PROJECTED',pair,feature_set,name,kind,flush=True)
                fig.update_layout(height=750,title=f'{kind} · {pair} · {feature_set} · independent bases; labels only color; fixed e16')
                figures.append((f'projection_{pair}_{feature_set[0]}_{kind}',fig))
    fig=go.Figure()
    for m in result['models']:
        if m['embedding_dim']!=16:continue
        history=json.loads((OUT/m['checkpoint']).with_name('history.json').read_text())
        fig.add_trace(go.Scatter(x=[h['epoch'] for h in history],y=[h['validation_case_f1'] for h in history],name=f"{m['pair']} {m['features'][0]} {m['method']}"))
    fig.update_layout(title='Validation case F1 / epoch · early stopping; not independent evaluation',xaxis_title='Epoch',yaxis_title='Case-macro F1');figures.append(('history_f1',fig))
    fig=go.Figure()
    for m in result['models']:
        if m['embedding_dim']!=16:continue
        history=json.loads((OUT/m['checkpoint']).with_name('history.json').read_text())
        fig.add_trace(go.Scatter(x=[h['epoch'] for h in history],y=[h['loss'] for h in history],name=f"{m['pair']} {m['features'][0]} {m['method']}"))
    fig.update_layout(title='Loss / epoch · CE and combined losses have different scales',xaxis_title='Epoch');figures.append(('history_loss',fig))
    fig=go.Figure()
    for m in result['models']:
        if m['embedding_dim']!=16:continue
        baseline=next(b for b in result['baselines'] if b['pair']==m['pair'] and b['features']==m['features'])
        cases=list(m['validation']['per_case']);delta=[m['validation']['per_case'][c]['macro_f1']-baseline['validation']['per_case'][c]['macro_f1'] for c in cases]
        fig.add_trace(go.Scatter(x=cases,y=delta,mode='lines+markers',name=f"{m['pair']} {m['features'][0]} {m['method']}"))
    fig.update_layout(title='Paired case ΔF1 vs k-NN · fixed e16',yaxis_title='MLP minus k-NN');figures.append(('paired_case_f1',fig))
    (OUT/'projection_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for name,fig in figures:(OUT/(name+'.html')).write_text('<meta charset="utf-8"><p>Representation learning на sampled voxels; не whole-volume segmentation. PCA/UMAP — отдельные координатные системы, не доказательство роста Dice.</p>'+fig.to_html(full_html=False,include_plotlyjs=True),encoding='utf-8')
    (OUT/'charts.html').write_text('<meta charset="utf-8"><h1>Learnable Feature Representation</h1>'+''.join(f'<p><a href="{name}.html">{name}</a></p>' for name,_ in figures),encoding='utf-8')


if __name__=='__main__':run()
