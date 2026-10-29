"""Real ct_1001 linked HTML, fixed-scale images and reproducible PCA/UMAP arrays."""
import argparse
import base64
import io
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
import ct_lab.launch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from ct_lab.filtering import (CONFIGS,PAIRS,SEED,FEATURES,apply_filter,feature_maps,sample_voxels,matrix,projection)
from ct_lab.filtering_run import DEFAULT_DATA,DEFAULT_OUTPUT

COLORS=['#ef476f','#06d6a0']


def png_url(values,low=0.,high=4095.,mask=None,pair=None):
    gray=(np.clip((values-low)/(high-low),0,1)*255).astype(np.uint8)
    rgb=np.repeat(gray[:,:,None],3,axis=2)
    if mask is not None:
        for cls,color in zip(pair,[(239,71,111),(6,214,160)]):
            rgb[mask==cls]=(.6*rgb[mask==cls]+.4*np.array(color)).astype(np.uint8)
    buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def run(umap=False,output=DEFAULT_OUTPUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    image=nib.load(DEFAULT_DATA/'ct_1001_image.nii.gz');z=image.shape[2]//2
    x=image.get_fdata(dtype=np.float32)[:,:,z].T
    y=np.asanyarray(nib.load(DEFAULT_DATA/'ct_1001_label.nii.gz').dataobj)[:,:,z].T
    spacing=nib.affines.voxel_sizes(image.affine)[[1,0]]
    filtered=[apply_filter(x,c,spacing)[0] for c in CONFIGS]
    maps=[feature_maps(a,spacing) for a in filtered]
    np.savez_compressed(output/'ct_1001_planes.npz',source=x,**{f'processed_c{i}':a for i,a in enumerate(filtered)})
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for ax,config,plane in zip(axes.flat,CONFIGS,filtered):
        ax.imshow(plane,cmap='gray',vmin=0,vmax=4095,origin='lower');ax.set_title(config['name']);ax.set_xlabel('Source X');ax.set_ylabel('Source Y')
    fig.suptitle('ct_1001 Z=110 · one fixed source-intensity display scale [0,4095]');fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(output/f'ct_1001_fixed_scale.{ext}',bbox_inches='tight')
    plt.close(fig)
    sections=[];metadata={};export={}
    images=[png_url(a) for a in filtered];diffs=[png_url(np.abs(a-x),0,500) for a in filtered]
    for p,(pair_name,pair) in enumerate(PAIRS.items()):
        sample=sample_voxels(y,pair,'ct_1001',z,image.shape,per_class=600)
        values=[matrix(m,sample) for m in maps]
        states=[]
        for i,(config,vals) in enumerate(zip(CONFIGS,values)):
            state={'name':config['name'],'image':images[i],'difference':diffs[i],'features':vals.tolist()}
            for kind in ['PCA']+(['UMAP'] if umap else []):
                cache=output/f'projection_p{p}_c{i}_{kind}.npz'
                cache_meta=cache.with_suffix('.json')
                if cache.exists() and cache_meta.exists():
                    with np.load(cache) as data:embedding=data['embedding']
                    info=json.loads(cache_meta.read_text())
                else:
                    embedding,info=projection(vals,sample['train'],kind)
                    np.savez_compressed(cache,embedding=embedding,voxel_ids=sample['voxel_ids'],labels=sample['labels'],xyz=sample['xyz'],train=sample['train'])
                    cache_meta.write_text(json.dumps(info,indent=2,allow_nan=False),encoding='utf-8')
                state[kind]=embedding.tolist();metadata[f'{pair_name} {config["name"]} {kind}']=info
            states.append(state)
            export[f'p{p}_c{i}_features']=vals
            print('PROJECTED',pair_name,config['name'],'PCA/UMAP' if umap else 'PCA',flush=True)
        for key in ('voxel_ids','labels','xyz','train'):export[f'p{p}_{key}']=sample[key]
        main=make_subplots(rows=2,cols=3,specs=[[{},{},{}],[{}, {'colspan':2},None]],
                           subplot_titles=('Source (same scale)','Filtered (same scale)','Expert mask','Absolute difference [0,500]','Feature scatter: select points'))
        main.add_trace(go.Image(source=images[0]),1,1)
        main.add_trace(go.Scattergl(x=[],y=[],mode='markers',marker=dict(color='yellow',size=5),name='Selection'),1,1)
        main.add_trace(go.Image(source=images[0]),1,2)
        main.add_trace(go.Scattergl(x=[],y=[],mode='markers',marker=dict(color='yellow',size=5),showlegend=False),1,2)
        main.add_trace(go.Image(source=png_url(x,mask=y,pair=pair)),1,3)
        main.add_trace(go.Image(source=diffs[0]),2,1)
        for c,color in zip(pair,COLORS):
            keep=sample['labels']==c
            custom=np.column_stack([np.full(keep.sum(),z),sample['rows'][keep],sample['cols'][keep],sample['voxel_ids'][keep],sample['labels'][keep]])
            main.add_trace(go.Scattergl(x=values[0][keep,0].tolist(),y=values[0][keep,1].tolist(),mode='markers',customdata=custom.tolist(),
                                       marker=dict(color=color,size=4,opacity=.65),name=str(c),
                                       hovertemplate='source X=%{customdata[2]}, Y=%{customdata[1]}, Z=%{customdata[0]}<br>ID=%{customdata[3]}<extra></extra>'),2,2)
        main.update_layout(height=780,title=f'ct_1001 Z={z} · {pair_name}',dragmode='lasso',clickmode='event+select',margin=dict(l=30,r=20,t=70,b=30))
        for col in range(1,4):main.update_yaxes(autorange='reversed',scaleanchor={1:'x',2:'x2',3:'x3'}[col],row=1,col=col)
        main.update_yaxes(autorange='reversed',scaleanchor='x4',row=2,col=1)
        main.update_xaxes(title_text='Intensity',row=2,col=2);main.update_yaxes(title_text='Gradient Magnitude',row=2,col=2)
        main_id=f'linked_{p}'
        html=main.to_html(full_html=False,include_plotlyjs=p==0,div_id=main_id)
        projection_ids={}
        for kind in ['PCA']+(['UMAP'] if umap else []):
            pf=make_subplots(rows=1,cols=2,subplot_titles=(kind+' Before',kind+' After'))
            for col in (1,2):
                for c,color in zip(pair,COLORS):
                    keep=sample['labels']==c;embedding=np.asarray(states[0][kind])
                    pf.add_trace(go.Scattergl(x=embedding[keep,0],y=embedding[keep,1],mode='markers',marker=dict(size=4,color=color,opacity=.6),name=str(c),showlegend=col==1),1,col)
            pf.update_layout(height=370,margin=dict(l=25,r=20,t=45,b=25))
            pid=f'{kind.lower()}_{p}';projection_ids[kind]=pid
            html+=pf.to_html(full_html=False,include_plotlyjs=False,div_id=pid)
        labels=sample['labels'].tolist()
        payload=json.dumps({'states':states,'labels':labels,'pair':pair,'projections':projection_ids},allow_nan=False)
        controls=f'<h2>{pair_name}</h2><label>Filter <select id="config_{p}">'+''.join(f'<option value="{i}">{c["name"]}</option>' for i,c in enumerate(CONFIGS))+'</select></label> '
        controls+=f'<label>Scatter X <select id="x_{p}">'+''.join(f'<option value="{i}">{f}</option>' for i,f in enumerate(FEATURES))+'</select></label> '
        controls+=f'<label>Scatter Y <select id="y_{p}">'+''.join(f'<option value="{i}"'+(' selected' if i==1 else '')+f'>{f}</option>' for i,f in enumerate(FEATURES))+'</select></label>'
        # IDs and numeric payload originate from this controlled experiment.
        script='''<script>(()=>{const data=PAYLOAD, plot=document.getElementById('MAIN');
const cfg=document.getElementById('CONFIG'), xs=document.getElementById('XS'),ys=document.getElementById('YS');
function update(){const state=data.states[+cfg.value],a=+xs.value,b=+ys.value;
Plotly.restyle(plot,{source:state.image},[2]);Plotly.restyle(plot,{source:state.difference},[5]);
for(let c=0;c<2;c++){let ix=data.labels.map((v,i)=>v===data.pair[c]?i:-1).filter(i=>i>=0);
Plotly.restyle(plot,{x:[ix.map(i=>state.features[i][a])],y:[ix.map(i=>state.features[i][b])]},[6+c]);
for(const [kind,id] of Object.entries(data.projections)){const emb=state[kind];Plotly.restyle(id,{x:[ix.map(i=>emb[i][0])],y:[ix.map(i=>emb[i][1])]},[2+c]);}}
Plotly.relayout(plot,{'xaxis5.title.text':xs.options[a].text,'yaxis5.title.text':ys.options[b].text});
Plotly.restyle(plot,{x:[[]],y:[[]]},[1,3]);}
cfg.onchange=update;xs.onchange=update;ys.onchange=update;
function select(e){let points=(e&&e.points||[]).filter(p=>p.curveNumber===6||p.curveNumber===7);
let x=points.map(p=>p.customdata[2]),y=points.map(p=>p.customdata[1]);Plotly.restyle(plot,{x:[x],y:[y]},[1,3]);}
plot.on('plotly_selected',select);plot.on('plotly_click',select);})();</script>'''
        replacements={'PAYLOAD':payload,'MAIN':main_id,'CONFIG':f'config_{p}','XS':f'x_{p}','YS':f'y_{p}'}
        for old,new in replacements.items():script=script.replace(old,new)
        sections.append(controls+html+script)
    np.savez_compressed(output/'ct_1001_sampled_features.npz',**export)
    metadata.update(seed=SEED,case='ct_1001',z_source=z,scope='central source slice; 80/20 stratified voxel fold',
                    input_feature_names=FEATURES,UMAP_included=umap,source_scale_not_HU=True)
    (output/'projection_manifest.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
    warning='<p>Source intensity, no HU/mm claim. Independent UMAP fits have no common coordinate scale. Train voxel folds only; labels are annotation, never input features. Selection on scatter highlights source coordinates on CT. Filters were not optimized on validation.</p>'
    (output/'linked_ct1001.html').write_text('<!doctype html><meta charset="utf-8"><title>Filtering experiment 01</title><style>body{font:15px system-ui;margin:24px}select{margin:8px;padding:6px}</style><h1>Filtering & Feature Analysis</h1>'+warning+''.join(sections),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--umap',action='store_true');args=parser.parse_args();run(args.umap)
