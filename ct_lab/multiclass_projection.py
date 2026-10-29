"""Common clean TRAIN projection for linked clean/noisy source IDs."""
import json
import numpy as np
from ct_lab.multiclass_core import *
from ct_lab.multiclass_data import assemble
from ct_lab.multiclass_train import slug
from ct_lab.spatial_projection import project


def choose(data):
    rng=np.random.default_rng(SEEDS[0]);indices=[]
    for case in np.unique(data['cases']):
        for c in CLASSES:
            pool=np.flatnonzero((data['cases']==case)&(data['labels']==c));indices.extend(rng.choice(pool,min(24,len(pool)),replace=False).tolist())
    return np.array(sorted(indices),int)


def projection(job,kind='PCA',representation='Embedding'):
    plan=json.loads((OUT/'protocol.json').read_text());name=slug(job) if representation=='Embedding' else job['features'][0]+'_Inputs'
    path=OUT/'projections'/f'{name}_{kind}.npz';info=path.with_suffix('.json')
    if path.exists():
        with np.load(path) as f:return {k:f[k] for k in f.files},json.loads(info.read_text())
    names=SETS[job['features']];train=assemble(plan['train_cases'],'train',names);val=assemble(plan['validation_cases'],'natural',names)
    ti=choose(train);vi=choose(val);noisy_train=assemble(plan['train_cases'],'train',names,'noise');noisy_val=assemble(plan['validation_cases'],'natural',names,'noise12')
    if representation=='Embedding':model,saved=load_model(OUT/'models'/slug(job)/'best.pt');sc=saved['scaler']
    else:model=None;sc=fit_scaler(train['features'],plan['train_cases'])
    parts=[(train,ti,'train','clean'),(val,vi,'validation','clean'),(noisy_train,ti,'train','noise8'),(noisy_val,vi,'validation','noise12')]
    values=[];sample={k:[] for k in ('xyz','ids','labels','cases','role','condition')}
    for data,index,role,condition in parts:
        x=normalize(data['features'][index],sc)
        if model is not None:_,x=infer(model,x)
        values.append(x)
        for k in ('xyz','ids','labels','cases'):sample[k].append(data[k][index])
        sample['role'].append(np.repeat(role,len(index)));sample['condition'].append(np.repeat(condition,len(index)))
    sample={k:np.concatenate(v) for k,v in sample.items()};values=np.concatenate(values)
    fit=(sample['role']=='train')&(sample['condition']=='clean');embedding,details=project(values,fit,kind)
    sample.update(embedding=embedding,fit=fit)
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**sample)
    details.update({'representation':representation,'job':job,'same_source_ids_clean_noisy':True,'basis':'clean TRAIN only, noisy TRAIN/validation transform; no labels in fit',
                    'train_noise_sigma':8,'validation_noise_sigma':12,'sampling':'24/class/case for visualization, NOT natural-distribution evaluation'})
    write(info,details);return sample,details


def run():
    seed_all(SEEDS[0])
    configs=[({'features':'E core+all','method':m,'augmentation':a,'seed':SEEDS[0]},r) for m,a,r in
             [('CE','clean','Inputs'),('CE','clean','Embedding'),('CE','combined','Embedding'),('CE+SupCon','combined','Embedding')]]
    for job,rep in configs:
        for kind in ('PCA','UMAP'):projection(job,kind,rep);print('PROJECTION',job,rep,kind,flush=True)


if __name__=='__main__':run()
