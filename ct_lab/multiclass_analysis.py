"""Held-out CT transformations, paired case/seed summaries, importance/correlations."""
import json
import numpy as np
from ct_lab.multiclass_core import *
from ct_lab.multiclass_data import assemble
from ct_lab.multiclass_metrics import evaluate,bootstrap
from ct_lab.multiclass_train import slug
from ct_lab.learnable_train import knn_probability


def run():
    result=json.loads((OUT/'results.json').read_text());plan=result['protocol'];seed_all(SEEDS[0]);rows=[];importance=[]
    for m in result['models']:
        directory=(OUT/m['checkpoint']).parent;path=directory/'robustness.json'
        if path.exists():rows.extend(json.loads(path.read_text()));continue
        model,saved=load_model(directory/'best.pt');features=saved['features'];clean=assemble(plan['validation_cases'],'natural',features)
        cp,cz=infer(model,normalize(clean['features'],saved['scaler']));local=[]
        for name in VAL_AUG:
            if name=='clean':data=clean;p,z=cp,cz
            else:data=assemble(plan['validation_cases'],'natural',features,name);p,z=infer(model,normalize(data['features'],saved['scaler']))
            score=evaluate(data,p,z);reference=evaluate({**clean,'valid':data['valid']},cp)
            valid=data['valid']&(data['labels']<8);cu=cz[valid]/np.maximum(np.linalg.norm(cz[valid],axis=1,keepdims=True),1e-12)
            zu=z[valid]/np.maximum(np.linalg.norm(z[valid],axis=1,keepdims=True),1e-12)
            local.append({k:m[k] for k in ('features','method','augmentation','seed')}|{'transformation':name,'metrics':score,
                         'clean_common_valid':reference,'case_delta':{c:score['per_case'][c]['macro_f1']-reference['per_case'][c]['macro_f1'] for c in score['per_case']},
                         'probability_abs_change_mean':float(np.abs(p[valid]-cp[valid]).mean()),'embedding_raw_RMSE':float(np.sqrt(np.mean((z[valid]-cz[valid])**2))),
                         'embedding_unit_RMSE':float(np.sqrt(np.mean((zu-cu)**2))),'paired_valid_voxels':int(valid.sum()),
                         'scope':'new source-CT transformation and feature recomputation, not feature-vector noise'})
            if name in ('clean','noise12','heldout-combined'):
                np.savez_compressed(directory/f'robust_{name}.npz',probability=p,embedding=z,valid=data['valid'])
        write(path,local);rows.extend(local);print('ROBUST',slug(m),flush=True)
    for feature in ('C 3D-core','E core+all'):
        path=OUT/'baselines'/f'{feature[0]}_robustness.json'
        if path.exists():rows.extend(json.loads(path.read_text()));continue
        train=assemble(plan['train_cases'],'train',SETS[feature]);clean=assemble(plan['validation_cases'],'natural',SETS[feature])
        baseline=next(b for b in result['baselines'] if b['features']==feature);sc=baseline['scaler'];tx=normalize(train['features'],sc)
        cp=knn_probability(tx,train['labels'],normalize(clean['features'],sc),CLASSES);local=[]
        for name in VAL_AUG:
            data=assemble(plan['validation_cases'],'natural',SETS[feature],name);values=normalize(data['features'],sc)
            p=knn_probability(tx,train['labels'],values,CLASSES);score=evaluate(data,p);ref=evaluate({**clean,'valid':data['valid']},cp)
            local.append({'features':feature,'method':'k-NN','augmentation':'clean','seed':None,'transformation':name,'metrics':score,'clean_common_valid':ref,
                          'case_delta':{c:score['per_case'][c]['macro_f1']-ref['per_case'][c]['macro_f1'] for c in score['per_case']},
                          'probability_abs_change_mean':float(np.abs(p-cp).mean()),'embedding_raw_RMSE':float(np.sqrt(np.mean((values-normalize(clean['features'],sc))**2))),
                          'embedding_unit_RMSE':None,'scope':'baseline standardized source feature stability'})
        rows.extend(local);write(path,local)
    for baseline in result['baselines']:
        if baseline['features'] in ('C 3D-core','E core+all'):continue
        rows.append({'features':baseline['features'],'method':'k-NN','augmentation':'clean','seed':None,'transformation':'clean',
                     'metrics':baseline['validation'],'clean_common_valid':baseline['validation'],
                     'case_delta':{c:0. for c in baseline['validation']['per_case']},'probability_abs_change_mean':0.,'embedding_raw_RMSE':0.,
                     'embedding_unit_RMSE':None,'scope':'deterministic clean baseline; no repeated identical seeds'})
    # First-seed E diagnostic permutation only; never used to tune/retrain.
    data=assemble(plan['validation_cases'],'natural',FEATURES);rng=np.random.default_rng(SEEDS[0]);subset=[]
    for case in plan['validation_cases']:
        pool=np.flatnonzero((data['cases']==case)&(data['labels']<8));subset.extend(rng.choice(pool,min(1024,len(pool)),replace=False))
    subset=np.array(subset);small={k:v[subset] for k,v in data.items()}
    for m in result['models']:
        if m['seed']!=SEEDS[0] or m['features']!='E core+all':continue
        model,saved=load_model(OUT/m['checkpoint']);x=normalize(columns(small['features'],saved['features']),saved['scaler']);p,_=infer(model,x)
        base=evaluate(small,p)
        for i,name in enumerate(saved['features']):
            altered=x.copy()
            for case in plan['validation_cases']:
                keep=np.flatnonzero(small['cases']==case);altered[keep,i]=altered[rng.permutation(keep),i]
            q,_=infer(model,altered);score=evaluate(small,q)
            importance.append({'features':m['features'],'method':m['method'],'augmentation':m['augmentation'],'feature':name,
                               'F1_drop':base['case_macro']['macro_f1_mean']-score['case_macro']['macro_f1_mean'],
                               'per_class_F1_drop':{str(c):base['pooled']['classwise'][str(c)]['f1']-score['pooled']['classwise'][str(c)]['f1'] for c in CLASSES},
                               'scope':'fixed natural validation subset, permutation within case; correlated-channel diagnostic, not causal'})
    train=assemble(plan['train_cases'],'train',FEATURES);corr=np.corrcoef(train['features'].astype(float).T)
    write(OUT/'correlations.json',{'features':FEATURES,'matrix':corr.tolist(),'fit_scope':'train only','strong_pairs':[
        {'left':FEATURES[i],'right':FEATURES[j],'r':float(corr[i,j])} for i in range(len(FEATURES)) for j in range(i+1,len(FEATURES)) if abs(corr[i,j])>.95]})
    # Equal-weight seed averages first, then bootstrap cases. Never voxel bootstrap.
    groups={}
    for r in rows:groups.setdefault((r['features'],r['method'],r['augmentation'],r['transformation']),[]).append(r)
    summaries=[]
    for key,parts in groups.items():
        cases=list(parts[0]['metrics']['per_case']);values=np.array([[p['metrics']['per_case'][c]['macro_f1'] for c in cases] for p in parts]);means=values.mean(0)
        summaries.append(dict(zip(('features','method','augmentation','transformation'),key))|{
            'n_seeds':len(parts),'seed_mean_F1':float(values.mean()),'seed_SD_F1':float(values.mean(1).std(ddof=1 if len(parts)>1 else 0)),'case_summary':bootstrap(means),
            'per_case_seed_mean':dict(zip(cases,means.tolist())),'per_class_F1_seed_mean':{str(c):float(np.mean([p['metrics']['pooled']['classwise'][str(c)]['f1'] for p in parts])) for c in CLASSES}})
    write(OUT/'robustness.json',rows);write(OUT/'importance.json',importance);write(OUT/'summary.json',summaries)
    print('ANALYSIS COMPLETE',len(rows),'scores',flush=True)


if __name__=='__main__':run()
