"""Fixed feature-space stress tests and permutation diagnostics; no model refit."""
import hashlib
import json
import numpy as np
from ct_lab.learnable_core import OUT,SETS,PAIRS,SEED,normalize,load_encoder,infer
from ct_lab.learnable_data import assemble
from ct_lab.learnable_train import evaluate,knn_probability

KINDS=['None','Gaussian .05 SD','Intensity +.1 SD','Intensity scale 1.1','Drop 3D LoG']


def perturb(values,kind,features,cases,ids):
    x=np.array(values,np.float32,copy=True)
    if kind=='None':return x
    if kind=='Gaussian .05 SD':
        for i,(case,voxel) in enumerate(zip(cases,ids)):
            seed=int.from_bytes(hashlib.blake2b(f'{SEED}|{case}|{voxel}'.encode(),digest_size=8).digest(),'little')
            x[i]+=np.random.default_rng(seed).normal(0,.05,x.shape[1]).astype(np.float32)
    elif kind=='Intensity +.1 SD':x[:,features.index('Intensity')]+=.1
    elif kind=='Intensity scale 1.1':x[:,features.index('Intensity')]*=1.1
    elif kind=='Drop 3D LoG':x[:,features.index('3D LoG')]=0.
    else:raise ValueError('Unknown perturbation')
    return x


def selection(data,per_class=32):
    rng=np.random.default_rng(SEED);indices=[]
    for case in np.unique(data['cases']):
        for cls in np.unique(data['labels']):
            pool=np.flatnonzero((data['cases']==case)&(data['labels']==cls))
            indices.extend(rng.choice(pool,min(per_class,len(pool)),replace=False).tolist())
    return np.array(sorted(indices),int)


def run():
    result=json.loads((OUT/'results.json').read_text());plan=result['protocol'];robustness=[];importance=[]
    for p,(pair,classes) in enumerate(PAIRS.items()):
        for feature_set in plan['neural_inputs']:
            features=SETS[feature_set];train=assemble(plan,'train',p,features);val=assemble(plan,'validation',p,features)
            base=next(b for b in result['baselines'] if b['pair']==pair and b['features']==feature_set)
            scaler=base['scaler'];tx=normalize(train['features'],scaler);vx=normalize(val['features'],scaler)
            subset=selection(val);fixed=vx[subset];truth=val['labels'][subset];cases=val['cases'][subset]
            models=[('k-NN',None)]+[(m['method'],m) for m in result['models'] if m['pair']==pair and m['features']==feature_set and m['embedding_dim']==16]
            for method,receipt in models:
                if receipt:
                    model,_=load_encoder(OUT/receipt['checkpoint'],features)
                    def forward(x):return infer(model,x)
                else:
                    def forward(x):return knn_probability(tx,train['labels'],x,classes),x.copy()
                original_prob,original_embedding=forward(vx)
                original_eval=evaluate(val['labels'],original_prob,val['cases'],classes)
                for kind in KINDS:
                    changed=perturb(vx,kind,features,val['cases'],val['ids']);prob,z=forward(changed)
                    left=original_embedding/np.maximum(np.linalg.norm(original_embedding,axis=1,keepdims=True),1e-12)
                    right=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
                    score=evaluate(val['labels'],prob,val['cases'],classes)
                    robustness.append({'pair':pair,'features':feature_set,'method':method,'embedding_dim':16 if receipt else len(features),'perturbation':kind,
                                       'case_f1_clean':original_eval['case_macro']['macro_f1_mean'],'case_f1_perturbed':score['case_macro']['macro_f1_mean'],
                                       'paired_case_delta':{c:score['per_case'][c]['macro_f1']-original_eval['per_case'][c]['macro_f1'] for c in score['per_case']},
                                       'probability_abs_change_mean':float(np.abs(prob-original_prob).mean()),
                                       'embedding_raw_RMSE':float(np.sqrt(np.mean((z-original_embedding)**2))),
                                       'embedding_unit_RMSE':float(np.sqrt(np.mean((right-left)**2))),
                                       'scope':'synthetic FEATURE perturbation; NOT scanner/clinical noise; same per-ID perturbation for all methods'})
                baseline_prob,_=forward(fixed);baseline_score=evaluate(truth,baseline_prob,cases,classes)
                for j,name in enumerate(features):
                    permuted=fixed.copy();rng=np.random.default_rng(SEED+j)
                    for case in np.unique(cases):
                        keep=np.flatnonzero(cases==case);permuted[keep,j]=permuted[rng.permutation(keep),j]
                    prob,_=forward(permuted);score=evaluate(truth,prob,cases,classes)
                    importance.append({'pair':pair,'features':feature_set,'method':method,'feature':name,'fixed_samples':len(fixed),
                                       'case_f1_baseline':baseline_score['case_macro']['macro_f1_mean'],'case_f1_permuted':score['case_macro']['macro_f1_mean'],
                                       'f1_drop':baseline_score['case_macro']['macro_f1_mean']-score['case_macro']['macro_f1_mean'],
                                       'scope':'within-case permutation on fixed validation subset; diagnostic, not feature selection or causal importance'})
                print('DIAGNOSTICS',pair,feature_set,method,flush=True)
    (OUT/'robustness.json').write_text(json.dumps(robustness,indent=2),encoding='utf-8')
    (OUT/'importance.json').write_text(json.dumps(importance,indent=2),encoding='utf-8')


if __name__=='__main__':run()
