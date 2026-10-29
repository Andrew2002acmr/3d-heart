"""36 predeclared CPU feature encoders + internal group CV; no CHD68 training."""
import json
import time
import numpy as np
import torch
from ct_lab.multiclass_core import *
from ct_lab.multiclass_data import assemble,prepare
from ct_lab.multiclass_metrics import evaluate
from ct_lab.learnable_train import knn_probability,balanced_pools


def slug(job):return f'{job["features"][0]}_{job["method"].replace("+","_")}_{job["augmentation"]}_s{job["seed"]}'


def fit(job,train_cases,val_cases,plan,cv=None):
    directory=OUT/('models' if cv is None else f'CV/fold{cv}')/slug(job);receipt=directory/'result.json'
    if receipt.exists():return json.loads(receipt.read_text())
    start=time.perf_counter();seed_all(job['seed']);features=SETS[job['features']];cfg=plan['training']
    train=assemble(train_cases,'train',features);validation=assemble(val_cases,'natural',features)
    scaler=fit_scaler(train['features'],train_cases);x=normalize(train['features'],scaler);vx=normalize(validation['features'],scaler)
    if job['augmentation']!='clean':aug=assemble(train_cases,'train',features,job['augmentation']);ax=normalize(aug['features'],scaler);av=aug['valid']
    else:ax=x;av=np.ones(len(x),bool)
    neighbors=[];valid=[]
    if job['method']=='CE+Spatial':
        for case in train_cases:
            with np.load(OUT/'data'/f'{case}.npz') as f:neighbors.append(columns(f['neighbor_features'],features));valid.append(f['neighbor_valid'])
        nx=torch.from_numpy(normalize(np.concatenate(neighbors),scaler));nv=np.concatenate(valid)
    model=FeatureEncoder(len(features),16,8);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    tx=torch.from_numpy(x);ta=torch.from_numpy(ax);ty=torch.from_numpy(train['labels'].astype(np.int64));pools=balanced_pools(train['labels'],train['cases'])
    steps=int(np.ceil(len(x)/cfg['batch_size']));history=[];best=-1;stale=0
    for epoch in range(cfg['epochs']):
        model.train();rng=np.random.default_rng(job['seed']+epoch*100003);sums=np.zeros(4)
        for _ in range(steps):
            index=np.concatenate([rng.choice(pool,cfg['batch_size']//8,replace=False,p=w) for pool,w in pools]);rng.shuffle(index)
            mixed=tx[index].clone();use=np.zeros(len(index),bool)
            if job['augmentation']!='clean':use=(rng.random(len(index))<.5)&av[index];mixed[use]=ta[index[use]]
            opt.zero_grad(set_to_none=True);logits,z=model(mixed);ce=torch.nn.functional.cross_entropy(logits,ty[index])
            con=supcon_loss(z,ty[index],cfg['temperature']) if job['method']=='CE+SupCon' else z.sum()*0
            spatial=z.sum()*0
            if job['method']=='CE+Spatial':
                keep=np.flatnonzero(nv[index]);_,nz=model(nx[index[keep]]);spatial=spatial_loss(z[keep],nz)
            loss=ce+.1*con+.1*spatial
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();opt.step();sums+=np.array([loss.item(),ce.item(),con.item(),spatial.item()])
        vp,_=infer(model,vx);score=evaluate(validation,vp);f1=score['case_macro']['macro_f1_mean'];tp,_=infer(model,x)
        train_score=evaluate(train,tp);history.append({'epoch':epoch+1,**dict(zip(('loss','CE','SupCon','Spatial'),(sums/steps).tolist())),
                          'train_balanced_case_f1':train_score['case_macro']['macro_f1_mean'],'validation_natural_case_f1':f1})
        if f1>best+cfg['min_delta']:
            best=f1;stale=0;directory.mkdir(parents=True,exist_ok=True)
            saved={'format':'multiclass_feature_04_v1','state_dict':model.state_dict(),'features':features,'feature_set':job['features'],'classes':CLASSES,
                   'embedding_dim':16,'scaler':scaler,'train_cases':train_cases,'validation_cases':val_cases,'split_sha256':plan['split_sha256'],
                   'job':job,'epoch':epoch+1,'training':cfg,'CV_fold':cv}
            torch.save(saved,directory/'best.pt')
        else:stale+=1
        if stale>=cfg['patience']:break
    model,saved=load_model(directory/'best.pt');vp,vz=infer(model,vx);tp,tz=infer(model,x)
    unit=tz/np.maximum(np.linalg.norm(tz,axis=1,keepdims=True),1e-12)
    if tz.std(0).mean()<1e-8 or unit.std(0).mean()<1e-8:raise ValueError('Embedding collapse')
    natural=assemble(train_cases,'natural',features);np_,nz=infer(model,normalize(natural['features'],scaler))
    np.savez_compressed(directory/'predictions.npz',validation_prob=vp,validation_embedding=vz,train_natural_prob=np_,train_natural_embedding=nz)
    result={**job,'CV_fold':cv,'checkpoint':str((directory/'best.pt').relative_to(OUT)),'checkpoint_sha256':fingerprint(directory/'best.pt'),
            'best_epoch':saved['epoch'],'epochs_run':len(history),'seconds':time.perf_counter()-start,
            'train_balanced':evaluate(train,tp,tz),'train_natural':evaluate(natural,np_,nz),'validation':evaluate(validation,vp,vz),
            'parameters':sum(p.numel() for p in model.parameters()),'unit_embedding_std_mean':float(unit.std(0).mean())}
    write(directory/'history.json',history);write(receipt,result);print('MODEL',slug(job),'CV',cv,'F1',result['validation']['case_macro']['macro_f1_mean'],flush=True)
    return result


def run():
    plan,_=freeze();prepare();seed_all(SEEDS[0]);baselines=[]
    for name,features in SETS.items():
        path=OUT/'baselines'/f'{name[0]}.json'
        if path.exists():baseline=json.loads(path.read_text())
        else:
            train=assemble(plan['train_cases'],'train',features);val=assemble(plan['validation_cases'],'natural',features)
            sc=fit_scaler(train['features'],plan['train_cases']);x=normalize(train['features'],sc);vx=normalize(val['features'],sc)
            probability=knn_probability(x,train['labels'],vx,CLASSES)
            baseline={'features':name,'method':'k-NN','scaler':sc,'validation':evaluate(val,probability,vx)}
            write(path,baseline);np.savez_compressed(path.with_suffix('.npz'),validation_prob=probability)
        baselines.append(baseline);print('BASELINE',name,baseline['validation']['case_macro']['macro_f1_mean'],flush=True)
    models=[]
    for job in plan['jobs']:
        models.append(fit(job,plan['train_cases'],plan['validation_cases'],plan));write(OUT/'progress.json',{'models':models})
    cv=[]
    for f,fold in enumerate(plan['CV_folds']):
        train=[c for c in plan['train_cases'] if c not in fold]
        for candidate in plan['CV_jobs']:cv.append(fit({**candidate,'seed':SEEDS[0]},train,fold,plan,f))
    write(OUT/'results.json',{'protocol':plan,'baselines':baselines,'models':models,'CV':cv});print('MULTICLASS TRAINING COMPLETE',len(models),len(cv),flush=True)


if __name__=='__main__':run()
