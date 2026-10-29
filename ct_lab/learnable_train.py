"""CPU-bounded MLP experiment; immutable production weights and sealed test."""
import json
import time
import numpy as np
import torch
from scipy.spatial import cKDTree,distance
from ct_lab.learnable_core import (OUT,REPO,SEED,PAIRS,SETS,FeatureEncoder,seed_all,supcon_loss,spatial_loss,
                                   normalize,fit_scaler,class_indices,infer,load_encoder)
from ct_lab.learnable_data import freeze,extract_case,assemble
from ct_lab.core import fingerprint


def classification(truth,pred,classes):
    cm=np.array([[np.count_nonzero((truth==a)&(pred==b)) for b in classes] for a in classes])
    tp=cm.diagonal();actual=cm.sum(1);predicted=cm.sum(0);den=actual+predicted
    recall=np.divide(tp,actual,out=np.zeros(len(classes),float),where=actual>0)
    precision=np.divide(tp,predicted,out=np.zeros(len(classes),float),where=predicted>0)
    f1=np.divide(2*tp,den,out=np.zeros(len(classes),float),where=den>0)
    return {'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall[actual>0].mean()),'confusion_matrix':cm.tolist(),
            'classwise':{str(c):{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'target':int(actual[i]),'predicted':int(predicted[i])} for i,c in enumerate(classes)}}


def evaluate(truth,prob,cases,classes,embedding=None):
    prediction=np.asarray(classes)[np.argmax(prob,1)]
    per_case={str(c):classification(truth[cases==c],prediction[cases==c],classes) for c in np.unique(cases)}
    result={'per_case':per_case,'case_macro':{key+'_mean':float(np.mean([v[key] for v in per_case.values()])) for key in ('macro_f1','balanced_accuracy')},
            'pooled':classification(truth,prediction,classes)}
    for key in ('macro_f1','balanced_accuracy'):result['case_macro'][key+'_std']=float(np.std([v[key] for v in per_case.values()]))
    if embedding is not None:
        from sklearn.metrics import silhouette_score
        rng=np.random.default_rng(SEED);indices=np.concatenate([rng.choice(np.flatnonzero(truth==c),min(256,np.count_nonzero(truth==c)),replace=False) for c in classes])
        z=np.asarray(embedding[indices],float);y=truth[indices];parts=[z[y==c] for c in classes]
        if len(parts)==2 and all(len(x)>1 for x in parts):
            intra=np.mean(np.concatenate([distance.pdist(x) for x in parts]));inter=distance.cdist(parts[0],parts[1]).mean()
            variance=sum(x.var(0).sum() for x in parts)
            result['geometry']={'silhouette':float(silhouette_score(z,y)),
                                'fisher_ratio':float(np.sum((parts[0].mean(0)-parts[1].mean(0))**2)/variance) if variance else None,
                                'intra_class_distance':float(intra),'inter_class_distance':float(inter),'inter_intra_ratio':float(inter/intra) if intra else None,
                                'raw_embedding_std_mean':float(np.std(embedding,axis=0).mean()),'sample_count':len(z),
                                'space':'raw encoder embedding; baseline uses train-standardized source features; NOT PCA/UMAP'}
    return result


def knn_probability(train,labels,query,classes,k=15):
    tree=cKDTree(train);d,i=tree.query(query,k=min(k,len(train)),workers=1)
    if d.ndim==1:d,i=d[:,None],i[:,None]
    zero=d==0;w=np.where(zero.any(1)[:,None],zero.astype(float),1/np.maximum(d,1e-300))
    votes=np.stack([(w*(labels[i]==c)).sum(1) for c in classes],1);return votes/np.maximum(votes.sum(1,keepdims=True),1e-30)


def legacy_ablation(plan):
    """First experiment: exactly the stage02 sampled points and five train groups."""
    source=REPO/'outputs/spatial_feature_02';old=json.loads((source/'protocol.json').read_text());results=[]
    for p,(pair,classes) in enumerate(PAIRS.items()):
        def get(role,names):
            values=[];truth=[];cases=[]
            for case in old[role+'_cases']:
                info=json.loads((source/f'case_{case}.json').read_text());indices=[info['names'].index(n) for n in names]
                with np.load(source/f'case_{case}.npz') as f:
                    values.append(f[f'p{p}_values'][:,indices]);truth.append(f[f'p{p}_labels']);cases.append(np.repeat(case,len(truth[-1])))
            return np.concatenate(values),np.concatenate(truth),np.concatenate(cases)
        for name,names in SETS.items():
            train,labels,_=get('train',names);query,truth,cases=get('validation',names)
            scaler=fit_scaler(train,old['train_cases']);t=normalize(train,scaler);q=normalize(query,scaler)
            prob=knn_probability(t,labels,q,classes)
            results.append({'scope':'legacy 5 train/10 validation; stage02 exact IDs','pair':pair,'features':name,'columns':names,
                            'scaler':scaler,**evaluate(truth,prob,cases,classes)})
            print('ABLATION',pair,name,results[-1]['case_macro']['macro_f1_mean'],flush=True)
    (OUT/'ablation_legacy.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    return results


def balanced_pools(labels,cases):
    pools=[]
    for c in np.unique(labels):
        indices=np.flatnonzero(labels==c);group,counts=np.unique(cases[indices],return_counts=True)
        weight=np.zeros(len(indices),float)
        for g,count in zip(group,counts):weight[cases[indices]==g]=1/count
        pools.append((indices,weight/weight.sum()))
    return pools


def train_one(plan,pair,feature_set,method,dim,data_train,data_val,scaler):
    p=list(PAIRS).index(pair);f=plan['neural_inputs'].index(feature_set)
    seed=SEED+1000*p+100*f+dim;seed_all(seed)
    directory=OUT/'models'/pair/feature_set[0]/f'{method.replace("+","_")}_e{dim}'
    directory.mkdir(parents=True,exist_ok=True);checkpoint=directory/'best.pt';receipt=directory/'result.json'
    if receipt.exists() and checkpoint.exists():return json.loads(receipt.read_text())
    config=plan['training'];features=SETS[feature_set];classes=PAIRS[pair]
    x=normalize(data_train['features'],scaler);vx=normalize(data_val['features'],scaler)
    neighbors=normalize(data_train['neighbors'],scaler);valid=data_train['neighbor_valid'];y=class_indices(data_train['labels'],classes)
    model=FeatureEncoder(len(features),dim,len(classes));optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    tx=torch.from_numpy(x);ty=torch.from_numpy(y);tn=torch.from_numpy(neighbors)
    pools=balanced_pools(y,data_train['cases']);steps=int(np.ceil(len(x)/config['batch_size']))
    best=-1.;stale=0;history=[];start=time.perf_counter()
    for epoch in range(config['epochs']):
        rng=np.random.default_rng(seed+epoch*100003);model.train();sums=np.zeros(4)
        for _ in range(steps):
            index=np.concatenate([rng.choice(pool,size=config['batch_size']//len(pools),replace=False,p=w) for pool,w in pools]);rng.shuffle(index)
            optimizer.zero_grad(set_to_none=True);logits,z=model(tx[index]);ce=torch.nn.functional.cross_entropy(logits,ty[index])
            con=supcon_loss(z,ty[index],config['temperature']) if method!='CE' else z.sum()*0
            spatial=z.sum()*0
            if method.endswith('Spatial'):
                keep=np.flatnonzero(valid[index])
                if len(keep):_,nz=model(tn[index[keep]]);spatial=spatial_loss(z[keep],nz)
            loss=ce+config['lambda_contrastive']*con+config['lambda_spatial']*spatial
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step();sums+=np.array([loss.item(),ce.item(),con.item(),spatial.item()])
        probability,embedding=infer(model,vx)
        score=evaluate(data_val['labels'],probability,data_val['cases'],classes)
        # A true training-only progress F1; validation does not backpropagate.
        train_probability,_=infer(model,x)
        train_score=evaluate(data_train['labels'],train_probability,data_train['cases'],classes)
        metric=score['case_macro']['macro_f1_mean']
        row={'epoch':epoch+1,**dict(zip(('loss','CE','SupCon','Spatial'),(sums/steps).tolist())),
             'train_case_f1':train_score['case_macro']['macro_f1_mean'],'validation_case_f1':metric,
             'embedding_std_mean':float(embedding.std(0).mean())}
        history.append(row)
        if metric>best+config['min_delta']:
            best=metric;stale=0
            saved={'format':'learnable_feature_03_v1','state_dict':model.state_dict(),'features':features,'classes':classes,'embedding_dim':dim,
                   'scaler':scaler,'training':{'train_cases':plan['train_cases'],'validation_cases':plan['validation_cases'],**config},
                   'split_sha256':plan['split_sha256'],'seed':seed,'method':method,'epoch':epoch+1,'validation_case_f1':metric}
            torch.save(saved,checkpoint)
        else:stale+=1
        if stale>=config['patience']:break
    model,saved=load_encoder(checkpoint,features)
    vp,vz=infer(model,vx);tp,tz=infer(model,x)
    if float(tz.std(0).mean())<1e-8:raise ValueError('Collapsed embedding: all training vectors identical')
    np.savez_compressed(directory/'predictions.npz',validation_prob=vp,validation_embedding=vz,train_prob=tp,train_embedding=tz)
    (directory/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    result={'pair':pair,'features':feature_set,'method':method,'embedding_dim':dim,'checkpoint':str(checkpoint.relative_to(OUT)),
            'checkpoint_sha256':fingerprint(checkpoint),'best_epoch':saved['epoch'],'epochs_run':len(history),'seconds':time.perf_counter()-start,
            'seed':seed,'validation':evaluate(data_val['labels'],vp,data_val['cases'],classes,vz),
            'train':evaluate(data_train['labels'],tp,data_train['cases'],classes,tz)}
    receipt.write_text(json.dumps(result,indent=2),encoding='utf-8');return result


def run():
    plan,records=freeze();seed_all(SEED)
    if not (OUT/'ablation_legacy.json').exists():legacy_ablation(plan)
    for role in ('train','validation'):
        for case in plan[role+'_cases']:
            extract_case(case,role,records[case],plan);print('DATA',role,case,flush=True)
    ablation=[];models=[];baselines=[]
    for p,(pair,classes) in enumerate(PAIRS.items()):
        for feature_set,features in SETS.items():
            train=assemble(plan,'train',p,features);val=assemble(plan,'validation',p,features)
            scaler=fit_scaler(train['features'],plan['train_cases'])
            x=normalize(train['features'],scaler);v=normalize(val['features'],scaler)
            prob=knn_probability(x,train['labels'],v,classes)
            result={'scope':'48 train/10 validation; seven Z positions','pair':pair,'features':feature_set,'columns':features,'scaler':scaler,
                    'validation':evaluate(val['labels'],prob,val['cases'],classes,v)}
            ablation.append(result)
            if feature_set not in plan['neural_inputs']:continue
            trainprob=knn_probability(x,train['labels'],x,classes)
            result['train']=evaluate(train['labels'],trainprob,train['cases'],classes,x)
            result['train_scope']='resubstitution; optimistic exact-self nearest neighbor; not generalization'
            baselines.append(result)
            directory=OUT/'baselines'/pair/feature_set[0];directory.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(directory/'predictions.npz',validation_prob=prob,validation_embedding=v,train_prob=trainprob,train_embedding=x)
            (directory/'scaler.json').write_text(json.dumps(scaler,indent=2),encoding='utf-8')
            for method in plan['methods']:
                for dim in plan['embedding_candidates']:
                    model=train_one(plan,pair,feature_set,method,dim,train,val,scaler);models.append(model)
                    print('MODEL',pair,feature_set,method,dim,model['best_epoch'],model['validation']['case_macro']['macro_f1_mean'],flush=True)
    (OUT/'ablation_48.json').write_text(json.dumps(ablation,indent=2),encoding='utf-8')
    (OUT/'results.json').write_text(json.dumps({'protocol':plan,'baselines':baselines,'models':models},indent=2),encoding='utf-8')
    print('TRAINING COMPLETE',len(models),'small feature encoders',flush=True)


if __name__=='__main__':run()
