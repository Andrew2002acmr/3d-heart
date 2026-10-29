"""Eight-class metrics with ignore excluded and cases as observational units."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from ct_lab.multiclass_core import CLASSES
HD95_RASTER_THRESHOLD=5_000_000


def classification(truth,prediction):
    truth=np.asarray(truth).ravel();prediction=np.asarray(prediction).ravel();keep=(truth>=0)&(truth<8)
    cm=np.bincount(truth[keep].astype(np.int64)*8+prediction[keep].astype(np.int64),minlength=64).reshape(8,8)
    tp=cm.diagonal();actual=cm.sum(1);pred=cm.sum(0);den=actual+pred
    recall=np.divide(tp,actual,out=np.zeros(8,float),where=actual>0);precision=np.divide(tp,pred,out=np.zeros(8,float),where=pred>0)
    f1=np.divide(2*tp,den,out=np.zeros(8,float),where=den>0)
    return {'macro_f1':float(f1.mean()),'foreground_macro_f1':float(f1[1:].mean()),'present_class_macro_f1':float(f1[actual>0].mean()) if actual.any() else 0.,
            'balanced_accuracy':float(recall[actual>0].mean()) if actual.any() else 0.,'confusion_matrix':cm.tolist(),'valid_voxels':int(keep.sum()),
            'classwise':{str(c):{'precision':float(precision[c]),'recall':float(recall[c]),'f1':float(f1[c]),'target':int(actual[c]),'predicted':int(pred[c]),
                               'one_vs_rest_FPR':float((pred[c]-tp[c])/max(1,cm.sum()-actual[c])),
                               'BG_to_class_FPR':float(cm[0,c]/max(1,actual[0])) if c else None} for c in CLASSES}}


def evaluate(data,probability,embedding=None):
    truth=np.where(data.get('valid',np.ones(len(probability),bool)),data['labels'],255);pred=np.argmax(probability,1)
    per={str(c):classification(truth[data['cases']==c],pred[data['cases']==c]) for c in np.unique(data['cases'])}
    result={'per_case':per,'pooled':classification(truth,pred),'case_macro':{}}
    for key in ('macro_f1','foreground_macro_f1','balanced_accuracy'):
        values=[s[key] for s in per.values()];result['case_macro'][key+'_mean']=float(np.mean(values));result['case_macro'][key+'_std']=float(np.std(values))
    if embedding is not None:
        from sklearn.metrics import silhouette_score
        rng=np.random.default_rng(20261029);ids=np.concatenate([rng.choice(np.flatnonzero(truth==c),min(64,int((truth==c).sum())),replace=False) for c in CLASSES])
        if len(np.unique(truth[ids]))>1:
            z=embedding[ids].astype(float);y=truth[ids];parts=[z[y==c] for c in CLASSES if (y==c).any()];mean=z.mean(0)
            within=sum(np.square(p-p.mean(0)).sum() for p in parts);between=sum(len(p)*np.square(p.mean(0)-mean).sum() for p in parts)
            result['geometry']={'silhouette':float(silhouette_score(z,y)),'fisher_ratio':float(between/max(within,1e-12)),
                                'raw_std_mean':float(embedding.std(0).mean()),'samples':len(ids),'space':'raw embedding, not PCA/UMAP'}
    return result


def bootstrap(values,seed=20261029):
    values=np.asarray(values,float);rng=np.random.default_rng(seed)
    samples=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
    return {'mean':float(values.mean()),'case_SD':float(values.std()),'case_bootstrap_CI95':np.quantile(samples,[.025,.975]).tolist(),'n_cases':len(values)}


def dense_metrics(prediction,target):
    valid=target<8;result=classification(target,prediction);classes={};structure=ndi.generate_binary_structure(3,1)
    fg=(target>0)&valid;distance=ndi.distance_transform_edt(~fg) if fg.any() else np.full(target.shape,np.inf)
    bins={}
    for lower,upper in ((0,2),(2,6),(6,20),(20,float('inf'))):
        bg=(target==0)&(distance>=lower)&(distance<upper);n=int(bg.sum())
        bins[f'{lower}-{upper}']={'background_voxels':n,'false_foreground_voxels':int((bg&(prediction>0)).sum()),'FPR':float((bg&(prediction>0)).sum()/max(1,n))}
    result['background_distance_bins_index_units']=bins
    for c in CLASSES:
        pred=(prediction==c)&valid;gt=(target==c);labeled,count=ndi.label(pred,structure);sizes=np.bincount(labeled.ravel())[1:]
        overlaps=np.bincount(labeled[gt],minlength=count+1)[1:]>0 if count else np.zeros(0,bool)
        entry={'target_voxels':int(gt.sum()),'prediction_voxels':int(pred.sum()),'dice':float(2*(pred&gt).sum()/max(1,pred.sum()+gt.sum())) if gt.any() else None,
               'components_6connected':int(count),'component_sizes_desc':sorted(sizes.tolist(),reverse=True),'false_components_no_GT_overlap':int((~overlaps).sum()),
               'false_component_voxels':int(sizes[~overlaps].sum()),'false_positive_voxels':int((pred&~gt).sum()),'HD95_index_units':None}
        if pred.any() and gt.any():
            # Ignore-adjacent interfaces are not valid surfaces for distance metrics.
            trusted=ndi.binary_erosion(valid,structure,border_value=0)
            ps=pred&~ndi.binary_erosion(pred,structure,border_value=0)&trusted;gs=gt&~ndi.binary_erosion(gt,structure,border_value=0)&trusted
            if ps.any() and gs.any():
                if ps.size>HD95_RASTER_THRESHOLD:
                    # Linear raster EDT avoids pathological far-query KD costs on fragmented full volumes.
                    union=ps|gs;bounds=[]
                    for a in range(3):
                        active=np.flatnonzero(np.any(union,axis=tuple(b for b in range(3) if b!=a)))
                        bounds.append(slice(int(active[0]),int(active[-1])+1))
                    pp=ps[tuple(bounds)];gg=gs[tuple(bounds)]
                    d1=ndi.distance_transform_edt(~gg)[pp];d2=ndi.distance_transform_edt(~pp)[gg]
                else:
                    pcoords=np.column_stack(np.nonzero(ps));gcoords=np.column_stack(np.nonzero(gs))
                    d1=cKDTree(gcoords).query(pcoords,k=1,workers=1)[0]
                    d2=cKDTree(pcoords).query(gcoords,k=1,workers=1)[0]
                entry['HD95_index_units']=float(np.percentile(np.concatenate([d1,d2]),95))
        classes[str(c)]=entry
    result['dense_classwise']=classes;result['connectivity']=6;result['HD95_units']='index grid, not millimeters; valid interior surfaces'
    return result
