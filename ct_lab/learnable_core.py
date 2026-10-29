"""Stage 03: small per-voxel feature encoder, NOT a segmentation architecture."""
import os
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import ct_lab.spatial_features
REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'outputs/learnable_feature_03'
os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache')
SEED=20261028
PAIRS={'LV_MYO':[1,5],'RV_PA':[2,7]}
CORE=['Intensity','3D Mean','3D Std','3D Gradient','3D LoG']
GAUSS=[f'3D Gaussian {s:g}' for s in (.5,1.,2.)]
SETS={'A Intensity':['Intensity'],'B 2D':['Intensity','XY Gradient','XY Mean','XY Std','XY LoG'],
      'C 3D-core':CORE,'D core+G0.5':CORE+GAUSS[:1],'E core+G1':CORE+GAUSS[1:2],
      'F core+G2':CORE+GAUSS[2:3],'G core+all':CORE+GAUSS}
FEATURES=list(dict.fromkeys(n for names in SETS.values() for n in names))


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)


class FeatureEncoder(nn.Module):
    def __init__(self,input_dim,embedding_dim=16,classes=2):
        super().__init__()
        if input_dim<1 or embedding_dim not in (8,16,32) or classes<2:raise ValueError('Invalid encoder dimensions')
        self.linear1=nn.Linear(input_dim,64);self.linear2=nn.Linear(64,32)
        self.embedding=nn.Linear(32,embedding_dim);self.head=nn.Linear(embedding_dim,classes)

    def forward(self,x,trace=False):
        first=self.linear1(x);relu1=F.relu(first);second=self.linear2(relu1);relu2=F.relu(second)
        embedding=self.embedding(relu2);logits=self.head(embedding)
        if trace:return {'standardized_input':x,'linear1':first,'relu1':relu1,'linear2':second,'relu2':relu2,
                         'embedding':embedding,'logits':logits,'probabilities':logits.softmax(1)}
        return logits,embedding


def supcon_loss(embedding,labels,temperature=.1):
    """All same-class nonself anchors are positives; denominator excludes self.

    Rows without positives contribute no loss. Label numbers never become input
    features. At least two classes should be supplied by the balanced sampler.
    """
    if temperature<=0:raise ValueError('Positive temperature required')
    if len(embedding)<2:return embedding.sum()*0
    normalized=F.normalize(embedding,dim=1,eps=1e-12)
    logits=normalized@normalized.T/temperature
    eye=torch.eye(len(embedding),device=embedding.device,dtype=torch.bool)
    positive=(labels[:,None]==labels[None,:])&~eye
    log_probability=logits-torch.logsumexp(logits.masked_fill(eye,-torch.inf),dim=1,keepdim=True)
    count=positive.sum(1);valid=count>0
    per_anchor=-(log_probability.masked_fill(~positive,0).sum(1)/count.clamp_min(1))
    return per_anchor[valid].mean() if valid.any() else embedding.sum()*0


def spatial_loss(anchor,neighbor):
    if len(anchor)==0:return anchor.sum()*0
    return (F.normalize(anchor,dim=1,eps=1e-12)-F.normalize(neighbor,dim=1,eps=1e-12)).square().sum(1).mean()


def admissible_neighbors(xyz,labels,mask,rng):
    """Train loss pairs only: true 6-neighbors, in bounds, same GT class.

    A missing neighbor is -1 with valid=False, never a replicated/self voxel.
    Validation calls must not invoke this function (enforced in data pipeline).
    """
    offsets=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    result=np.full_like(xyz,-1);valid=np.zeros(len(xyz),bool)
    for i,(point,label) in enumerate(zip(xyz,labels)):
        candidates=point+offsets
        candidates=candidates[np.all((candidates>=0)&(candidates<np.array(mask.shape)),axis=1)]
        candidates=candidates[mask[tuple(candidates.T)]==label]
        if len(candidates):result[i]=candidates[rng.integers(len(candidates))];valid[i]=True
    return result,valid


def class_indices(labels,classes):
    labels=np.asarray(labels);result=np.full(len(labels),-1,np.int64)
    for i,c in enumerate(classes):result[labels==c]=i
    if np.any(result<0):raise ValueError('Label outside declared class mapping')
    return result


def normalize(values,scaler):
    return ((np.asarray(values,np.float32)-np.asarray(scaler['mean'],np.float32))/np.asarray(scaler['scale'],np.float32)).astype(np.float32)


def fit_scaler(train,train_cases):
    x=np.asarray(train,np.float64);mean=x.mean(0);std=x.std(0);std[std<1e-12]=1
    return {'mean':mean.tolist(),'scale':std.tolist(),'fit_cases':list(train_cases),'fit_partition':'train','count':len(x)}


def infer(model,values,batch=4096):
    model.eval();probabilities=[];embeddings=[]
    with torch.inference_mode():
        for start in range(0,len(values),batch):
            logits,z=model(torch.from_numpy(np.asarray(values[start:start+batch],np.float32)))
            probabilities.append(logits.softmax(1).numpy());embeddings.append(z.numpy())
    return np.concatenate(probabilities),np.concatenate(embeddings)


def load_encoder(path,expected_features=None):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format')!='learnable_feature_03_v1':raise ValueError('Not a feature encoder checkpoint')
    features=saved['features'];classes=saved['classes'];config=saved['training']
    if not set(features)<=set(FEATURES) or len(set(features))!=len(features):raise ValueError('Checkpoint includes forbidden or duplicate input features')
    if len(set(classes))!=len(classes) or not set(classes)<=set(range(8)):raise ValueError('Invalid source class mapping')
    if expected_features is not None and features!=list(expected_features):raise ValueError('Input feature schema differs')
    if saved['scaler']['fit_partition']!='train' or set(saved['scaler']['fit_cases'])!=set(config['train_cases']):raise ValueError('Checkpoint scaler provenance differs')
    if set(config['train_cases'])&set(config['validation_cases']):raise ValueError('Checkpoint group leakage')
    from ct_lab.core import fingerprint
    if saved['split_sha256']!=fingerprint(REPO/'configs/splits/cardiac_chd68_v1.json'):raise ValueError('Frozen split changed')
    frozen=json.loads((REPO/'configs/splits/cardiac_chd68_v1.json').read_text())['partitions']
    if set(config['train_cases'])!=set(frozen['train']) or set(config['validation_cases'])!=set(frozen['validation']):raise ValueError('Checkpoint cases differ from frozen 48/10 split')
    model=FeatureEncoder(len(features),saved['embedding_dim'],len(classes));model.load_state_dict(saved['state_dict'],strict=True);model.eval()
    return model,saved
