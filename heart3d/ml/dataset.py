"""Lazy memory-mapped axial 2.5D samples with explicit physical context."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from heart3d.ml.augmentation import augment
from heart3d.ml.geometry import physical_context
from heart3d.ml.splits import load_frozen_split
from heart3d.storage import sha256_file


class HeartDataset(Dataset):
    def __init__(self,config_path,partition='train',data_root=None,augmentation=True):
        self.config=json.loads(Path(config_path).read_text())
        self.root=Path(data_root or self.config['data_root']);self.partition=partition
        self.split=load_frozen_split(self.config['split'],self.config['cohort'])
        self.preproc=json.loads(Path(self.config['preprocessing']).read_text())
        if self.preproc['fitted_partition']!='train' or self.preproc['split_SHA256']!=sha256_file(self.config['split']):
            raise ValueError('Preprocessing provenance differs from this split')
        self.rows=self.split['partitions'][partition];self.provenance={};self.opened=OrderedDict()
        self.augmentation=bool(augmentation and partition=='train')
        for row in self.rows:
            directory=self.root/self.config['cache_relative_path']/row['patient_id']
            provenance=json.loads((directory/'provenance.json').read_text())
            if provenance['patient_id']!=row['patient_id'] or provenance['partition']!=partition:
                raise ValueError('Cache patient/partition leakage')
            if provenance['split_SHA256']!=sha256_file(self.config['split']) or provenance['preprocessing_SHA256']!=sha256_file(self.config['preprocessing']):
                raise ValueError('Stale preprocessing cache')
            self.provenance[row['patient_id']]=provenance
        self.set_epoch(0)

    def set_epoch(self,epoch):
        self.epoch=epoch;self.indices=[]
        for row in self.rows:
            pid=row['patient_id'];p=self.provenance[pid]
            positive=p['positive_indices'];negative=p['negative_indices']
            if self.partition=='train' and self.config['sampling']=='all_positive_equal_negative_per_patient':
                seed=int.from_bytes(hashlib.sha256(f"{self.config['seed']}|{epoch}|{pid}".encode()).digest()[:8],'little')
                rng=np.random.default_rng(seed)
                chosen=rng.choice(negative,size=min(len(positive),len(negative)),replace=False).tolist()
                centers=sorted(positive+chosen)
            else:centers=list(range(len(p['transform']['z_positions_mm'])))
            self.indices.extend((pid,i) for i in centers)

    def _arrays(self,pid):
        if pid not in self.opened:
            directory=self.root/self.config['cache_relative_path']/pid
            self.opened[pid]=(np.load(directory/'image.npy',mmap_mode='r'),np.load(directory/'mask.npy',mmap_mode='r'))
            while len(self.opened)>2:
                _,arrays=self.opened.popitem(last=False)
                for array in arrays:array._mmap.close()
        self.opened.move_to_end(pid)
        return self.opened[pid]

    def __len__(self):return len(self.indices)

    def __getitem__(self,index):
        pid,center=self.indices[index];image,mask=self._arrays(pid)
        positions=self.provenance[pid]['transform']['z_positions_mm']
        low,high,weight,outside=physical_context(positions,center,self.preproc['context_offsets_mm'])
        context=np.stack([(1-w)*image[a]+w*image[b] for a,b,w in zip(low,high,weight)]).astype(np.float32)
        x=torch.from_numpy(context);target=torch.from_numpy(np.array(mask[center: center+1],dtype=np.float32))
        if self.augmentation:
            generator=torch.Generator().manual_seed(self.config['seed']+self.epoch*1000003+index)
            x,target=augment(x,target,self.config['augmentation'],generator)
        return {'image':x,'target':target,'patient_id':pid,'center_index':center,
            'center_position_mm':positions[center],'context_outside_scan':torch.from_numpy(outside)}
