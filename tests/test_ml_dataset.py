import json
from pathlib import Path
import numpy as np
import pytest
import torch
from heart3d.ml.dataset import HeartDataset
from heart3d.ml.augmentation import augment
from heart3d.ml.geometry import make_transform
from heart3d.ml.splits import load_frozen_split
from heart3d.pediatric import write_json
from heart3d.storage import sha256_file


@pytest.fixture
def toy_experiment(tmp_path):
    rows=[{'patient_id':f'p{i}','review_status':'approved','geometry':{'physical_geometry_confirmed_from_DICOM':True}} for i in range(6)]
    cohort=tmp_path/'cohort.json';write_json(cohort,{'records':rows})
    split=tmp_path/'split.json';write_json(split,{'frozen':True,'split_unit':'patient',
        'cohort_manifest_SHA256':sha256_file(cohort),'development_patients_excluded_from_test':['p0'],
        'partitions':{'train':rows[:2],'validation':rows[2:4],'test':rows[4:]}})
    pre=tmp_path/'preprocessing.json';write_json(pre,{'fitted_partition':'train','split_SHA256':sha256_file(split),
        'context_offsets_mm':[-4,-2,0,2,4]})
    transform=make_transform((16,16,8),np.diag([-1.,-1.,2.,1.]),16,2.)
    image=np.stack([np.full((16,16),i*.1-1,dtype=np.float32) for i in range(8)])
    mask=np.zeros_like(image,dtype=np.uint8);mask[2:4,4:12,4:12]=1
    for name,partition in [('p0','train'),('p1','train'),('p2','validation'),('p3','validation'),('p4','test'),('p5','test')]:
        d=tmp_path/'cache'/name;d.mkdir(parents=True)
        np.save(d/'image.npy',image);np.save(d/'mask.npy',mask)
        write_json(d/'provenance.json',{'patient_id':name,'partition':partition,
            'preprocessing_SHA256':sha256_file(pre),'split_SHA256':sha256_file(split),
            'transform':transform,'positive_indices':[2,3],'negative_indices':[0,1,4,5,6,7]})
    config=tmp_path/'config.json';write_json(config,{'data_root':str(tmp_path),'split':str(split),
        'cohort':str(cohort),'preprocessing':str(pre),'cache_relative_path':'cache','seed':42,
        'sampling':'all_positive_equal_negative_per_patient',
        'augmentation':{'rotation_degrees':5,'scale_delta':.05,'intensity_shift':.03,'noise_std':.01}})
    return config,split,cohort


def test_lazy_index_physical_context_target_and_balanced_train(toy_experiment):
    config,_,_=toy_experiment;dataset=HeartDataset(config,augmentation=False)
    assert len(dataset)==8 and not dataset.opened
    index=dataset.indices.index(('p0',3));sample=dataset[index]
    assert sample['image'].shape==(5,16,16) and sample['target'].shape==(1,16,16)
    assert np.allclose(sample['image'][:,0,0],[-.9,-.8,-.7,-.6,-.5])
    assert sample['target'].sum()==64 and not sample['context_outside_scan'].any()
    assert all(isinstance(a,np.memmap) for a in dataset.opened['p0'])


def test_seed_reproducibility_epoch_and_validation_no_augmentation(toy_experiment):
    config,_,_=toy_experiment;a=HeartDataset(config);b=HeartDataset(config)
    assert a.indices==b.indices and torch.equal(a[0]['image'],b[0]['image'])
    a.set_epoch(1);b.set_epoch(1)
    assert a.indices==b.indices and torch.equal(a[0]['image'],b[0]['image'])
    validation=HeartDataset(config,'validation',augmentation=True)
    assert not validation.augmentation and len(validation)==16
    sample=validation[0]
    assert np.allclose(sample['image'][:,0,0],[-1,-1,-1,-.9,-.8])
    assert sample['context_outside_scan'].tolist()==[True,True,False,False,False]


def test_split_rejects_duplicate_patient_and_changed_cohort(toy_experiment):
    config,split,cohort=toy_experiment;data=json.loads(split.read_text())
    data['partitions']['test'][0]=data['partitions']['train'][0];write_json(split,data)
    with pytest.raises(ValueError,match='leakage'):load_frozen_split(split,cohort)
    write_json(cohort,{'records':[]})
    with pytest.raises(ValueError,match='differs'):load_frozen_split(split,cohort)


def test_json_hashes_have_platform_independent_lf(tmp_path):
    path=tmp_path/'manifest.json';write_json(path,{'value':'pediatric','rows':[1,2]})
    assert b'\r\n' not in path.read_bytes()


def test_spatial_augmentation_is_shared_across_context_and_binary_target():
    mask=torch.zeros(1,32,32);mask[:,8:24,8:24]=1
    image=(2*mask-1).repeat(5,1,1)
    settings={'rotation_degrees':5,'scale_delta':.05,'intensity_shift':0,'noise_std':0}
    transformed,target=augment(image,mask,settings,torch.Generator().manual_seed(20261006))
    assert all(torch.equal(transformed[0],channel) for channel in transformed)
    assert set(target.unique().tolist())=={0.,1.}
    foreground=transformed[0]>0;truth=target[0]>0
    assert (foreground & truth).sum()/(foreground | truth).sum()>.95
    with pytest.raises(ValueError,match='Flips'):
        augment(image,mask,dict(settings,flips=True),torch.Generator())
