"""Tiny synthetic integration coverage; no clinical data or quality experiment."""
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
import torch
from heart3d.ml.cardiac_data import (CardiacDataset,write_json,read_json,load_protocol,index_transform,
                                     resize_plane,restore_probability_plane,index_neighbors,confusion_counts,scores_from_confusion)
from heart3d.ml.cardiac_train import benchmark,train,evaluate,check_benchmark
from heart3d.storage import sha256_file


@pytest.fixture
def cardiac_toy(tmp_path):
    root=tmp_path/'data';root.mkdir();cohort_path=tmp_path/'cohort.json';split_path=tmp_path/'split.json'
    pre_path=tmp_path/'pre.json';config_path=tmp_path/'config.json';rows=[]
    partitions={'train':['a','b'],'validation':['c'],'test':['d']}
    for number,case in enumerate(['a','b','c','d'],1):
        mask=np.broadcast_to(np.arange(8,dtype=np.uint8)[None,:,None],(8,8,8)).copy()
        gt=root/f'{case}_gt.nii.gz';nib.save(nib.Nifti1Image(mask,np.eye(4)),gt)
        rows.append({'case_id':case,'group_id':case,'image_SHA256':f'{number:064x}',
                     'mask_SHA256':sha256_file(gt),'mask_relative_path':gt.name})
    write_json(cohort_path,{'records':rows})
    write_json(split_path,{'cohort_manifest_SHA256':sha256_file(cohort_path),'partitions':partitions,'development_cases':['a'],'frozen':True})
    pre={'fitted_partition':'train','split_SHA256':sha256_file(split_path),'cohort_SHA256':sha256_file(cohort_path),
         'context_offsets_indices':[-2,-1,0,1,2]}
    write_json(pre_path,pre);files=[]
    for partition,cases in partitions.items():
        for case in cases:
            row=next(r for r in rows if r['case_id']==case);directory=root/'cache'/case;directory.mkdir(parents=True)
            mask=np.asanyarray(nib.load(root/row['mask_relative_path']).dataobj).transpose(2,1,0).copy()
            np.save(directory/'mask.npy',mask)
            np.save(directory/'image.npy',(mask/3.5-1).astype(np.float32))
            p={'case_id':case,'partition':partition,'transform':index_transform((8,8,8),8),'positive_indices':list(range(8)),
               'negative_indices':[],'original_affine':np.eye(4).tolist(),'source_image_SHA256':row['image_SHA256'],
               'source_mask_SHA256':row['mask_SHA256'],'split_SHA256':sha256_file(split_path),
               'cohort_SHA256':sha256_file(cohort_path),'preprocessing_SHA256':sha256_file(pre_path)}
            write_json(directory/'provenance.json',p)
            for name in ['image.npy','mask.npy','provenance.json']:
                path=directory/name;files.append({'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,'SHA256':sha256_file(path)})
    write_json(root/'cache/prepared_manifest.json',{'files':files,'cohort_SHA256':sha256_file(cohort_path),
               'split_SHA256':sha256_file(split_path),'preprocessing_SHA256':sha256_file(pre_path)})
    config={'data_root':str(root),'cohort':str(cohort_path),'split':str(split_path),'preprocessing':str(pre_path),
            'cache_relative_path':'cache','outputs_relative_path':'runs','checkpoint_relative_path':'checkpoints',
            'reserve_GB':0,'model':{'input_channels':5,'encoder_channels':[4,8,16,32],'norm_groups':4,'output_channels':8},
            'batch_size':2,'cpu_threads':1,'num_workers':0,'seed':20261009,'learning_rate':.005,'epochs':2,
            'loss':{'variant':'ce_dice','ignore_index':255},'benchmark_warmup_batches':0,'sanity_batches':15,
            'augmentation':{'rotation_degrees':0,'scale_delta':0,'intensity_shift':0,'noise_std':0,'flips':False}}
    write_json(config_path,config);return config_path


def test_index_fit_pad_inverse_and_no_physical_claim():
    t=index_transform((12,6,5),12);plane=np.arange(72,dtype=np.float32).reshape(12,6)
    resized=resize_plane(plane,t,1,-1)
    assert np.allclose(restore_probability_plane(resized[None],t)[0],plane)
    assert not t['physical_geometry_used'] and not t['z_resampling']
    assert resized.shape==(12,12) and (resized[:3]==-1).all()
    indices,outside=index_neighbors(0,5,[-2,-1,0,1,2])
    assert indices.tolist()==[0,0,0,1,2] and outside.tolist()==[True,True,False,False,False]


def test_case_and_group_isolation_development_exclusion(cardiac_toy):
    config=read_json(cardiac_toy);rows,split=load_protocol(config)
    assert len(rows)==4
    split['partitions']['test']=['a'];write_json(config['split'],split)
    with pytest.raises(ValueError,match='leakage'):load_protocol(config)


def test_dataset_lazy_context_augmentation_and_deterministic_seed(cardiac_toy):
    config=read_json(cardiac_toy)
    a=CardiacDataset(cardiac_toy);b=CardiacDataset(cardiac_toy)
    x=a[0];same=b[0]
    assert x['image'].shape==(5,8,8) and x['target'].shape==(8,8)
    assert x['target'].dtype==torch.long and torch.equal(x['image'],same['image'])
    assert x['context_outside_source'].tolist()==[True,True,False,False,False]
    assert set(c for c,_ in a.indices)=={'a','b'}
    validation=CardiacDataset(cardiac_toy,'validation')
    assert not validation.augmentation and set(c for c,_ in validation.indices)=={'c'}
    # Unspecified voxels must survive spatial augmentation as ignore=255.
    a.close();b.close()
    path=Path(config['data_root'])/'cache/a/mask.npy';m=np.load(path);m[:,0,:]=255;np.save(path,m)
    a=CardiacDataset(cardiac_toy)
    assert (a[0]['target'][0]==255).all()
    a.close();validation.close()


def test_multiclass_counts_ignore_and_empty_class_policy():
    counts=confusion_counts(np.array([1,1,7]),np.array([1,2,255]))
    report=scores_from_confusion(counts)
    assert report['classes']['LV']['Dice']==pytest.approx(2/3)
    assert report['classes']['RV']['Dice']==0
    assert report['classes']['PA']['Dice'] is None
    assert report['macro_foreground_Dice']==pytest.approx(1/3)


def test_tiny_benchmark_train_original_grid_evaluation(cardiac_toy):
    config=read_json(cardiac_toy);root=Path(config['data_root'])
    bench=benchmark(cardiac_toy,root/'bench',batches=3,device='cpu',require_cuda=False)
    assert bench['sanity_passed'] and not bench['test_used'] and not bench['validation_used']
    with pytest.raises(ValueError,match='CUDA'):check_benchmark(root/'bench/summary.json',{},'cpu')
    with pytest.raises(ValueError,match='Explicit'):train(cardiac_toy,'denied',None,device='cpu',require_cuda=False)
    result=train(cardiac_toy,'toy',None,allow_full_training=True,device='cpu',require_cuda=False)
    assert result['epochs']==2 and not result['test_evaluated']
    best=root/'checkpoints/toy/best.pt';checkpoint=torch.load(best,weights_only=True)
    assert checkpoint['training_scope']=='public_chd68_multiclass_scratch_v1'
    assert (root/'checkpoints/toy/last.pt').exists()
    before=sha256_file(root/'d_gt.nii.gz')
    evaluation=evaluate(cardiac_toy,best,root/'evaluation',device='cpu',require_cuda=False)
    assert evaluation['case_count']==1 and not evaluation['HD95_ASSD_mm_calculated']
    output=nib.load(root/'evaluation/d/cardiac_prediction_original.nii.gz');gt=nib.load(root/'d_gt.nii.gz')
    assert output.shape==gt.shape and np.array_equal(output.affine,gt.affine)
    assert output.header.get_xyzt_units()[0]=='unknown' and sha256_file(root/'d_gt.nii.gz')==before
    with pytest.raises(ValueError,match='exists'):evaluate(cardiac_toy,best,root/'evaluation',device='cpu',require_cuda=False)



def test_training_archive_verifies_payloads_and_refuses_corrupt_receipt(tmp_path):
    from scripts.verify_training_archive import verify_archive
    from heart3d.ml.bundle import pack
    root=tmp_path/'source';root.mkdir();(root/'array.bin').write_bytes(b'public training cache'*100)
    archive=tmp_path/'bundle.tar.gz';manifest=tmp_path/'manifest.json'
    pack(root,['array.bin'],archive,manifest)
    receipt=archive.with_suffix('.receipt.json')
    assert verify_archive(archive,receipt)['payload_files_verified']==1
    corrupt=read_json(receipt);corrupt['archive_SHA256']='0'*64;write_json(receipt,corrupt)
    with pytest.raises(ValueError,match='SHA'):verify_archive(archive,receipt)


def test_protocol_json_is_byte_identical_across_windows_linux(tmp_path):
    path=tmp_path/'protocol.json'
    write_json(path,{'value':'КТ','shape':[256,256]})
    raw=path.read_bytes()
    assert b'\r\n' not in raw and raw.endswith(b'\n')
    assert read_json(path)['value']=='КТ'
