import inspect,json
import numpy as np
import torch
import pytest
from ct_lab.dynamics_core import OUT,REPO,CONFIG,cohort,training_cases,atomic_checkpoint,seed_runtime
from ct_lab.dynamics_evaluate import predict_batch_native
from heart3d.ml.cardiac_data import CardiacDataset,index_transform,restore_probability_plane,resize_plane,confusion_counts,scores_from_confusion
from heart3d.ml.model import HeartUNet25D
from heart3d.ml.losses import MulticlassSegmentationLoss

def test_cross_platform_audit_preserves_frozen_export_bytes(tmp_path):
    from ct_lab.dynamics_core import write_if_changed,write_text_if_changed
    config=tmp_path/'frozen.json';config.write_bytes(b'{"value": 1}\r\n');before=config.read_bytes()
    assert not write_if_changed(config,{'value':1}) and config.read_bytes()==before
    source=tmp_path/'exported.py';source.write_bytes(b'value = 1\r\n');before=source.read_bytes()
    assert not write_text_if_changed(source,'value = 1\n') and source.read_bytes()==before
    assert write_if_changed(config,{'value':2}) and json.loads(config.read_text())=={'value':2}

def test_repeated_manual_preparation_preserves_both_exported_JSON_files(tmp_path,monkeypatch):
    import ct_lab.dynamics_suite as suite
    from ct_lab.dynamics_core import protocol
    plan=protocol();monkeypatch.setattr(suite,'protocol',lambda:plan)
    monkeypatch.setattr(suite,'REPO',tmp_path);monkeypatch.setattr(suite,'OUT',tmp_path/'outputs')
    suite.prepare();paths=[tmp_path/'configs/experiments/training_dynamics_06_runpod.json',tmp_path/'outputs/manual_runpod_plan.json']
    for path in paths:path.write_bytes(path.read_text(encoding='utf-8').replace('\n','\r\n').encode('utf-8'))
    before=[p.read_bytes() for p in paths];suite.prepare();assert before==[p.read_bytes() for p in paths]

def test_historical_budget_and_optimizer_evidence():
    audit=json.loads((OUT/'training_audit.json').read_text())
    for row in audit['historical']:
        assert row['samples_per_epoch']==12381 and row['configured_total_steps']==23220 and row['configured_total_examples']==371430
        assert row['checkpoint_optimizer_steps']==[row['checkpoint_epoch']*774]
        assert row['optimizer_groups'][0]['weight_decay']==0 and row['config']['batch_size']==16
        assert all(r['working_tree_matches_historical'] for r in row['source_comparison'] if r['path'].endswith(('model.py','losses.py','augmentation.py','runtime.py')))

def test_nested_groups_are_training_only_and_reproducible():
    split,rows=cohort();previous=set()
    for n in [8,16,24,32,48]:
        selected=training_cases(n);assert len(selected)==n and previous<=set(selected)<=set(split['train'])
        assert selected==training_cases(n) and not set(selected)&set(split['validation']+split['test']);previous=set(selected)

def test_sealed_dataset_rejected_before_any_cache_read():
    with pytest.raises(ValueError,match='sealed'):CardiacDataset(CONFIG,'test',OUT)

def test_prepared_cache_matches_original_CT_preprocessing():
    import nibabel as nib
    from ct_lab.dynamics_core import DATA
    if not (OUT/'preparation.json').exists():pytest.skip('Preparation not complete')
    case='ct_1001';z=110;source=nib.load(DATA/f'{case}_image.nii.gz');target=nib.load(DATA/f'{case}_label.nii.gz')
    t=index_transform(source.shape,256);raw=np.asarray(source.dataobj[:,:,z],np.float32);expected=np.clip(resize_plane(raw,t,1,0)/2015,0,1)*2-1
    actual=np.load(OUT/'prepared256'/case/'image.npy',mmap_mode='r');assert actual.dtype==np.float32
    np.testing.assert_array_equal(actual[z],expected)
    gt=resize_plane(np.asanyarray(target.dataobj[:,:,z]),t,0,255);gt=np.where(gt<=7,gt,255).astype(np.uint8)
    np.testing.assert_array_equal(np.load(OUT/'prepared256'/case/'mask.npy',mmap_mode='r')[z],gt)

def test_native_probability_inverse_before_argmax_and_no_GT_inputs():
    assert not any(p in inspect.signature(predict_batch_native).parameters for p in ['mask','target','GT'])
    generator=torch.Generator().manual_seed(44);logits=torch.randn(3,8,32,32,generator=generator)
    class Fixed(torch.nn.Module):
        def forward(self,x):return logits
    t=index_transform((17,19,3),32);prepared,native=predict_batch_native(Fixed(),np.zeros((3,5,32,32),np.float32),t)
    probability=logits.softmax(1).numpy();expected=np.stack([restore_probability_plane(p,t).argmax(0).T for p in probability])
    np.testing.assert_array_equal(native,expected);np.testing.assert_array_equal(prepared,logits.argmax(1).numpy())

def test_identical_ignore_and_absent_class_policy():
    target=np.array([1,1,0,255]);pred=np.array([1,2,2,7]);r=scores_from_confusion(confusion_counts(pred,target))
    assert r['macro_foreground_Dice']==2/3 and r['classes']['PA']['target_voxels']==0 and r['classes']['PA']['predicted_voxels']==0

def test_atomic_training_resume_CPU_equivalence(tmp_path):
    snapshots=[]
    x=torch.rand(2,5,32,32,generator=torch.Generator().manual_seed(2));y=torch.randint(0,8,(2,32,32),generator=torch.Generator().manual_seed(3));criterion=MulticlassSegmentationLoss()
    for resume in (False,True):
        seed_runtime(20261009,1,'cpu');model=HeartUNet25D(input_channels=5,encoder_channels=(16,32,64,128),norm_groups=8,output_channels=8);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        for step in range(4):
            optimizer.zero_grad();loss=criterion(model(x),y);loss.backward();optimizer.step()
            if resume and step==1:
                path=tmp_path/'last.pt';atomic_checkpoint(path,{'model':model.state_dict(),'optimizer':optimizer.state_dict()});saved=torch.load(path,weights_only=True)
                fresh=HeartUNet25D(input_channels=5,encoder_channels=(16,32,64,128),norm_groups=8,output_channels=8);fresh.load_state_dict(saved['model']);model=fresh
                optimizer=torch.optim.Adam(model.parameters(),lr=.001);optimizer.load_state_dict(saved['optimizer'])
        snapshots.append({k:v.detach().clone() for k,v in model.state_dict().items()})
    for key in snapshots[0]:assert torch.equal(snapshots[0][key],snapshots[1][key])

def test_sampling_order_matches_original_DataLoader_and_train_guard():
    from torch.utils.data import DataLoader
    from ct_lab.dynamics_sampling import historical_order,sampling_plan
    for epoch in [0,1,7]:
        actual=[int(i) for batch in DataLoader(range(101),batch_size=16,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(20261009+epoch)) for i in batch]
        assert actual==historical_order(101,20261009,epoch)
    class Invalid:
        partition='train';rows=[{'case_id':cohort()[0]['validation'][0]}]
        def set_epoch(self,epoch):raise AssertionError('Validation candidate extraction occurred')
    with pytest.raises(ValueError,match='Validation'):sampling_plan(Invalid(),'historical',1,0,{})

def test_sampler_reproducibility_exposure_and_case_balance():
    from ct_lab.dynamics_sampling import sampling_plan
    data=CardiacDataset(CONFIG,'train',OUT,augmentation=False);stats={r['case']:r for r in json.loads((OUT/'case_statistics.json').read_text())}
    for variant in ['historical','ROI05_pool','equal_case','foreground_boost']:
        first,length=sampling_plan(data,variant,20261009,0,stats);a=[data.indices[i] for i in first]
        second,length2=sampling_plan(data,variant,20261009,0,stats);b=[data.indices[i] for i in second]
        assert length==length2==12381 and a==b
        if variant!='historical':
            from collections import Counter
            counts=Counter(c for c,z in a);assert max(counts.values())-min(counts.values())<=1
    data.close()

def test_ablation_losses_finite_ignore_gradients_and_train_only_weights():
    from ct_lab.dynamics_losses import AdditionalLoss,train_weights
    cases=training_cases(8);stats=json.loads((OUT/'case_statistics.json').read_text());weights=train_weights(cases,stats)
    changed=json.loads(json.dumps(stats))
    for r in changed:
        if r['role']=='validation':r['prepared_class_counts']=[999999999]*8
    assert weights==train_weights(cases,changed)
    with pytest.raises(ValueError):train_weights([cohort()[0]['validation'][0]],stats)
    for criterion in [MulticlassSegmentationLoss('ce'),MulticlassSegmentationLoss('ce_dice'),AdditionalLoss('weighted_ce_dice',weights['weights']),AdditionalLoss('focal_dice')]:
        logits=torch.randn(2,8,16,16,requires_grad=True);target=torch.randint(0,8,(2,16,16));target[:,:4]=255
        loss=criterion(logits,target);assert torch.isfinite(loss);loss.backward();assert torch.count_nonzero(logits.grad[:,:,:4])==0
        empty=torch.full_like(target,255);assert criterion(logits,empty)==0

def test_manual_matrix_is_complete_and_preserves_isolation():
    from ct_lab.dynamics_suite import jobs
    matrix=jobs();assert len(matrix)==33 and sum(j['steps'] for j in matrix)==301860
    assert [j['group'] for j in matrix[:3]]==['baseline']*3
    for j in matrix:
        assert j['steps']==(23220 if j['family']=='baseline' else 7740)
        if j['family']=='sampling':assert j['n']==48 and j['loss_name']=='ce_dice'
        if j['family']=='loss':assert j['n']==48 and j['sampling']=='historical'
        if j['family']=='data':assert j['sampling']=='historical' and j['loss_name']=='ce_dice'
    for family in ['baseline','sampling','loss','data']:
        grouped={}
        for j in matrix:
            if j['family']==family:grouped.setdefault((j['n'],j['sampling'],j['loss_name']),set()).add(j['seed'])
        assert all(len(seeds)==3 for seeds in grouped.values())

def test_immutable_checkpoint_lookup_and_remote_raw_fallback(tmp_path,monkeypatch):
    from ct_lab.dynamics_ui import native_directory
    p=OUT/'runs/baseline_s20261009_n48'
    assert native_directory(p/'step000480.pt','ct_1001')==p/'evaluation/step000480/ct_1001'
    assert native_directory(p/'selected.pt','ct_1001')==p/'evaluation/selected/ct_1001'
    # A downloaded run can have metrics/weights without the large prepared cache.
    # Exercise that path with real NIfTI geometry and a CPU predictor.
    import ct_lab.dynamics_ui as ui
    import nibabel as nib
    source=tmp_path/'ct_1001_image.nii.gz';nib.save(nib.Nifti1Image(np.ones((9,11,3),np.float32),np.diag([2.,3.,4.,1.])),source)
    class Fixed(torch.nn.Module):
        def forward(self,x):
            result=torch.zeros(len(x),8,x.shape[-2],x.shape[-1]);result[:,2]=1;return result
    pre={'input_size':256,'clip_source_intensity':[0,2015],'context_offsets_indices':[-2,-1,0,1,2]}
    monkeypatch.setattr(ui,'OUT',tmp_path/'remote_outputs');monkeypatch.setattr(ui,'DATA',tmp_path)
    monkeypatch.setattr(ui,'cached_model',lambda *a:(Fixed(),{},pre));before=sorted(tmp_path.rglob('*'))
    result,scope=ui.checkpoint_plane.__wrapped__(str(p/'step000480.pt'),'test','ct_1001',0)
    assert result.shape==(9,11) and np.all(result==2) and 'CPU single-slice' in scope
    assert before==sorted(tmp_path.rglob('*'))
