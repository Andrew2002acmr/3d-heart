"""Read-only SHA/geometry verification of Stage06 and protected previous stages."""
import argparse
import nibabel as nib
from ct_lab.dynamics_core import *

def run(final=False):
    plan=protocol();assert fingerprint(CONFIG)==plan['config_SHA256']
    original=json.loads((OUT/'protected_files.json').read_text());assert original==protected_files(),'Protected source weights/config/split changed'
    previous=json.loads((OUT/'previous_files.json').read_text());current=previous_files()
    added=sorted(set(current)-set(previous));missing=sorted(set(previous)-set(current));changed=[p for p in previous.keys()&current.keys() if previous[p]!=current[p]]
    assert not added and not missing and not changed,dict(added=added,missing=missing,changed=changed)
    split,records=cohort();sources=0;caches=0;predictions=0
    for role in ['train','validation']:
        for case in split[role]:
            allowed(case)
            for suffix,key in [('image','image_SHA256'),('label','mask_SHA256')]:
                assert fingerprint(DATA/f'{case}_{suffix}.nii.gz')==records[case][key],case;sources+=1
            cache=OUT/'prepared256'/case;provenance=json.loads((cache/'provenance.json').read_text())
            assert provenance['source_image_SHA256']==records[case]['image_SHA256'] and provenance['source_mask_SHA256']==records[case]['mask_SHA256']
            for name,key in [('image.npy','image_SHA256'),('mask.npy','mask_SHA256')]:assert fingerprint(cache/name)==provenance[key],str(cache/name);caches+=1
    assert not any((OUT/'prepared256'/case).exists() for case in split['test'])
    for path in (OUT/'runs').glob('*/evaluation/*/*/metrics.json'):
        value=json.loads(path.read_text());case=value['case'];allowed(case)
        ckpt=path.parents[2].parent/(path.parents[1].name+'.pt');assert fingerprint(ckpt)==value['checkpoint_SHA256']
        source=nib.load(DATA/f'{case}_image.nii.gz');pred=nib.load(path.with_name('prediction.nii.gz'))
        assert pred.shape==source.shape and np.array_equal(pred.affine,source.affine),str(path);predictions+=1
        topology=path.with_name('topology.json')
        if topology.exists():
            t=json.loads(topology.read_text());assert t['confusion_matrix']==value['native_confusion'] and t['checkpoint_SHA256']==value['checkpoint_SHA256']
    result_path=OUT/'runs/baseline_s20261009_n48/result.json'
    if final:
        assert result_path.exists() and (OUT/'topology_completion.json').exists()
        result=json.loads(result_path.read_text());assert result['steps']==23220 and result['processed_examples']==371430 and result['unique_slices']==12381
        assert fingerprint(result_path.parent/'selected.pt')==result['selected_checkpoint_SHA256']
        assert predictions>=114
    receipt={'final':final,'protected_files_unchanged':len(original),'previous_files_unchanged':len(previous),'previous_added':added,'previous_missing':missing,'previous_changed':changed,
      'train_dev_source_files_verified':sources,'cache_files_verified':caches,'native_prediction_headers_verified':predictions,'sealed_test_read':False,'all_checks_passed':True}
    write(OUT/('final_integrity.json' if final else 'interim_integrity.json'),receipt);print('DYNAMICS INTEGRITY',receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();run(a.final)
