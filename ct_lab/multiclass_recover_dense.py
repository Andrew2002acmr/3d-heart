"""Verify completed pre-receipt dense maps before resuming expensive diagnostics."""
import json
import numpy as np
import nibabel as nib
from ct_lab.multiclass_core import *
from ct_lab.multiclass_dense import inference_signature


def run():
    plan,records=freeze();case=plan['dense_full_volume_case'];job={**plan['dense_full_model'],'seed':plan['dense_seed']}
    from ct_lab.multiclass_train import slug
    name=slug(job);directory=OUT/'dense_full'/case/name;source=DEFAULT_DATA/f'{case}_image.nii.gz'
    assert fingerprint(source)==records[case]['image_SHA256']
    image=nib.load(source);prob=np.load(directory/'probabilities.npy',mmap_mode='r');pred=np.load(directory/'prediction.npy',mmap_mode='r');entropy=np.load(directory/'entropy.npy',mmap_mode='r')
    assert prob.shape==(*image.shape,8) and pred.shape==image.shape and entropy.shape==image.shape
    for z in range(0,image.shape[2],4):
        p=prob[:,:,z:z+4].astype(np.float32);r=pred[:,:,z:z+4];e=entropy[:,:,z:z+4]
        assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1 and r.max()<8
        np.testing.assert_allclose(p.sum(-1),1,atol=.002)
        assert np.isfinite(e).all() and e.min()>=-1e-6 and e.max()<=np.log(8)+1e-5
        chosen=np.take_along_axis(p,r[...,None],axis=-1)[...,0]
        np.testing.assert_allclose(chosen,p.max(-1),atol=.0005)
    seed_all(SEEDS[0]);model,saved=load_model(OUT/'models'/name/'best.pt');volume=image.get_fdata(dtype=np.float32)
    ids=np.random.default_rng(SEEDS[0]).choice(volume.size,512,replace=False);xyz=np.column_stack(np.unravel_index(ids,volume.shape))
    values=columns(extract_points(volume,xyz,image.affine),saved['features']);p,_=infer(model,normalize(values,saved['scaler']))
    np.testing.assert_allclose(prob[tuple(xyz.T)].astype(np.float32),p,atol=.001,rtol=.002)
    write(directory/'inference.json',{'complete':True,'signature':inference_signature(model,saved['scaler'],saved['features'],[0,0,0],image.shape,image.shape),
                                    'recovery_audit':'ALL stored rows finite/sum/entropy/argmax, plus 512 independently re-extracted source-point predictions',
                                    'source_image_sha256':records[case]['image_SHA256']})
    print('DENSE MAPS VERIFIED: all voxels, independent source points; safe diagnostic resume',flush=True)


if __name__=='__main__':run()
