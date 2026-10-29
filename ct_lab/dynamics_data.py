"""Historical full-FOV F32 preparation, strictly58 train/development cases."""
import gc,math
import nibabel as nib
from scipy import ndimage as ndi
from ct_lab.dynamics_core import *
from heart3d.ml.cardiac_data import index_transform,resize_plane,CardiacDataset

def prepare_case(case,role,row):
    allowed(case);directory=OUT/'prepared256'/case
    if (directory/'provenance.json').exists():return json.loads((directory/'statistics.json').read_text())
    image_path=DATA/f'{case}_image.nii.gz';mask_path=DATA/f'{case}_label.nii.gz'
    assert fingerprint(image_path)==row['image_SHA256'] and fingerprint(mask_path)==row['mask_SHA256']
    source=nib.load(image_path);mask=nib.load(mask_path);assert source.shape==mask.shape and np.array_equal(source.affine,mask.affine)
    if nib.aff2axcodes(source.affine)!=('R','A','S'):raise ValueError('Native RAS gate')
    image=source.get_fdata(dtype=np.float32);target=np.asanyarray(mask.dataobj);t=index_transform(image.shape,256)
    temporary=directory.with_name(case+'.partial');temporary.mkdir(parents=True,exist_ok=True)
    assert temporary.resolve().is_relative_to(OUT.resolve()) and directory.resolve().is_relative_to(OUT.resolve())
    shape=(image.shape[2],256,256);x=np.lib.format.open_memmap(temporary/'image.npy',mode='w+',dtype=np.float32,shape=shape);y=np.lib.format.open_memmap(temporary/'mask.npy',mode='w+',dtype=np.uint8,shape=shape)
    pre=json.loads((REPO/'configs/cardiac_chd68_preprocessing_v1.json').read_text());low,high=pre['clip_source_intensity'];positive=[];retained=np.zeros(8,np.int64);slice_counts=[]
    for z in range(shape[0]):
        # Exact historical operation order: linear raw counts, THEN clip/normalize.
        raw=resize_plane(image[:,:,z],t,1,0);x[z]=np.clip((raw-low)/(high-low),0,1)*2-1
        rx,ry=t['resized_xy'];px,py=t['pad_xy'];x[z,:py]=-1;x[z,py+ry:]=-1;x[z,:,:px]=-1;x[z,:,px+rx:]=-1
        rawmask=resize_plane(target[:,:,z],t,0,255);y[z]=np.where(rawmask<=7,rawmask,255).astype(np.uint8)
        counts=np.bincount(y[z][y[z]<8],minlength=8);retained+=counts;slice_counts.append(counts.tolist())
        if counts[1:].sum():positive.append(z)
    x.flush();y.flush();del x,y
    p={'case_id':case,'partition':role,'transform':t,'positive_indices':positive,'negative_indices':sorted(set(range(shape[0]))-set(positive)),
       'source_image_SHA256':row['image_SHA256'],'source_mask_SHA256':row['mask_SHA256'],'original_affine':source.affine.tolist(),'original_spatial_unit':source.header.get_xyzt_units()[0],
       'physical_geometry_verified':False,'split_SHA256':fingerprint(REPO/'configs/splits/cardiac_chd68_v1.json'),'cohort_SHA256':fingerprint(REPO/'metadata/pediatric/cardiac_chd68_cohort_v1.json'),
       'preprocessing_SHA256':fingerprint(REPO/'configs/cardiac_chd68_preprocessing_v1.json'),'image_SHA256':fingerprint(temporary/'image.npy'),'mask_SHA256':fingerprint(temporary/'mask.npy')}
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(case.encode()).digest()[:8],'little')^20261009);coordinates=[rng.integers(0,n,65536) for n in image.shape];sample=image[tuple(coordinates)]
    gt=cardiac_target(target);fg=(gt>0)&(gt<8);profiles=[fg.sum(axis=tuple(j for j in range(3) if j!=a)) for a in range(3)]
    occupied=[np.flatnonzero(p) for p in profiles];lo=[int(p[0]) for p in occupied];hi=[int(p[-1])+1 for p in occupied]
    center=np.array([np.dot(np.arange(len(p)),p)/max(1,p.sum()) for p in profiles])
    stats={'case':case,'role':role,'shape_xyz':list(image.shape),'slice_count':shape[0],'prepared_class_counts':retained.tolist(),'prepared_slice_class_counts':slice_counts,
      'positive_slices':len(positive),'negative_slices':shape[0]-len(positive),'source_sample_mean':float(sample.mean()),'source_sample_std':float(sample.std()),
      'source_sample_percentiles':np.percentile(sample,[1,10,50,90,99.5]).tolist(),'foreground_fraction':float(fg.mean()),'class_source_voxels':[int((gt==c).sum()) for c in range(8)],
      'GT_bbox':[lo,hi],'normalized_center_mass':(center/image.shape).tolist(),'GT_bbox_FOV':float(np.prod(np.array(hi)-lo)/image.size)}
    write(temporary/'statistics.json',stats);write(temporary/'provenance.json',p);os.replace(temporary,directory)
    print('HISTORICAL CACHE READY',role,case,shape[0],len(positive),flush=True);return stats

def run():
    from concurrent.futures import ThreadPoolExecutor,as_completed
    split,rows=cohort();results=[]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(prepare_case,c,role,rows[c]) for role in ('train','validation') for c in split[role]]
        for future in as_completed(futures):results.append(future.result())
    results.sort(key=lambda r:r['case']);write(OUT/'case_statistics.json',results)
    original=json.loads((REPO/'metadata/pediatric/cardiac_chd68_train_statistics_v1.json').read_text())['cases'];lookup={r['case']:r for r in results}
    for row in original:
        current=lookup[row['case_id']]
        assert current['positive_slices']==row['positive_slices'] and current['negative_slices']==row['negative_slices']
        assert current['prepared_class_counts'][1:]==row['prepared_foreground_class_voxels']
    dataset=CardiacDataset(CONFIG,'train',OUT,augmentation=False);assert len(dataset)==12381;dataset.close()
    write(OUT/'preparation.json',{'cases':58,'training_samples_per_epoch':12381,'steps_per_epoch_batch16':774,'historical_positive_and_class_counts_exact':True,'test_cases_prepared':0})
    print('HISTORICAL CACHE COMPLETE: exact train slice/class-count match',flush=True)

if __name__=='__main__':run()
