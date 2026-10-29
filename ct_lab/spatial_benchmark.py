"""Record real sampled CPU runtime/RSS without dense dataset feature allocation."""
import json
import threading
import time
import numpy as np
import nibabel as nib
import psutil
from ct_lab.spatial_features import extract
from ct_lab.spatial_run import OUT,DEFAULT_DATA


def run():
    image=nib.load(DEFAULT_DATA/'ct_1001_image.nii.gz');volume=image.get_fdata(dtype=np.float32)
    with np.load(OUT/'case_ct_1001.npz') as data:
        points=np.concatenate([data[f'p{p}_xyz'][data[f'p{p}_diagnostic_indices']] for p in (0,1)])
    process=psutil.Process();records=[]
    for config in ({'method':'Baseline','dimensions':3},{'method':'Gaussian','sigma':2.,'dimensions':2},
                   {'method':'Gaussian','sigma':2.,'dimensions':3},{'method':'Bilateral','dimensions':2},{'method':'Bilateral','dimensions':3}):
        stop=threading.Event();rss=[]
        def poll():
            while not stop.wait(.02):rss.append(process.memory_info().rss)
        worker=threading.Thread(target=poll,daemon=True);worker.start();start=time.perf_counter()
        result=extract(volume,points,image.affine,filter_config=config,include_gaussians=False,chunk_size=16)
        elapsed=time.perf_counter()-start;stop.set();worker.join()
        records.append({'filter':config,'sampled_points':len(points),'seconds':elapsed,'peak_sampled_RSS_bytes':max(rss or [process.memory_info().rss]),
                        'retained_feature_bytes':result['values'].nbytes,'estimated_patch_buffers_bytes':result['estimated_temporary_patch_bytes']})
    (OUT/'cpu_benchmark.json').write_text(json.dumps({'case':'ct_1001','scope':'real fixed reduced filter subset','CPU_only':True,
                    'first_bilateral_call_includes_JIT':True,'RSS_includes_source_volume_and_runtime':True,'records':records},indent=2),encoding='utf-8')
    print('CPU BENCHMARK',[(r['filter'],round(r['seconds'],2)) for r in records],flush=True)


if __name__=='__main__':run()
