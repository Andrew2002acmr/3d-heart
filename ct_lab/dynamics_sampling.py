"""Train-only sampler ablations and measured exposure statistics; no model fitting."""
from collections import Counter
from torch.utils.data import DataLoader
from ct_lab.dynamics_core import *
from heart3d.ml.cardiac_data import CardiacDataset

def roi05_centers(stat):
    lo,hi=np.array(stat['GT_bbox'])[:,2];nz=stat['slice_count'];positive=np.linspace(lo,hi-1,12).round().astype(int)
    outside=np.array([z for z in np.linspace(0,nz-1,12).round().astype(int) if not lo<=z<hi])
    extra=outside[np.linspace(0,len(outside)-1,min(4,len(outside))).round().astype(int)] if len(outside) else []
    return sorted(set(positive.tolist()+list(extra)))

def historical_order(n,seed,epoch):
    # DataLoader consumes one base-seed draw even with zero workers.
    generator=torch.Generator().manual_seed(seed+epoch);torch.empty((),dtype=torch.int64).random_(generator=generator)
    return torch.randperm(n,generator=generator).tolist()

def sampling_plan(data,variant,seed,epoch,statistics):
    """Returns indices into the same dataset; only sampling changes, not augmentation."""
    if data.partition!='train':raise ValueError('Sampling policies are train only')
    cases=[r['case_id'] for r in data.rows];split,_=cohort()
    if not set(cases)<=set(split['train']):raise ValueError('Validation in training sampler')
    data.set_epoch(epoch);original=list(data.indices);length=len(original)
    if variant=='historical':return historical_order(length,seed,epoch),length
    if variant not in ('ROI05_pool','equal_case','foreground_boost'):raise ValueError('Unknown sampler')
    pools={c:[] for c in cases};positive={c:set(data.provenance[c]['positive_indices']) for c in cases}
    for i,(case,z) in enumerate(original):pools[case].append(i)
    if variant=='ROI05_pool':
        lookup={value:i for i,value in enumerate(data.indices)}
        for case in cases:
            pools[case]=[]
            for z in roi05_centers(statistics[case]):
                if (case,z) not in lookup:lookup[case,z]=len(data.indices);data.indices.append((case,z))
                pools[case].append(lookup[case,z])
    case_rng=np.random.default_rng(seed+epoch*100003+17);slice_rng=np.random.default_rng(seed+epoch*100003+29)
    order=np.concatenate([case_rng.permutation(len(cases)) for _ in range(int(np.ceil(length/len(cases))))])[:length];indices=[]
    splitp={c:([i for i in pools[c] if data.indices[i][1] in positive[c]],[i for i in pools[c] if data.indices[i][1] not in positive[c]]) for c in cases}
    for which in order:
        case=cases[which];pool=pools[case]
        if variant=='foreground_boost':
            fg,bg=splitp[case];p=len(fg)/max(1,len(fg)+len(bg));probability=p+.5*(1-p)
            pool=fg if fg and (not bg or slice_rng.random()<probability) else bg
        indices.append(pool[int(slice_rng.integers(len(pool)))])
    return indices,length

def simulate():
    cfg=protocol()['config'];stats={r['case']:r for r in json.loads((OUT/'case_statistics.json').read_text())};results=[];seed=cfg['seed'];steps=cfg['stage06']['ablation_steps']
    seed_runtime(seed,1,'cpu')
    for variant in cfg['stage06']['sampling_variants']:
        data=CardiacDataset(CONFIG,'train',OUT);counts=Counter();per_case=Counter();class_counts=np.zeros(8,np.int64);fg=multi=examples=completed=0;hist=np.zeros(32,np.int64);input_samples=0;unique_epoch=[]
        for epoch in range(100):
            order,length=sampling_plan(data,variant,seed,epoch,stats);seen=set()
            for start in range(0,length,16):
                chosen=order[start:start+16]
                for index in chosen:
                    case,z=data.indices[index];key=f'{case}:{z}';counts[key]+=1;seen.add(key);per_case[case]+=1;examples+=1
                    values=np.array(stats[case]['prepared_slice_class_counts'][z]);class_counts+=values;fg+=int(values[1:].sum()>0);multi+=int((values[1:]>0).sum()>=2)
                # Actual augmented input distribution, bounded deterministic subsample.
                if completed%64==0 and input_samples<128:
                    index=chosen[0];image=data[index]['image'].numpy()[:,::8,::8];hist+=np.histogram(image,bins=np.linspace(-1,1,33))[0];input_samples+=1
                completed+=1
                if completed>=steps:break
            unique_epoch.append(len(seen))
            if completed>=steps:break
        row={'sampler':variant,'scope':'train-only sampling simulation, not learned quality','seed':seed,'steps':completed,'examples':examples,'unique_cases':len(per_case),'unique_slices':len(counts),
          'foreground_slice_fraction':fg/examples,'background_slice_fraction':1-fg/examples,'multiple_structure_fraction':multi/examples,'prepared_clean_class_voxel_counts':class_counts.tolist(),
          'repeat_mean':float(np.mean(list(counts.values()))),'repeat_max':max(counts.values()),'unique_slices_by_epoch':unique_epoch,'case_example_counts':dict(per_case),
          'augmented_input_histogram':hist.tolist(),'input_histogram_edges':np.linspace(-1,1,33).tolist(),'actual_augmented_input_examples_sampled':input_samples,
          'clean_counts_vs_augmented_histogram':'class counts describe sampled clean targets; input histogram uses128 actual original-augmentation samples'}
        results.append(row);write(OUT/'sampling'/f'{variant}.json',row);write(OUT/'sampling'/f'{variant}_exposures.json',dict(counts));data.close();print('SAMPLING SIMULATION',variant,examples,len(counts),row['foreground_slice_fraction'],flush=True)
    write(OUT/'sampling_analysis.json',results)

if __name__=='__main__':simulate()
