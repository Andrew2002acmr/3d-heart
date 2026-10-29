"""Final provenance check and portable export of experiment 01 results."""
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
import zipfile
import ct_lab.launch
from ct_lab.core import fingerprint
from ct_lab.filtering_run import DEFAULT_OUTPUT,DEFAULT_DATA,REPO,protected_hashes


def main():
    root=DEFAULT_OUTPUT;results=json.loads((root/'results.json').read_text())
    if results['protected_files_unchanged']!=protected_hashes():raise ValueError('Frozen split/preprocessing/weights changed')
    source_hashes={}
    for case in results['source_cases_unchanged']:
        saved=json.loads((root/f'measurements_{case}.json').read_text())
        hashes={kind:fingerprint(DEFAULT_DATA/f'{case}_{suffix}.nii.gz') for kind,suffix in [('image','image'),('mask','label')]}
        if hashes['image']!=saved['image_sha256'] or hashes['mask']!=saved['mask_sha256']:raise ValueError('Source changed: '+case)
        source_hashes[case]=hashes
    versions={}
    for name in ['numpy','scipy','scikit-image','nibabel','umap-learn','numba','llvmlite','scikit-learn','pynndescent','streamlit','plotly','matplotlib']:
        versions[name]=importlib.metadata.version(name)
    runtime={'python':sys.version,'packages':versions,'voxel_id_definition':'np.ravel_multi_index((X,Y,Z), source_shape_xyz, order=C)',
             'source_files_unchanged':results['source_cases_unchanged'],'protected_files_unchanged':True,
             'final_source_sha256':source_hashes,
             'source_code_sha256':{p.name:fingerprint(p) for p in (REPO/'ct_lab').glob('filtering*.py')}}
    (root/'runtime_and_provenance.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    (root/'requirements-filtering-lock.txt').write_text('\n'.join(f'{name}=={version}' for name,version in versions.items())+'\n',encoding='utf-8')
    regions={'homogeneous_selection':'GT class erosion 3 pixels; lower quartile baseline local std separately per class; unchanged positions across filters',
             'expert_interface':'binary dilation contact band between the two selected GT classes; 4-connected default structuring element',
             'local_variance':'Local Std squared; median over fixed homogeneous proxy regions',
             'aggregation':'median available slices per case, then median cases; k-NN primary mean per-case macro F1',
             'contrast':'abs(mean(class A)-mean(class B))','Fisher':'squared mean difference / sum of population variances',
             'histogram_overlap':'sum min(normalized counts A, normalized counts B), fixed 128 bins [0,4095]'}
    (root/'measurement_protocol.json').write_text(json.dumps(regions,indent=2),encoding='utf-8')
    destination=REPO/'outputs/FILTERING_FEATURE_EXPERIMENT_01_RESULTS.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and 'numba_cache' not in path.parts:archive.write(path,Path('results')/path.relative_to(root))
        archive.write(REPO/'FILTERING_FEATURE_EXPERIMENT_01.md','FILTERING_FEATURE_EXPERIMENT_01.md')
        archive.write(REPO/'requirements-filtering.txt','requirements-filtering.txt')
        archive.write(REPO/'ct_lab/README.md','ct_lab/README.md')
        for path in (REPO/'ct_lab').glob('filtering*'):
            if path.is_file():archive.write(path,Path('ct_lab')/path.name)
        archive.write(REPO/'tests/test_filtering_features.py','tests/test_filtering_features.py')
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():raise ValueError('Archive CRC failure')
        print('PACKAGED',len(archive.namelist()),destination.stat().st_size,flush=True)


if __name__=='__main__':main()
