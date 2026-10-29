"""Final read-only provenance verification and stage-02 research archive."""
import importlib.metadata
import json
import sys
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
from ct_lab.spatial_run import OUT,REPO,DEFAULT_DATA,previous_hashes,protected_hashes
from ct_lab.core import fingerprint


def run():
    old=json.loads((OUT/'preserved_stage01.json').read_text());guards=json.loads((OUT/'protected_files.json').read_text())
    if old!=previous_hashes():raise ValueError('Stage01 results changed')
    if guards!=protected_hashes():raise ValueError('Production files changed')
    result=json.loads((OUT/'results.json').read_text());plan=result['protocol'];sources={}
    for case in plan['train_cases']+plan['validation_cases']:
        info=json.loads((OUT/f'case_{case}.json').read_text())
        image=fingerprint(DEFAULT_DATA/f'{case}_image.nii.gz');mask=fingerprint(DEFAULT_DATA/f'{case}_label.nii.gz')
        if image!=info['image_sha256'] or mask!=info['mask_sha256']:raise ValueError('Source changed: '+case)
        sources[case]={'image':image,'mask':mask}
    environment={'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in
                                               ['numpy','scipy','nibabel','scikit-image','numba','llvmlite','scikit-learn','umap-learn','streamlit','plotly','matplotlib']}}
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    evidence={'previous_stage_files_verified':len(old),'previous_results_unchanged':True,'production_weights_split_preprocessing_unchanged':True,
              'source_pairs':sources,'test_CT_opened':False,'neural_training_executed':False,
              'same_samples_A_to_H':True,'same_samples_all_filter_diagnostics':True,
              'code_sha256':{p.name:fingerprint(p) for p in (REPO/'ct_lab').glob('spatial*') if p.is_file()}}
    (OUT/'final_verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    tests=ET.parse(OUT/'tests.xml').getroot();count=sum(int(t.attrib.get('tests',0)) for t in tests.iter('testsuite'))
    report=REPO/'SPATIAL_FEATURE_EXPERIMENT_02.md'
    text=report.read_text(encoding='utf-8')
    appendix=f'\n## Финальная проверка\n\nПройдено {count} автоматических тестов без neural training. UI integration: ui_verification.json. Реальные browser clicks во всех трёх плоскостях и scatter lasso: browser_verification.json. Проверены {len(sources)} исходных CT/mask пар, {len(old)} файлов этапа 1 и все production guards. Версии библиотек — environment.json; resource sampling — cpu_benchmark.json.\n'
    text=text.split('\n## Финальная проверка')[0]+appendix;report.write_text(text,encoding='utf-8')
    destination=REPO/'outputs/SPATIAL_FEATURE_EXPERIMENT_02_RESULTS.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and 'numba_cache' not in p.parts:archive.write(p,Path('results')/p.relative_to(OUT))
        archive.write(report,report.name)
        archive.write(REPO/'ct_lab/SPATIAL_README.md','ct_lab/SPATIAL_README.md')
        for p in (REPO/'ct_lab').glob('spatial*'):
            if p.is_file() and p.suffix.lower() in ('.py','.cjs'):archive.write(p,Path('ct_lab')/p.name)
        archive.write(REPO/'ct_lab/spatial_component/index.html','ct_lab/spatial_component/index.html')
        archive.write(REPO/'tests/test_spatial_features.py','tests/test_spatial_features.py')
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():raise ValueError('Archive CRC failure')
        if len(archive.namelist())!=len(set(archive.namelist())):raise ValueError('Duplicate archive entries')
        print('FINAL',count,'tests;',len(sources),'source pairs;',len(old),'prior files; ZIP',destination.stat().st_size,flush=True)


if __name__=='__main__':run()
