"""Real source UI checks; validation never fits projections and test is not loaded."""
import json
from pathlib import Path
import ct_lab.spatial_features
from streamlit.testing.v1 import AppTest


def run():
    root=Path(__file__).resolve().parents[1];app=AppTest.from_file(str(root/'ct_lab/app.py'),default_timeout=180).run();checks=[]
    def check(name):
        errors=[e.message for e in app.exception];checks.append({'check':name,'exceptions':errors});print(name,errors,flush=True);assert not errors
    app.sidebar.radio[0].set_value('Spatial Feature Explorer').run();check('Native 3 planes, neighbors and fixed voxel IDs')
    next(c for c in app.checkbox if c.label=='Общая проекция пересечения признаков — контроль').check().run();check('Shared PCA intersection control')
    next(s for s in app.selectbox if s.label=='Spatial feature set').set_value('A Intensity').run();check('One-dimensional shared PCA')
    next(s for s in app.selectbox if s.label=='Spatial error overlay').set_value('Correction vs 2D B').run();check('GT distance diagnostic / correction maps separate from features')
    next(s for s in app.selectbox if s.label=='Spatial feature set').set_value('G 3D').run();check('Full 3D feature bank')
    next(s for s in app.selectbox if s.label=='Spatial projection').set_value('UMAP').run();check('Independent and common train-only UMAP')
    next(s for s in app.selectbox if s.label=='Пара структур').set_value('RV/PA').run();check('Control RV/PA pair')
    next(s for s in app.selectbox if s.label=='Spatial projection').set_value('PCA').run()
    next(s for s in app.selectbox if s.label=='Пара структур').set_value('Произвольная foreground пара').run()
    next(s for s in app.multiselect if s.label=='Выберите два foreground класса').set_value([1,6]).run();check('Arbitrary train foreground pair')
    next(s for s in app.selectbox if s.label=='Пара структур').set_value('LV/MYO').run()
    import ct_lab.spatial_ui as ui
    original=ui.project_local;calls=[]
    def forbidden(*args,**kwargs):calls.append('fit');raise AssertionError('No validation projection fit')
    ui.project_local=forbidden
    try:
        from ct_lab.core import discover
        cases=discover(root.parent/'laboratories/data/chd68/pairs')
        next(s for s in app.sidebar.selectbox if s.label=='Исследование').set_value(next(i for i,r in enumerate(cases) if r['id']=='ct_1027')).run()
        check('Validation raw viewer with fixed metrics, no projection fit');assert not calls
    finally:ui.project_local=original
    import ct_lab.core as core
    loader=core.load_pair;reads=[]
    def no_test(row):reads.append(row['image']);raise AssertionError('Sealed test load')
    core.load_pair=no_test
    try:
        next(s for s in app.sidebar.selectbox if s.label=='Исследование').set_value(next(i for i,r in enumerate(cases) if r['id']=='ct_1022')).run()
        check('Sealed test gated before source load');assert not reads
    finally:core.load_pair=loader
    (root/'outputs/spatial_feature_02/ui_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')


if __name__=='__main__':run()
