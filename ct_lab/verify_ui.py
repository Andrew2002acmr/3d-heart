"""Streamlit integration smoke on a real CT; output contains no synthetic analysis."""
import json
from pathlib import Path
import ct_lab.launch  # workspace-local UI dependencies
from streamlit import config
config.set_option('global.developmentMode',False)
config.set_option('logger.level','error')
from streamlit.testing.v1 import AppTest


def main():
    repo=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(repo/'ct_lab/app.py'), default_timeout=180)
    app.run()
    records=[]
    def check(name):
        errors=[str(x.message) for x in app.exception]
        records.append({'section':name,'exceptions':errors})
        print(name, errors,flush=True)
        if errors: raise AssertionError(errors)
    check('Dataset Overview')
    for name in ['CT Explorer','HU Analysis','Preprocessing','Spatial Context','Mask Anatomy',
                 'Model Input Inspector','Model Diagnostics','Experiments']:
        app.sidebar.radio[0].set_value(name).run()
        check(name)
        if name=='HU Analysis':
            next(s for s in app.selectbox if s.label=='Область распределения').set_value('По классам').run()
            next(s for s in app.selectbox if s.label=='Сравнение распределений').set_value('Box').run()
            check('class box plot')
        if name=='Model Input Inspector':
            next(b for b in app.button if b.label=='Получить фактический preprocessing / тензор').click().run()
            check('actual tensor inspector')
    output=repo/'outputs/ct_preprocessing_lab';output.mkdir(parents=True,exist_ok=True)
    (output/'ui_verification.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__': main()
