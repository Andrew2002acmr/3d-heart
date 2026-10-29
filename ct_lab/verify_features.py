"""Target changed UI paths using real data, including persisted exports."""
import json
from pathlib import Path
import ct_lab.launch
from streamlit.testing.v1 import AppTest


def main():
    repo=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(repo/'ct_lab/app.py'),default_timeout=180).run()
    results=[]
    def check(name):
        errors=[e.message for e in app.exception]
        results.append(dict(feature=name,exceptions=errors));print(name,errors,flush=True)
        assert not errors
    check('Overview with completed batch')
    app.sidebar.radio[0].set_value('Experiments').run()
    next(c for c in app.checkbox if c.label.startswith('Экспериментальное multi-window')).check().run()
    check('Three real-source window channels and ZIP generation')
    next(b for b in app.button if b.label=='Сохранить в сравнение').click().run()
    check('Persist experiment to disk')
    next(s for s in app.selectbox if s.label=='2D filter / исследовательская карта').set_value('Gaussian').run()
    check('Gaussian and quantitative boundary comparison')
    app.sidebar.radio[0].set_value('Spatial Context').run()
    check('Neighbor statistics and selected difference offset')
    (repo/'outputs/ct_preprocessing_lab/ui_features.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
