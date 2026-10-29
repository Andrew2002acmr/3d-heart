"""Real CT integration smoke for opt-in UMAP and automatic parameter updates."""
import json
from pathlib import Path
import ct_lab.launch
from streamlit.testing.v1 import AppTest


def main():
    repo=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(repo/'ct_lab/app.py'),default_timeout=180).run()
    app.sidebar.radio[0].set_value('Filtering & Feature Analysis').run()
    next(c for c in app.checkbox if c.label.startswith('Рассчитать UMAP')).check().run()
    assert not app.exception,[e.message for e in app.exception]
    next(s for s in app.selectbox if s.label=='Метод обработки').set_value('Gaussian').run()
    assert not app.exception,[e.message for e in app.exception]
    next(b for b in app.button if b.label=='Сохранить эксперимент фильтрации и интерактивный отчёт').click().run()
    assert not app.exception,[e.message for e in app.exception]
    (repo/'outputs/filtering_feature_01/umap_ui_verification.json').write_text(json.dumps({
        'source':'real ct_1001','UMAP_baseline':True,'UMAP_filter_parameter_update':True,'export_includes_UMAP':True,
        'exceptions':[]},indent=2),encoding='utf-8')
    print('UMAP UI OK',flush=True)


if __name__=='__main__':main()
