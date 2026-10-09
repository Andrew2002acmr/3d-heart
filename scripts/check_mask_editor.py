"""Exercise editor widgets on a local CT; probe edits are not anatomical corrections."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from PySide6 import QtCore,QtTest,QtWidgets
from heart3d.interactive.editor import EditorWindow,capture_window
from heart3d.storage import sha256_file

def check(a):
    a.output.mkdir(parents=True,exist_ok=True)
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window=EditorWindow(a.ct,a.mask,a.geometry,a.output/"technical_probe_versions",off_screen=False)
    window.show();QtTest.QTest.qWait(200)
    point=np.argwhere(window.original>0)[len(np.argwhere(window.original>0))//2]
    window.pick(nib.affines.apply_affine(window.editor.affine,point))
    assert window.indices==point.tolist()
    assert all(window.sliders[i].value()==point[i] for i in range(3))
    window.tool.setCurrentIndex(2);window.radius.setValue(.5)
    canvas=window.canvases[2]
    width=min(canvas.width(),canvas.height()*canvas.aspect);height=width/canvas.aspect
    x=(point[0]+.5)/window.ct.shape[0];y=1-(point[1]+.5)/window.ct.shape[1]
    clicked=QtCore.QPoint(round((canvas.width()-width)/2+x*width),round((canvas.height()-height)/2+y*height))
    QtTest.QTest.mouseClick(canvas,QtCore.Qt.LeftButton,pos=clicked)
    changed=int(np.count_nonzero(window.original!=window.editor.mask));assert changed>0
    window.undo();assert np.array_equal(window.original,window.editor.mask)
    QtTest.QTest.mouseClick(canvas,QtCore.Qt.LeftButton,pos=clicked)
    window.note.setText("TECHNICAL UI PROBE ONLY: brush/save round-trip; not an anatomical correction")
    version=window.save();assert version is not None
    restored=nib.load(version/"research_mask.nii.gz")
    assert np.array_equal(restored.dataobj,window.editor.mask)
    assert np.allclose(restored.affine,window.editor.affine,atol=1e-5,rtol=0)
    window.mode.setCurrentIndex(2);window.rebuild();QtTest.QTest.qWait(200)
    capture_window(window,a.output/"comparison_editor.png")
    render=window.plotter.screenshot(str(a.output/"comparison_3d.png"),return_img=True)
    assert render.std()>1
    unchanged=sha256_file(a.ct)==window.editor.source_ct_hash and sha256_file(a.mask)==window.editor.source_mask_hash
    assert unchanged
    result={"widget_brush_undo_save_passed":True,"picked_world_to_source_grid_passed":True,
        "original_CT_and_draft_unchanged":unchanged,"original_grid_roundtrip":True,
        "probe_changed_voxels":changed,"probe_annotation_version":str(version),
        "probe_is_anatomical_correction":False,"clinical_accuracy_assessed":False}
    (a.output/"editor_check.json").write_text(json.dumps(result,indent=2)+"\n")
    window.close();app.processEvents()
    print("EDITOR_CHECK",json.dumps({k:v for k,v in result.items() if k!="probe_annotation_version"}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["ct","mask","geometry","output"]:p.add_argument("--"+name,type=Path,required=True)
    check(p.parse_args())
