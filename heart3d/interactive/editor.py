"""Local original-grid editor: linked CT planes, source/draft comparison and 3D.

Explicit namespace: binary Heart OAR is never displayed as LV.
"""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ["QT_API"]="pyside6"
import nibabel as nib
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtCore, QtGui, QtWidgets
from heart3d.annotation import MaskEditor
from heart3d.labels import LABELS
from heart3d.storage import sha256_file
from heart3d.surfaces import build_surface
from heart3d.interactive.slices import ImageCanvas

class EditCanvas(ImageCanvas):
    touched=QtCore.Signal(float,float)
    def mousePressEvent(self,event):
        if event.button()==QtCore.Qt.LeftButton:self.touch(event)
    def mouseMoveEvent(self,event):
        if event.buttons() & QtCore.Qt.LeftButton:self.touch(event)
    def touch(self,event):
        width=min(self.width(),self.height()*self.aspect);height=width/self.aspect
        left=(self.width()-width)/2;top=(self.height()-height)/2
        x=(event.position().x()-left)/width;y=(event.position().y()-top)/height
        if 0<=x<1 and 0<=y<1:self.touched.emit(x,y)

class EditorWindow(QtWidgets.QMainWindow):
    def __init__(self,ct_path,mask_path,geometry_path,output,schema="heart",off_screen=False):
        super().__init__()
        self.ct_path,self.mask_path,self.output=Path(ct_path),Path(mask_path),Path(output)
        geometry=json.loads(Path(geometry_path).read_text())
        if not geometry["geometry"]["physical_geometry_confirmed_from_DICOM"] or sha256_file(ct_path)!=geometry["nifti_sha256"]:
            raise ValueError("CT does not match confirmed DICOM conversion")
        ct=nib.load(ct_path);mask=nib.load(mask_path)
        if ct.header.get_xyzt_units()[0]!="mm" or mask.header.get_xyzt_units()[0]!="mm":
            raise ValueError("Confirmed millimetre grids required")
        if ct.shape!=mask.shape or not np.allclose(ct.affine,mask.affine,atol=1e-5,rtol=0):
            raise ValueError("Original image/mask grid mismatch")
        self.ct=ct.get_fdata(dtype=np.float32);raw=np.asanyarray(mask.dataobj)
        if not np.isfinite(raw).all() or np.any(raw<0) or np.any(raw>65535) or not np.equal(raw,np.rint(raw)).all():raise ValueError("Invalid discrete mask")
        self.original=raw.astype(np.uint16)
        if not np.isfinite(self.ct).all():raise ValueError("Nonfinite CT")
        provenance_path=Path(mask_path).parent/"provenance.json"
        if schema=="cardiac7" and provenance_path.exists():
            source=json.loads(provenance_path.read_text())
            if "Heart OAR" in source.get("interpretation", ""):
                raise ValueError("Binary Heart OAR cannot be reinterpreted as LV")
        labels={1:"Heart OAR"} if schema=="heart" else {k:v[0] for k,v in LABELS.items()}
        self.editor=MaskEditor(self.original,ct.affine,labels,sha256_file(ct_path),sha256_file(mask_path),True)
        self.indices=[(n-1)//2 for n in self.ct.shape]
        if self.original.any():
            points=np.argwhere(self.original>0);self.indices=np.rint(points.mean(axis=0)).astype(int).tolist()
        self.setWindowTitle("КТ → черновая маска → ручная правка → 3D")
        self.resize(1440,940)
        root=QtWidgets.QWidget();self.setCentralWidget(root);layout=QtWidgets.QVBoxLayout(root)
        tools=QtWidgets.QHBoxLayout()
        self.tool=QtWidgets.QComboBox();self.tool.addItems(["Навигация","Кисть","Ластик"])
        self.label=QtWidgets.QComboBox()
        for number,name in labels.items():self.label.addItem(name,number)
        self.radius=QtWidgets.QDoubleSpinBox();self.radius.setRange(.2,15);self.radius.setValue(2);self.radius.setSuffix(" мм")
        self.mode=QtWidgets.QComboBox();self.mode.addItems(["Правка","Черновик","Сравнение"])
        self.low=QtWidgets.QSpinBox();self.high=QtWidgets.QSpinBox()
        for spin,value in [(self.low,-200),(self.high,700)]:spin.setRange(-10000,20000);spin.setValue(value)
        undo=QtWidgets.QPushButton("Отменить");rebuild=QtWidgets.QPushButton("Обновить 3D")
        save=QtWidgets.QPushButton("Сохранить версию")
        for widget in [self.tool,self.label,self.radius,self.mode,QtWidgets.QLabel("Окно"),self.low,self.high,undo,rebuild,save]:
            tools.addWidget(widget)
        layout.addLayout(tools)
        self.notice=QtWidgets.QLabel("Черновая исследовательская разметка · анатомия экспертом не проверена")
        layout.addWidget(self.notice)
        grid=QtWidgets.QGridLayout();layout.addLayout(grid,1)
        self.canvases={};self.sliders={}
        for axis,name,position in [(2,"Axial",(0,0)),(1,"Coronal",(0,1)),(0,"Sagittal",(1,0))]:
            group=QtWidgets.QGroupBox(name);box=QtWidgets.QVBoxLayout(group)
            canvas=EditCanvas();other=[i for i in range(3) if i!=axis]
            spacing=nib.affines.voxel_sizes(ct.affine)
            canvas.aspect=self.ct.shape[other[0]]*spacing[other[0]]/(self.ct.shape[other[1]]*spacing[other[1]])
            slider=QtWidgets.QSlider(QtCore.Qt.Horizontal);slider.setRange(0,self.ct.shape[axis]-1);slider.setValue(self.indices[axis])
            canvas.touched.connect(lambda x,y,a=axis:self.touch(a,x,y))
            slider.valueChanged.connect(lambda v,a=axis:self.slice_changed(a,v))
            box.addWidget(canvas,1);box.addWidget(slider);grid.addWidget(group,*position)
            self.canvases[axis]=canvas;self.sliders[axis]=slider
        group=QtWidgets.QGroupBox("3D · ПКМ на поверхности → исходная КТ")
        box=QtWidgets.QVBoxLayout(group)
        self.plotter=QtInteractor(group,off_screen=off_screen,auto_update=False,multi_samples=0)
        self.plotter.set_background("#17212b");box.addWidget(self.plotter.interactor);grid.addWidget(group,1,1)
        if self.plotter.iren is not None:
            self.plotter.enable_surface_point_picking(callback=self.pick,show_message=False,show_point=True)
        self.note=QtWidgets.QLineEdit();self.note.setPlaceholderText("Комментарий к правке / сомнительные границы")
        layout.addWidget(self.note)
        undo.clicked.connect(self.undo);rebuild.clicked.connect(self.rebuild);save.clicked.connect(self.save)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.low.valueChanged.connect(self.refresh);self.high.valueChanged.connect(self.refresh)
        self.refresh();self.rebuild()
    def slice_changed(self,axis,value):
        self.indices[axis]=value;self.refresh()
    def touch(self,axis,x,y):
        other=[i for i in range(3) if i!=axis]
        u=x*self.ct.shape[other[0]]-.5
        v=(1-y)*self.ct.shape[other[1]]-.5
        if self.tool.currentIndex()==0:
            self.indices[other[0]]=int(np.clip(round(u),0,self.ct.shape[other[0]]-1))
            self.indices[other[1]]=int(np.clip(round(v),0,self.ct.shape[other[1]]-1))
            self.sync()
        else:
            if self.mode.currentIndex()==1:
                self.statusBar().showMessage("Переключите просмотр на Правка или Сравнение");return
            label=0 if self.tool.currentIndex()==2 else self.label.currentData()
            self.editor.paint(axis,self.indices[axis],(u,v),self.radius.value(),label)
            self.refresh();self.statusBar().showMessage("Маска изменена; обновите 3D перед проверкой поверхности")
    def pick(self,point):
        try:
            self.indices=self.editor.world_to_voxel(point).tolist();self.sync()
            self.statusBar().showMessage("Участок поверхности сопоставлен с исходной сеткой КТ")
        except ValueError as e:self.statusBar().showMessage(str(e))
    def sync(self):
        for axis,slider in self.sliders.items():
            blocker=QtCore.QSignalBlocker(slider);slider.setValue(self.indices[axis]);del blocker
        self.refresh()
    def refresh(self,*_):
        for axis,canvas in self.canvases.items():
            selection=[slice(None)]*3;selection[axis]=self.indices[axis]
            plane=self.ct[tuple(selection)].T
            gray=np.clip((plane-self.low.value())/max(1,self.high.value()-self.low.value()),0,1)
            rgb=np.repeat((gray*255)[...,None],3,axis=2)
            current=self.editor.mask[tuple(selection)].T;original=self.original[tuple(selection)].T
            displayed=original if self.mode.currentIndex()==1 else current
            for label in self.editor.labels:
                color=np.array([255,153,70]) if len(self.editor.labels)==1 else np.array([int(LABELS[label][2][i:i+2],16) for i in (1,3,5)])
                selected=displayed==label;rgb[selected]=.6*rgb[selected]+.4*color
            if self.mode.currentIndex()==2:
                delta=original!=current;rgb[delta]=.4*rgb[delta]+.6*np.array([70,230,190])
            rgb=np.ascontiguousarray(np.flipud(rgb).astype(np.uint8))
            other=[i for i in range(3) if i!=axis]
            row=self.ct.shape[other[1]]-1-self.indices[other[1]];column=self.indices[other[0]]
            rgb[row,:,0]=220;rgb[:,column,0]=220
            h,w=rgb.shape[:2]
            canvas.image=QtGui.QImage(rgb.data,w,h,rgb.strides[0],QtGui.QImage.Format_RGB888).copy();canvas.update()
    def mode_changed(self,*_):self.refresh();self.rebuild()
    def undo(self):
        self.editor.undo();self.refresh();self.statusBar().showMessage("Действие отменено; обновите 3D")
    def rebuild(self):
        camera=self.plotter.camera_position if self.plotter.actors else None
        self.plotter.clear();self.plotter.enable_lightkit()
        masks=[("source",self.original,"#ffaa66",.35)] if self.mode.currentIndex()==1 else [("edited",self.editor.mask,"#ff9966",1)]
        if self.mode.currentIndex()==2:masks=[("source",self.original,"#ff9966",.3),("edited",self.editor.mask,"#4be0bd",.7)]
        for prefix,mask,color,opacity in masks:
            for label in self.editor.labels:
                mesh,_=build_surface(mask,label,self.editor.affine)
                if mesh is not None:self.plotter.add_mesh(mesh,color=color,opacity=opacity,name=f"{prefix}_{label}",reset_camera=False)
        if camera:self.plotter.camera_position=camera
        else:self.plotter.view_isometric();self.plotter.reset_camera()
        self.plotter.render()
    def save(self):
        try:
            version=self.editor.save(self.output,self.ct_path,self.mask_path,self.note.text())
            summaries=[]
            for label,name in self.editor.labels.items():
                mesh,info=build_surface(self.editor.mask,label,self.editor.affine)
                if mesh is not None:
                    mesh.field_data["unit"]=["mm"];mesh.field_data["coordinate_system"]=["RAS"]
                    mesh.save(version/f"label_{label}.vtp");info.update({"short_name":name,"name":name});summaries.append(info)
            (version/"mesh_summary.json").write_text(json.dumps(summaries,indent=2)+"\n",encoding="utf-8")
            self.statusBar().showMessage("Версия сохранена: "+str(version))
            return version
        except Exception as e:self.statusBar().showMessage(str(e));return None
    def closeEvent(self,event):
        self.plotter.close();event.accept()

def capture_window(window,path):
    """Qt grab does not include native VTK pixels; composite actual VTK render."""
    pixmap=window.grab()
    rendered=np.ascontiguousarray(window.plotter.screenshot(return_img=True)[:,:,:3])
    h,w=rendered.shape[:2]
    image=QtGui.QImage(rendered.data,w,h,rendered.strides[0],QtGui.QImage.Format_RGB888).copy()
    position=window.plotter.interactor.mapTo(window,QtCore.QPoint(0,0))
    painter=QtGui.QPainter(pixmap)
    painter.drawImage(QtCore.QRect(position,window.plotter.interactor.size()),image);painter.end()
    Path(path).parent.mkdir(parents=True,exist_ok=True);pixmap.save(str(path))

def run_editor(ct,mask,geometry,output,schema="heart",screenshot=None):
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window=EditorWindow(ct,mask,geometry,output,schema,False);window.show()
    if screenshot:
        def capture():
            capture_window(window,screenshot);window.close();app.quit()
        QtCore.QTimer.singleShot(1500,capture)
    return app.exec()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["ct","mask","geometry","output"]:p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--schema",choices=["heart","cardiac7"],default="heart")
    p.add_argument("--screenshot",type=Path)
    a=p.parse_args()
    return run_editor(a.ct,a.mask,a.geometry,a.output,a.schema,a.screenshot)
if __name__=="__main__":raise SystemExit(main())
