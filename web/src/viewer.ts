// Cornerstone owns world/canvas transforms, MPR, hit testing and brush rasterization.
// Only plain LPS control points and native voxel deltas cross the API boundary.
import * as core from '@cornerstonejs/core';
import * as tools from '@cornerstonejs/tools';
import type {AnnotationDocument, AnatomyLocator, Point, Project, SliceScope} from './api';
import {copy,sliceScope} from './api';

let initialized:Promise<void>|undefined;
function init(){return initialized ||= (async()=>{
  // Share the volume texture across our three MPR planes instead of allocating
  // independent contexts/textures for each plane on hybrid laptop GPUs.
  const config=core.getConfiguration();
  await core.init({...config,rendering:{...config.rendering,webGlContextCount:1}}); await tools.init();
  [tools.SplineROITool,tools.BrushTool,tools.PanTool,tools.ZoomTool,tools.WindowLevelTool,tools.StackScrollTool].forEach(t=>tools.addTool(t));
})();}
export type Mode='contour'|'brush'|'erase'|'pan'|'zoom'|'window';
// Hydration may rotate/reverse a closed polygon's vertex order. Camera clicks
// must not turn that equivalent ordering into a new annotation version.
function sameClosedPoints(first:Point[],second:Point[]){
  if(first.length!==second.length||!first.length)return false;
  const equal=(a:Point,b:Point)=>a.every((v,i)=>Math.abs(v-b[i])<1e-6);
  for(let start=0;start<second.length;start++)if(equal(first[0],second[start])){
    if([1,-1].some(direction=>first.every((p,i)=>equal(p,second[(start+direction*i+second.length)%second.length]))))return true;
  }
  return false;
}
export class Viewer {
  // Native annotation structures contain spline classes and are intentionally not JSON DTOs.
  private engine!:core.RenderingEngine;
  private group!:ReturnType<typeof tools.ToolGroupManager.createToolGroup>;
  private volume!:core.Types.IImageVolume;
  private mask!:core.Types.IImageVolume;
  private ids:string[]=[];
  private prefix=crypto.randomUUID();
  private segId=this.prefix+'-seg';
  private baseline=new Uint8Array();
  private maskHasValues=true;
  private anatomyCameras?:core.Types.ICamera[];
  private suppress=false;
  private timer:ReturnType<typeof setTimeout>|undefined;
  private listeners:Array<[string,EventListener]>=[];
  private cleanup:Array<()=>void>=[];
  private resize?:ResizeObserver;
  private activeLabel=1;
  private contoursVisible=true;
  private hiddenLabels=new Set<number>();
  private doc:AnnotationDocument;
  private readOnly=false;
  private editScope?:SliceScope;
  private currentMode:Mode='contour';
  constructor(private project:Project,private elements:HTMLDivElement[],
    private changed:(document:AnnotationDocument)=>void,private slices:(values:number[])=>void){this.doc=copy(project.document);this.editScope=sliceScope(project,'draft');}
  async start(ct:ArrayBuffer,mask:ArrayBuffer){
    await init();
    const g=this.project.geometry;
    const volumeId=this.prefix+'-ct',maskId=this.prefix+'-mask';
    const options={dimensions:g.shape,spacing:g.spacing,origin:g.origin_lps,direction:g.direction_lps as core.Types.Mat3,
      metadata:{Modality:'CT',FrameOfReferenceUID:this.prefix,SeriesInstanceUID:volumeId,
        BitsAllocated:32,BitsStored:32,HighBit:31,PixelRepresentation:1,PhotometricInterpretation:'MONOCHROME2',SamplesPerPixel:1,
        ImageOrientationPatient:g.direction_lps.slice(0,6),PixelSpacing:g.spacing.slice(0,2),Columns:g.shape[0],Rows:g.shape[1],
        voiLut:[{windowWidth:400,windowCenter:100}],VOILUTFunction:'LINEAR'}};
    this.volume=core.volumeLoader.createLocalVolume(volumeId,{...options,scalarData:new Float32Array(ct)});
    this.mask=core.volumeLoader.createLocalLabelmapVolume({...options,metadata:{...options.metadata,Modality:'SEG'},
      scalarData:new Uint8Array(mask),referencedVolumeId:volumeId},maskId);
    // createLocalVolume 5.13 creates image-backed slices with a shared origin.
    // Supply each slice's physical plane from vtk's native indexToWorld transform.
    for(const volume of [this.volume,this.mask])volume.imageIds.forEach((imageId,k)=>{
      const plane=core.metaData.get('imagePlaneModule',imageId);
      core.utilities.genericMetadataProvider.add(imageId,{type:'imagePlaneModule',metadata:{...plane,
        frameOfReferenceUID:this.prefix,imagePositionPatient:volume.imageData!.indexToWorld([0,0,k])}});
    });
    this.engine=new core.RenderingEngine(this.prefix);
    const orientations=[core.Enums.OrientationAxis.AXIAL,core.Enums.OrientationAxis.CORONAL,core.Enums.OrientationAxis.SAGITTAL];
    this.ids=orientations.map((_,i)=>this.prefix+'-'+i);
    this.engine.setViewports(this.ids.map((viewportId,i)=>({viewportId,type:core.Enums.ViewportType.ORTHOGRAPHIC,
      element:this.elements[i],defaultOptions:{orientation:orientations[i],background:[.025,.04,.065] as Point}})));
    await core.setVolumesForViewports(this.engine,[{volumeId}],this.ids);
    this.group=tools.ToolGroupManager.createToolGroup(this.prefix);
    if(!this.group) throw Error('Не удалось создать группу инструментов');
    this.group.addTool(tools.SplineROITool.toolName,{spline:{type:tools.SplineROITool.SplineTypes.Linear,drawPreviewEnabled:true},
      calculateStats:false,preventHandleOutsideImage:true,getTextLines:()=>[]});
    [tools.BrushTool,tools.PanTool,tools.ZoomTool,tools.WindowLevelTool,tools.StackScrollTool].forEach(t=>this.group!.addTool(t.toolName));
    this.ids.forEach(id=>this.group!.addViewport(id,this.prefix));
    this.group.setToolActive(tools.PanTool.toolName,{bindings:[{mouseButton:tools.Enums.MouseBindings.Auxiliary}]});
    this.group.setToolActive(tools.ZoomTool.toolName,{bindings:[{mouseButton:tools.Enums.MouseBindings.Secondary}]});
    this.group.setToolActive(tools.StackScrollTool.toolName,{bindings:[{mouseButton:tools.Enums.MouseBindings.Wheel}]});
    // Keep the shared MPR tool group, but allow primary editing only on the
    // declared source plane. Navigation remains available in all three views.
    this.elements.forEach((el,i)=>{
      const guard=(event:Event)=>{
        if(this.readOnly||!this.editScope||!['contour','brush','erase'].includes(this.currentMode))return;
        const button=(event as MouseEvent).button;if(button!==undefined&&button!==0)return;
        const plane=this.plane(i);if(this.inScope(plane.axis,plane.slice))return;
        event.preventDefault();event.stopImmediatePropagation();
      };
      ['pointerdown','mousedown','dblclick','touchstart'].forEach(name=>{
        el.addEventListener(name,guard,{capture:true,passive:false});this.cleanup.push(()=>el.removeEventListener(name,guard,true));
      });
    });
    tools.segmentation.addSegmentations([{segmentationId:this.segId,representation:{type:tools.Enums.SegmentationRepresentations.Labelmap,data:{volumeId:maskId}}}]);
    this.ids.forEach(id=>{
      tools.segmentation.addLabelmapRepresentationToViewport(id,[{segmentationId:this.segId}]);
      tools.segmentation.activeSegmentation.setActiveSegmentation(id,this.segId);
      this.project.labels.forEach(label=>{
        const hex=label.color.replace('#','');
        tools.segmentation.config.color.setSegmentIndexColor(id,this.segId,label.id,
          [parseInt(hex.slice(0,2),16),parseInt(hex.slice(2,4),16),parseInt(hex.slice(4,6),16),255]);
      });
    });
    this.opacity(.3); this.select(1); this.window(400,100);this.brush(12);
    for(let i=0;i<3;i++) await this.jump(i,Math.floor(this.viewport(i).getNumberOfSlices()/2));
    this.baseline=new Uint8Array(this.mask.voxelManager!.getCompleteScalarDataArray!());
    this.maskHasValues=this.baseline.some(v=>v!==0);
    this.restoreContours(); this.mode('contour');
    const E=tools.Enums.Events;
    [E.ANNOTATION_COMPLETED,E.ANNOTATION_MODIFIED,E.ANNOTATION_REMOVED].forEach(name=>this.listen(name,()=>{
      if(this.suppress||this.readOnly)return;
      clearTimeout(this.timer);this.timer=setTimeout(()=>this.captureContours(),180);
    }));
    // SplineROI's add/delete callbacks invalidate rendering without consistently
    // emitting ANNOTATION_MODIFIED in 5.13. Capture completed native actions too.
    const capture=()=>{if(!this.suppress&&!this.readOnly){clearTimeout(this.timer);this.timer=setTimeout(()=>{this.captureBrush();this.captureContours();},180);}};
    this.elements.forEach(el=>[E.MOUSE_DOWN,E.MOUSE_UP,E.MOUSE_CLICK].forEach(name=>{
      el.addEventListener(name,capture,true);this.cleanup.push(()=>el.removeEventListener(name,capture,true));
    }));
    const escape=(e:KeyboardEvent)=>{if(e.key==='Escape'&&this.drawing()){
      for(const el of this.elements)tools.cancelActiveManipulations(el);
      this.restoreContours();
    }};
    window.addEventListener('keydown',escape);this.cleanup.push(()=>window.removeEventListener('keydown',escape));
    this.listen(E.SEGMENTATION_DATA_MODIFIED,()=>{if(!this.suppress&&!this.readOnly)this.captureBrush();});
    const update=()=>this.slices(this.ids.map((_,i)=>this.viewport(i).getSliceIndex()));
    this.elements.forEach(el=>{el.addEventListener(core.Enums.Events.VOLUME_NEW_IMAGE,update);this.cleanup.push(()=>el.removeEventListener(core.Enums.Events.VOLUME_NEW_IMAGE,update));});
    this.resize=new ResizeObserver(()=>{this.engine.resize(true,false);this.engine.render();});
    this.elements.forEach(el=>this.resize!.observe(el));
    this.engine.render();update();
    // A valid volume and camera do not prove that the GPU shader rendered it.
    // Keep this local: reduce pixels to a non-flat check, never export pixels.
    await new Promise<void>(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>resolve())));
    const nonFlat=this.elements.some(element=>{
      const canvas=element.querySelector('canvas');if(!canvas)return false;
      const ctx=canvas.getContext('2d');if(!ctx)return false;
      const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data,colors=new Set<number>();
      for(let i=0;i<pixels.length;i+=64){colors.add((pixels[i]<<16)|(pixels[i+1]<<8)|pixels[i+2]);if(colors.size>16)return true;}
      return false;
    });
    if(!nonFlat)throw Error('Графический драйвер не отрисовал КТ. Откройте совместимое локальное окно Heart3D на NVIDIA; чёрный экран не означает отсутствие данных.');
  }
  private listen(name:string,fn:EventListener){core.eventTarget.addEventListener(name,fn);this.listeners.push([name,fn]);}
  private viewport(i:number){return this.engine.getViewport(this.ids[i]) as core.Types.IVolumeViewport;}
  private inScope(axis:number,slice:number){return !this.editScope||(axis===this.editScope.axis&&slice===this.editScope.index);}
  private style(uid:string,label:number){const color=this.project.labels.find(l=>l.id===label)!.color;
    tools.annotation.config.style.setAnnotationStyles(uid,{color,colorSelected:color,colorHighlighted:'#ffffff',lineWidth:'2',showHandlesAlways:true});}
  private captureContours(){
    const native=tools.annotation.state.getAllAnnotations().filter((a:any)=>a.metadata.FrameOfReferenceUID===this.prefix&&a.metadata.toolName===tools.SplineROITool.toolName);
    const next=copy(this.doc); const present=new Set<string>();
    for(const a of native as any[]){
      if(!a.data.contour?.closed||a.data.handles.points.length<3)continue;
      const old=next.contours.find(c=>c.id===a.annotationUID);
      const normal=a.metadata.viewPlaneNormal as Point;
      // Orthogonal MPR slice axis is determined in the CT's physical directions.
      const d=this.project.geometry.direction_lps;
      const axis=[0,1,2].map(i=>Math.abs(normal[0]*d[i*3]+normal[1]*d[i*3+1]+normal[2]*d[i*3+2])).reduce((best,x,i,arr)=>x>arr[best]?i:best,0);
      const index=core.utilities.transformWorldToIndex(this.volume.imageData,a.data.handles.points[0]);
      const nativePoints=a.data.handles.points.map((p:Point)=>[...p] as Point);
      const c={id:a.annotationUID,label:old?.label||this.activeLabel,axis,slice:Math.round(index[axis]),
        points:old&&sameClosedPoints(old.points,nativePoints)?copy(old.points):nativePoints,closed:true,applied:old?.applied||false,replace_label_slice:old?.replace_label_slice||false};
      if(!this.inScope(c.axis,c.slice)){tools.annotation.state.removeAnnotation(c.id);continue;}
      present.add(c.id);this.style(c.id,c.label);
      if(old)next.contours[next.contours.indexOf(old)]=c;else next.contours.push(c);
    }
    next.contours=next.contours.filter(c=>present.has(c.id));
    next.operations=next.operations.filter(o=>o.kind!=='contour'||present.has(o.contour_id!));
    if(JSON.stringify(next)!==JSON.stringify(this.doc)){this.doc=next;this.changed(copy(next));}
  }
  private captureBrush(){
    const data=this.mask.voxelManager!.getCompleteScalarDataArray!();
    const indices:number[]=[],values:number[]=[];
    let restored=false;
    const planeSize=this.project.geometry.shape[0]*this.project.geometry.shape[1];
    for(let i=0;i<data.length;i++)if(data[i]!==this.baseline[i]){
      if(this.editScope&&Math.floor(i/planeSize)!==this.editScope.index){this.mask.voxelManager!.setAtIndex(i,this.baseline[i]);restored=true;continue;}
      indices.push(i);values.push(data[i]);
    }
    if(restored){this.suppress=true;this.mask.modified();tools.segmentation.triggerSegmentationEvents.triggerSegmentationDataModified(this.segId);this.engine.render();this.suppress=false;}
    if(!indices.length)return;
    indices.forEach((index,i)=>{this.baseline[index]=values[i];});
    this.maskHasValues=this.baseline.some(v=>v!==0);
    this.doc.operations.push({kind:'brush',indices,values});this.changed(copy(this.doc));
  }
  private restoreContours(){
    this.suppress=true;
    for(const a of tools.annotation.state.getAllAnnotations())if(a.metadata?.FrameOfReferenceUID===this.prefix)tools.annotation.state.removeAnnotation(a.annotationUID!);
    for(const c of this.doc.contours){
      const i=({2:0,1:1,0:2} as Record<number,number>)[c.axis];
      // Native tools mutate handles in place. Never alias the saved DTO points:
      // otherwise capture would compare an already-mutated document with itself.
      tools.SplineROITool.hydrate(this.ids[i],copy(c.points),{annotationUID:c.id,splineType:tools.SplineROITool.SplineTypes.Linear,
        toolInstance:this.group!.getToolInstance(tools.SplineROITool.toolName)});
      this.style(c.id,c.label);
      tools.annotation.locking.setAnnotationLocked(c.id,this.readOnly||!this.inScope(c.axis,c.slice));
    }
    this.applyContourVisibility();
    this.suppress=false;this.engine.render();
  }
  restore(document:AnnotationDocument){this.doc=copy(document);this.restoreContours();}
  sync(document:AnnotationDocument){this.doc=copy(document);}
  setMask(buffer:ArrayBuffer){
    this.suppress=true;
    const data=new Uint8Array(buffer);
    const vm=this.mask.voxelManager!;
    // Local volumes use image-backed voxel managers; setAtIndex updates that cache.
    for(let i=0;i<data.length;i++)if(data[i]!==this.baseline[i])vm.setAtIndex(i,data[i]);
    this.baseline=data.slice();this.maskHasValues=data.some(v=>v!==0);this.mask.modified();
    tools.segmentation.triggerSegmentationEvents.triggerSegmentationDataModified(this.segId);
    this.engine.render();this.suppress=false;
  }
  select(label:number){this.activeLabel=label;tools.segmentation.segmentIndex.setActiveSegmentIndex(this.segId,label);}
  private applyContourVisibility(){this.doc.contours.forEach(c=>tools.annotation.visibility.setAnnotationVisibility(c.id,this.contoursVisible&&!this.hiddenLabels.has(c.label)));}
  showContours(visible:boolean){this.contoursVisible=visible;this.applyContourVisibility();this.engine.render();}
  plane(i=0){
    const vp=this.viewport(i),normal=vp.getCamera().viewPlaneNormal!,d=this.project.geometry.direction_lps;
    const dots=[0,1,2].map(axis=>Math.abs(normal[0]*d[axis*3]+normal[1]*d[axis*3+1]+normal[2]*d[axis*3+2]));
    const axis=dots.indexOf(Math.max(...dots));
    return {axis,slice:Math.round(core.utilities.transformWorldToIndex(this.volume.imageData,vp.getCamera().focalPoint!)[axis])};
  }
  viewSnapshot(i:number,width:number,level:number):AnatomyLocator {
    if(!Number.isInteger(i)||i<0||i>2||!this.volume?.imageData)throw Error('Выберите доступную плоскость CT.');
    if(this.maskHasValues||this.doc.contours.some(c=>tools.annotation.visibility.isAnnotationVisible(c.id)))throw Error('Сначала откройте анатомический разбор в режиме «Только CT» без масок и контуров.');
    const vp=this.viewport(i),camera=vp.getCamera();
    const focal=[...camera.focalPoint!] as Point,normal=[...camera.viewPlaneNormal!] as Point;
    // Keep subvoxel anchors: transformWorldToIndex rounds all three coordinates.
    const anchor=[...this.volume.imageData.worldToIndex(focal)] as Point;
    if([...focal,...normal,...anchor].some(v=>!Number.isFinite(v))||anchor.some((v,j)=>v<0||v>this.project.geometry.shape[j]-1))
      throw Error('Центр вида находится вне CT. Сдвиньте вид к области исследования.');
    const range=vp.getProperties()?.voiRange;
    const result:AnatomyLocator={view:(['AX','COR','SAG'] as const)[i],anchor_ijk:anchor,
      focal_point_lps_mm:focal,normal_lps:normal,window_width:range?range.upper-range.lower:width,
      window_level:range?(range.upper+range.lower)/2:level,overlay_visible:false};
    const d=this.project.geometry.direction_lps,normalLength=Math.hypot(...normal);
    const dots=[0,1,2].map(axis=>Math.abs(normal.reduce((sum,v,j)=>sum+v*d[axis*3+j],0))/normalLength);
    const axis=dots.indexOf(Math.max(...dots)),index=Math.round(anchor[axis]);
    // Oblique planes retain their physical focal point and normal. A rounded
    // voxel index alone must never claim that they are native CT slices.
    if(dots[axis]>1-1e-6&&Math.abs(anchor[axis]-index)<1e-3){result.native_axis=axis;result.native_index=index;}
    return result;
  }
  focusViewSnapshot(locator:AnatomyLocator){
    const i=['AX','COR','SAG'].indexOf(locator.view),g=this.project.geometry;
    const point=locator.focal_point_lps_mm,normal=locator.normal_lps,anchor=locator.anchor_ijk;
    if(i<0||!this.volume?.imageData||[point,normal,anchor].some(p=>!Array.isArray(p)||p.length!==3||p.some(v=>!Number.isFinite(v)))||
      anchor.some((v,j)=>v<-.001||v>g.shape[j]-1+.001)||Math.abs(Math.hypot(...normal)-1)>.001||
      !Number.isFinite(locator.window_width)||locator.window_width<=0||!Number.isFinite(locator.window_level))throw Error('Сохранённая плоскость не соответствует геометрии CT.');
    const expected=this.volume.imageData.worldToIndex(point) as Point;
    if(expected.some((v,j)=>Math.abs(v-anchor[j])>1e-3))throw Error('Координаты записи не соответствуют текущей CT.');
    if(locator.native_axis!==undefined||locator.native_index!==undefined){
      const axis=locator.native_axis,index=locator.native_index;
      if(axis===undefined||index===undefined||!Number.isInteger(axis)||axis<0||axis>2||!Number.isInteger(index)||
        Math.abs(anchor[axis]-index)>1e-3||Math.abs(normal.reduce((s,v,j)=>s+v*g.direction_lps[axis*3+j],0))<1-1e-6)
        throw Error('Исходный срез записи не соответствует её физической плоскости.');
    }
    const vp=this.viewport(i),camera=vp.getCamera(),distance=Math.hypot(...camera.position!.map((v,j)=>v-camera.focalPoint![j]));
    let up=camera.viewUp!;
    const dot=up.reduce((s,v,j)=>s+v*normal[j],0);
    up=up.map((v,j)=>v-dot*normal[j]) as Point;
    if(Math.hypot(...up)<1e-6){const seed:Point=Math.abs(normal[2])<.9?[0,0,1]:[0,1,0];const projection=seed.reduce((s,v,j)=>s+v*normal[j],0);up=seed.map((v,j)=>v-projection*normal[j]) as Point;}
    const length=Math.hypot(...up);up=up.map(v=>v/length) as Point;
    this.anatomyCameras ||= this.ids.map((_,j)=>copy(this.viewport(j).getCamera()));
    vp.setCamera({focalPoint:[...point] as Point,viewPlaneNormal:[...normal] as Point,viewUp:up,
      position:point.map((v,j)=>v+distance*normal[j]) as Point});
    vp.setProperties({voiRange:{lower:locator.window_level-locator.window_width/2,upper:locator.window_level+locator.window_width/2}});
    this.engine.render();this.slices(this.ids.map((_,j)=>this.viewport(j).getSliceIndex()));
  }
  restoreAnnotationViews(){
    if(!this.anatomyCameras)return;
    this.anatomyCameras.forEach((camera,i)=>this.viewport(i).setCamera(camera));this.anatomyCameras=undefined;
    this.engine.render();this.slices(this.ids.map((_,j)=>this.viewport(j).getSliceIndex()));
  }
  focusSlice(axis:number,slice:number){
    const i=({2:0,1:1,0:2} as Record<number,number>)[axis];
    const xyz=this.project.geometry.shape.map(n=>(n-1)/2);xyz[axis]=slice;
    const point=this.volume.imageData!.indexToWorld(xyz as Point),vp=this.viewport(i),c=vp.getCamera(),n=c.viewPlaneNormal!,f=c.focalPoint!,p=c.position!;
    const delta=n.reduce((sum,v,j)=>sum+v*(point[j]-f[j]),0);
    vp.setCamera({focalPoint:f.map((v,j)=>v+delta*n[j]) as Point,position:p.map((v,j)=>v+delta*n[j]) as Point});this.engine.render();
    this.slices(this.ids.map((_,j)=>this.viewport(j).getSliceIndex()));
  }
  mode(mode:Mode){
    this.currentMode=mode;
    if(!this.group)return;
    const S=tools.SplineROITool.toolName,B=tools.BrushTool.toolName;
    [S,B,tools.PanTool.toolName,tools.ZoomTool.toolName,tools.WindowLevelTool.toolName].forEach(name=>this.group!.setToolPassive(name,{removeAllBindings:true}));
    this.group.setToolActive(tools.PanTool.toolName,{bindings:[{mouseButton:tools.Enums.MouseBindings.Auxiliary}]});
    this.group.setToolActive(tools.ZoomTool.toolName,{bindings:[{mouseButton:tools.Enums.MouseBindings.Secondary}]});
    const name=({contour:S,brush:B,erase:B,pan:tools.PanTool.toolName,zoom:tools.ZoomTool.toolName,window:tools.WindowLevelTool.toolName})[mode];
    if(this.readOnly&&['contour','brush','erase'].includes(mode)){this.group.setToolEnabled(S);this.group.setToolDisabled(B);return;}
    if(mode==='brush'||mode==='erase')this.group.setActiveStrategy(B,mode==='erase'?'ERASE_INSIDE_CIRCLE':'FILL_INSIDE_CIRCLE');
    this.group.setToolActive(name,{bindings:[{mouseButton:tools.Enums.MouseBindings.Primary}]});
  }
  lock(value:boolean){this.readOnly=value;for(const c of this.doc.contours)tools.annotation.locking.setAnnotationLocked(c.id,value||!this.inScope(c.axis,c.slice));this.mode(value?'pan':'contour');}
  brush(size:number){tools.utilities.segmentation.setBrushSizeForToolGroup(this.prefix,size);}
  window(width:number,level:number){this.ids.forEach((_,i)=>this.viewport(i).setProperties({voiRange:{lower:level-width/2,upper:level+width/2}}));this.engine.render();}
  opacity(value:number){tools.segmentation.segmentationStyle.setStyle({type:tools.Enums.SegmentationRepresentations.Labelmap,segmentationId:this.segId},{fillAlpha:value,outlineWidth:1,renderOutline:true});this.engine.render();}
  visible(label:number,value:boolean){this.ids.forEach(id=>tools.segmentation.config.visibility.setSegmentIndexVisibility(id,{segmentationId:this.segId,type:tools.Enums.SegmentationRepresentations.Labelmap},label,value));
    if(value)this.hiddenLabels.delete(label);else this.hiddenLabels.add(label);
    this.applyContourVisibility();this.engine.render();}
  async jump(i:number,index:number){await core.utilities.jumpToSlice(this.elements[i],{imageIndex:index});this.slices(this.ids.map((_,j)=>this.viewport(j).getSliceIndex()));}
  count(i:number){return this.viewport(i).getNumberOfSlices();}
  async focus(id:string){const c=this.doc.contours.find(c=>c.id===id);if(!c)return;
    const i=({2:0,1:1,0:2} as Record<number,number>)[c.axis];
    const target=c.points[0];const vp=this.viewport(i);const camera=vp.getCamera();
    // Move only along the view normal; Cornerstone retains zoom and in-plane pan.
    const n=camera.viewPlaneNormal!,f=camera.focalPoint!,p=camera.position!;
    const delta=n.reduce((sum,v,j)=>sum+v*(target[j]-f[j]),0);
    vp.setCamera({focalPoint:f.map((v,j)=>v+delta*n[j]) as Point,position:p.map((v,j)=>v+delta*n[j]) as Point});
    tools.annotation.selection.setAnnotationSelected(id,true);this.engine.render();
    this.slices(this.ids.map((_,j)=>this.viewport(j).getSliceIndex()));
  }
  drawing(){return !!this.group?.getToolInstance(tools.SplineROITool.toolName)?.isDrawing;}
  destroy(){clearTimeout(this.timer);this.listeners.forEach(([n,f])=>core.eventTarget.removeEventListener(n,f));this.cleanup.forEach(f=>f());this.resize?.disconnect();
    if(!this.engine)return;tools.ToolGroupManager.destroyToolGroup(this.prefix);tools.segmentation.removeSegmentation(this.segId);
    for(const a of tools.annotation.state.getAllAnnotations())if(a.metadata?.FrameOfReferenceUID===this.prefix)tools.annotation.state.removeAnnotation(a.annotationUID!);
    this.engine.destroy();this.volume?.removeFromCache();this.mask?.removeFromCache();}
}
