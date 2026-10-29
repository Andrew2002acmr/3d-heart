import {useEffect,useRef,useState} from 'react';
import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkXMLPolyDataReader from '@kitware/vtk.js/IO/XML/XMLPolyDataReader';
import {api,type Project} from './api';

type Structure={label:number,short_name:string,color:string,voxels:number,components_26_connected:number,touches_scan_boundary:boolean};
export function SurfaceView({project,versionId,close}:{project:Project,versionId:string,close:()=>void}){
  const ref=useRef<HTMLDivElement>(null),actors=useRef<Record<number,ReturnType<typeof vtkActor.newInstance>>>({});
  const render=useRef<()=>void>(()=>{}),[rows,setRows]=useState<Structure[]>([]),[error,setError]=useState(''),[busy,setBusy]=useState(true);
  useEffect(()=>{let live=true;const view=vtkGenericRenderWindow.newInstance({background:[.04,.075,.11]});
    view.setContainer(ref.current!);const renderer=view.getRenderer(),rw=view.getRenderWindow();render.current=()=>rw.render();
    const objects:Array<{delete:()=>void}>=[];const resize=new ResizeObserver(()=>view.resize());resize.observe(ref.current!);view.resize();
    (async()=>{try{
      const data=await api<{structures:Structure[]}>(`/projects/${project.id}/reconstruction/${versionId}`);
      for(const row of data.structures){
        const bytes=await api<ArrayBuffer>(`/projects/${project.id}/mesh/${versionId}/${row.label}`);if(!live)return;
        const reader=vtkXMLPolyDataReader.newInstance();reader.parseAsArrayBuffer(bytes);
        const mapper=vtkMapper.newInstance({scalarVisibility:false});mapper.setInputData(reader.getOutputData());
        const actor=vtkActor.newInstance();actor.setMapper(mapper);
        const hex=row.color.replace('#','');actor.getProperty().setColor(...[0,2,4].map(i=>parseInt(hex.slice(i,i+2),16)/255) as [number,number,number]);
        actor.getProperty().setOpacity(.85);renderer.addActor(actor);actors.current[row.label]=actor;objects.push(reader,mapper,actor);
      }
      if(live){setRows(data.structures);renderer.resetCamera();rw.render();setBusy(false);}
    }catch(e){if(live){setError(String(e));setBusy(false);}}})();
    return()=>{live=false;resize.disconnect();view.delete();objects.forEach(o=>o.delete());actors.current={};};
  },[versionId]);
  return <div className="modal"><section className="surface-dialog"><header><strong>3D · {project.study_id} · RAS / мм</strong><button onClick={close}>Закрыть 3D</button></header><div className="surface-main"><div className="surface-canvas" ref={ref} data-testid="surface-canvas"/><aside><p>Левая кнопка — вращение.<br/>Колесо — масштаб.<br/>Shift + перетаскивание — сдвиг.</p>{busy&&<p>Построение исходных поверхностей…</p>}{error&&<p role="alert">{error}</p>}{rows.map(row=><label key={row.label}><input type="checkbox" defaultChecked onChange={e=>{actors.current[row.label]?.setVisibility(e.target.checked);render.current();}}/>{row.short_name} · {row.voxels.toLocaleString()} vox<small>{row.components_26_connected} компонент{row.touches_scan_boundary?' · касается границы CT (искусственная крышка)':''}</small></label>)}<p>Все компоненты сохранены. Сглаживания и заполнения отсутствующих классов нет.</p><p>AI prediction не является экспертной разметкой.</p></aside></div></section></div>;
}
