import {useEffect,useRef,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {api,post,session,copy,sliceScope,type User,type Project,type AnnotationDocument,type Contour} from './api';
import {Viewer,type Mode} from './viewer';
import './style.css';
import {PredictionControls} from './PredictionControls';
import {SurfaceView} from './SurfaceView';
import {ReviewControls} from './ReviewControls';
import {AnatomyReview,type AnatomyReviewHandle} from './AnatomyReview';

const status:Record<string,string>={not_started:'Не начато',in_progress:'В работе',submitted:'На проверке',needs_revision:'На доработке',approved:'Утверждено'};
function Brand(){return <div className="brand"><span className="brandmark">♥</span><span>Heart<span className="accent">3D</span><small>ANNOTATION WORKSPACE</small></span></div>}
function App(){
  const [user,setUser]=useState<User|null>(null),[checking,setChecking]=useState(true),[project,setProject]=useState<Project|null>(null);
  const [projects,setProjects]=useState<Project[]>([]),[error,setError]=useState('');
  async function load(){try{setProjects(await api<Project[]>('/projects'));}catch(e){setError(String(e));}}
  useEffect(()=>{api<User>('/auth/me').then(u=>{session(u);setUser(u);}).catch(()=>{}).finally(()=>setChecking(false));},[]);
  async function openProject(id:string){try{const anatomy=new URLSearchParams(window.location.search).get('view')==='anatomy';const p=await api<Project>('/projects/'+id);window.history.replaceState(null,'','?project='+encodeURIComponent(id)+(anatomy?'&view=anatomy':''));setProject(p);}catch(e){setError(String(e));}}
  useEffect(()=>{if(user){void load();const id=new URLSearchParams(window.location.search).get('project');if(id)void openProject(id);}},[user]);
  async function logout(){await post('/auth/logout');session(null);setUser(null);setProject(null);setProjects([]);window.history.replaceState(null,'',window.location.pathname);}
  if(checking)return <div className="loading">Открываем рабочее пространство…</div>;
  if(!user)return <Login onLogin={u=>{session(u);setUser(u);}}/>;
  if(project)return <Editor key={project.id+':'+(project.version_id||'new')} initial={project} user={user} replace={setProject} back={()=>{setProject(null);window.history.replaceState(null,'',window.location.pathname);void load();}} logout={logout}/>;
  return <div className="dashboard"><header><Brand/><div className="user"><span className="avatar">{user.username.slice(0,1).toUpperCase()}</span><span>{user.username}<small>{user.role==='resident'?'Ординатор':user.role}</small></span><button className="quiet" onClick={logout}>Выйти</button></div></header>
    <main><div className="eyebrow">РАБОЧИЙ КАБИНЕТ</div><div className="page-title"><div><h1>Мои исследования</h1><p>Контуры, которые становятся точной анатомией.</p></div><span className="pill">Локальный исследовательский стенд</span></div>
      <div className="summary"><div><strong>{projects.length.toString().padStart(2,'0')}</strong><span>назначено</span></div><div><strong>{projects.filter(p=>p.status==='in_progress').length.toString().padStart(2,'0')}</strong><span>в работе</span></div><div><strong>07</strong><span>структур сердца</span></div></div>
      {error&&<p role="alert" className="error">{error}</p>}<div className="section-line"><h2>Доступные исследования</h2><span>Локальное хранение · контролируемый доступ</span></div>
      {projects.map(p=><article className="study-card" key={p.id}><div className="study-preview"><span>AXIAL</span><div className="phantom-icon">♥</div><small>{p.synthetic?'SYNTHETIC CT':'RESEARCH CT'}</small></div><div className="study-info"><div className="card-top"><span className="eyebrow">{p.study_id}</span><span className={'status '+p.status}>{status[p.status]}</span></div><h2>{p.patient_id}</h2><p>{p.synthetic?'Синтетический объём':'Обезличенный CT'} · {p.geometry.shape.join(' × ')} · {p.geometry.spacing.map(n=>n.toFixed(1)).join(' × ')} мм</p><div className="progress"><i style={{width:`${p.progress_classes/7*100}%`}}/></div><div className="card-bottom"><span>{p.progress_classes} из 7 структур с подтверждённым контуром<br/><small>{p.version_number?`Последнее сохранение: версия ${p.version_number}`:'Сохранённых версий пока нет'}</small></span><button className="primary" onClick={()=>void openProject(p.id)}>Открыть разметку <b>↗</b></button></div></div></article>)}
      {!projects.length&&<div className="empty">Пока нет назначений. Владелец стенда может назначить проект через CLI.</div>}<footer>Heart3D Annotation · MVP 0.1 <span>Локальные исследования · без публикации</span></footer></main></div>;
}
function Login({onLogin}:{onLogin:(u:User)=>void}){
  const [error,setError]=useState(''),[busy,setBusy]=useState(false);
  return <div className="login"><div className="login-art"><Brand/><div className="orb">♥</div><div><div className="eyebrow">ОТ СРЕЗА К СТРУКТУРЕ</div><h1>Увидеть.<br/>Обвести.<br/><span>Восстановить.</span></h1><p>Рабочее пространство экспертной<br/>разметки анатомии сердца.</p></div><small>Локальный исследовательский стенд · Heart3D</small></div><div className="login-form"><div><span className="pill">Закрытый доступ</span><h2>Вход в кабинет</h2><p>Используйте учётную запись, созданную владельцем стенда.</p><form onSubmit={async e=>{e.preventDefault();setError('');setBusy(true);const f=new FormData(e.currentTarget);try{onLogin(await post<User>('/auth/login',{username:f.get('username'),password:f.get('password')}));}catch{setError('Не удалось войти. Проверьте логин и пароль.');}finally{setBusy(false);}}}>
      <label>Логин<input name="username" autoComplete="username" required autoFocus/></label><label>Пароль<input name="password" type="password" autoComplete="current-password" required/></label>{error&&<p role="alert" className="error">{error}</p>}<button className="primary" disabled={busy}>{busy?'Входим…':'Войти в рабочее пространство'} <span>→</span></button></form><small className="login-note">Исследования доступны только назначенным участникам.<br/>Восстановление доступа — через владельца стенда.</small></div></div></div>;
}
function Editor({initial,user,back,logout,replace}:{initial:Project,user:User,back:()=>void,logout:()=>Promise<void>,replace:(p:Project)=>void}){
  const [project,setProject]=useState(initial),[document,setDocument]=useState(copy(initial.document)),[ready,setReady]=useState(false),[error,setError]=useState('');
  const [label,setLabel]=useState(1),[mode,setMode]=useState<Mode>('contour'),[slices,setSlices]=useState([0,0,0]),[selected,setSelected]=useState('');
  const [saveState,setSaveState]=useState('Сохранено'),[dirty,setDirty]=useState(false),[history,setHistory]=useState<AnnotationDocument[]>([]),[future,setFuture]=useState<AnnotationDocument[]>([]);
  const [width,setWidth]=useState(400),[level,setLevel]=useState(100),[opacity,setOpacity]=useState(30),[brush,setBrush]=useState(12),[hidden,setHidden]=useState<number[]>([]);
  const [solo,setSolo]=useState(false);
  const previousHidden=useRef<number[]>([]),temporaryPan=useRef<Mode|null>(null),overViewport=useRef(false);
  const refs=[useRef<HTMLDivElement>(null),useRef<HTMLDivElement>(null),useRef<HTMLDivElement>(null)];
  const viewer=useRef<Viewer|null>(null),docRef=useRef(copy(initial.document)),version=useRef(initial.version_id),saved=useRef(JSON.stringify(initial.document));
  const saving=useRef<Promise<void>|null>(null),sequence=useRef(0),alive=useRef(true),historyRef=useRef<AnnotationDocument[]>([]),futureRef=useRef<AnnotationDocument[]>([]);
  const [reviewOpen,setReviewOpen]=useState(()=>new URLSearchParams(window.location.search).get('view')==='anatomy');
  const anatomyReview=useRef<AnatomyReviewHandle>(null);
  const [comparison,setComparison]=useState(new URLSearchParams(window.location.search).get('view')==='anatomy'?'ct':(initial.version_id?'draft':(initial.predictions?.find(p=>p.provenance.variant==='anatomy_slice_draft')?.id||initial.predictions?.find(p=>p.provenance.variant==='slice_draft')?.id||initial.predictions?.[0]?.id||'draft')));
  const [surfaceVersion,setSurfaceVersion]=useState('');
  const workflowLocked=project.status==='submitted'||project.status==='approved'||user.role==='surgeon';
  const locked=workflowLocked||comparison!=='draft';
  const activeScope=sliceScope({...project,document},comparison),draftScope=sliceScope({...project,document},'draft');
  const projectRef=useRef(project);projectRef.current=project;
  async function preview(doc:AnnotationDocument){if(locked)return;const n=++sequence.current;try{const mask=await post<ArrayBuffer>(`/projects/${project.id}/preview`,doc);if(alive.current&&n===sequence.current)viewer.current?.setMask(mask);}catch(e){if(alive.current)setError(String(e));}}
  function change(doc:AnnotationDocument,track=true){
    if(track){historyRef.current=[...historyRef.current.slice(-39),copy(docRef.current)];futureRef.current=[];setHistory(historyRef.current);setFuture([]);}
    docRef.current=copy(doc);setDocument(copy(doc));viewer.current?.sync(doc);setDirty(JSON.stringify(doc)!==saved.current);setSaveState('Есть изменения');void preview(doc);
  }
  useEffect(()=>{alive.current=true;let instance:Viewer|null=null;
    (async()=>{try{const [ct,mask]=await Promise.all([api<ArrayBuffer>(`/projects/${initial.id}/volume`),comparison==='ct'?Promise.resolve(new ArrayBuffer(initial.geometry.shape.reduce((a,b)=>a*b,1))):api<ArrayBuffer>(`/projects/${initial.id}/mask${comparison==='draft'?'':'?version_id='+comparison}`)]);
      if(!alive.current)return;instance=new Viewer(initial,refs.map(r=>r.current!),doc=>change(doc),values=>setSlices(values));viewer.current=instance;
      await instance.start(ct,mask);if(!alive.current){instance.destroy();return;}instance.lock(locked);instance.showContours(comparison==='draft');const scope=sliceScope(initial,comparison);if(scope)instance.focusSlice(scope.axis,scope.index);setReady(true);
    }catch(e){instance?.destroy();viewer.current=null;if(alive.current)setError(String(e));}})();
    return()=>{alive.current=false;sequence.current++;instance?.destroy();viewer.current=null;};
  },[]);
  async function save(){
    if(saving.current){await saving.current;if(JSON.stringify(docRef.current)!==saved.current)return save();return;}
    if(JSON.stringify(docRef.current)===saved.current)return;
    const snapshot=copy(docRef.current),json=JSON.stringify(snapshot);setSaveState('Сохраняем…');
    const job=(async()=>{try{
      const result=await post<{version_id:string,version_number:number}>(`/projects/${initial.id}/drafts`,{base_version_id:version.current,ct_sha256:initial.ct_sha256,label_schema:initial.label_schema,document:snapshot});
      version.current=result.version_id;saved.current=json;
      if(alive.current){setProject(p=>({...p,version_id:result.version_id,version_number:result.version_number,status:'in_progress'}));const pending=JSON.stringify(docRef.current)!==json;setDirty(pending);setSaveState(pending?'Есть изменения':`Сохранено · v${result.version_number}`);setError('');}
    }catch(e){setSaveState('Ошибка сохранения');setError(String(e));throw e;}finally{saving.current=null;}})();
    saving.current=job;return job;
  }
  useEffect(()=>{if(!ready||!dirty||locked)return;const timer=setTimeout(()=>{void save().catch(()=>{});},2200);return()=>clearTimeout(timer);},[document,ready,dirty,locked]);
  useEffect(()=>{
    function restoreTemporaryPan(){const previous=temporaryPan.current;if(previous){temporaryPan.current=null;tool(previous);}}
    function keydown(event:KeyboardEvent){
      if(!ready||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey||event.isComposing)return;
      const target=event.target as HTMLElement|null;
      if(target?.closest('input,textarea,select,[contenteditable="true"]'))return;
      if(event.code==='Space'){
        if(!overViewport.current)return;event.preventDefault();
        if(event.repeat||temporaryPan.current||viewer.current?.drawing())return;
        temporaryPan.current=mode;tool('pan');return;
      }
      if(event.repeat||temporaryPan.current||viewer.current?.drawing())return;
      const next=({KeyH:'pan',KeyZ:'zoom',KeyV:'contour'} as Record<string,Mode>)[event.code];
      if(next&&(!locked||next!=='contour')){event.preventDefault();tool(next);}
      if(event.code==='KeyI'){event.preventDefault();toggleSolo();}
    }
    const keyup=(event:KeyboardEvent)=>{if(event.code==='Space')restoreTemporaryPan();};
    const visibility=()=>{if(window.document.hidden)restoreTemporaryPan();};
    window.addEventListener('keydown',keydown);window.addEventListener('keyup',keyup);window.addEventListener('blur',restoreTemporaryPan);window.document.addEventListener('visibilitychange',visibility);
    return()=>{window.removeEventListener('keydown',keydown);window.removeEventListener('keyup',keyup);window.removeEventListener('blur',restoreTemporaryPan);window.document.removeEventListener('visibilitychange',visibility);};
  },[ready,mode,locked,solo,label,hidden]);
  useEffect(()=>{const guard=(e:BeforeUnloadEvent)=>{if(JSON.stringify(docRef.current)!==saved.current||saving.current||viewer.current?.drawing()||anatomyReview.current?.dirty()){e.preventDefault();e.returnValue='';}};window.addEventListener('beforeunload',guard);return()=>window.removeEventListener('beforeunload',guard);},[]);
  async function leave(action:()=>void|Promise<void>){if(viewer.current?.drawing()){setError('Закончите контур нажатием на первую точку или отмените его клавишей Esc.');return false;}try{await save();if(JSON.stringify(docRef.current)!==saved.current)await save();await anatomyReview.current?.flush();await action();return true;}catch(e){setError(String(e));if(anatomyReview.current?.dirty())setReviewOpen(true);return false;}}
  function undo(redo=false){const stack=redo?futureRef:historyRef;if(!stack.current.length)return;
    const next=copy(stack.current[stack.current.length-1]);stack.current=stack.current.slice(0,-1);
    const other=redo?historyRef:futureRef;other.current=[...other.current,copy(docRef.current)];setHistory(historyRef.current);setFuture(futureRef.current);
    viewer.current?.restore(next);change(next,false);
  }
  function apply(){const next=copy(docRef.current);const target=next.contours.find(c=>c.id===selected)||next.contours.at(-1);if(!target){setError('Сначала замкните контур нажатием на первую точку.');return;}
    target.applied=true;if(!next.operations.some(o=>o.kind==='contour'&&o.contour_id===target.id))next.operations.push({kind:'contour',contour_id:target.id});setSelected(target.id);change(next);}
  function remove(){if(!selected)return;const next=copy(docRef.current);next.contours=next.contours.filter(c=>c.id!==selected);next.operations=next.operations.filter(o=>o.contour_id!==selected);viewer.current?.restore(next);change(next);setSelected('');}
  function tool(next:Mode){setMode(next);viewer.current?.mode(next);}
  function showLabels(nextHidden:number[]){setHidden(nextHidden);project.labels.forEach(l=>viewer.current?.visible(l.id,!nextHidden.includes(l.id)));}
  function chooseLabel(id:number){setLabel(id);viewer.current?.select(id);if(solo)showLabels(project.labels.filter(l=>l.id!==id).map(l=>l.id));}
  function toggleSolo(){if(solo){setSolo(false);showLabels(previousHidden.current);}else{previousHidden.current=[...hidden];setSolo(true);showLabels(project.labels.filter(l=>l.id!==label).map(l=>l.id));}}
  function toggleLabel(id:number){setSolo(false);showLabels(hidden.includes(id)?hidden.filter(x=>x!==id):[...hidden,id]);}
  async function compare(next:string){return leave(async()=>{sequence.current++;const bytes=next==='ct'?new ArrayBuffer(initial.geometry.shape.reduce((a,b)=>a*b,1)):await api<ArrayBuffer>(`/projects/${initial.id}/mask${next==='draft'?'':'?version_id='+next}`);if(next!=='ct')viewer.current?.restoreAnnotationViews();viewer.current?.setMask(bytes);viewer.current?.showContours(next==='draft');viewer.current?.lock(workflowLocked||next!=='draft');const scope=sliceScope({...project,document:docRef.current},next);if(scope)viewer.current?.focusSlice(scope.axis,scope.index);setComparison(next);setMode(next==='draft'?'contour':'pan');});}
  async function openAnatomy(){setReviewOpen(true);if(comparison!=='ct'&&!await compare('ct'))return;viewer.current?.showContours(false);const params=new URLSearchParams(window.location.search);params.set('view','anatomy');window.history.replaceState(null,'','?'+params.toString());}
  function closeAnatomy(){setReviewOpen(false);const params=new URLSearchParams(window.location.search);params.delete('view');window.history.replaceState(null,'','?'+params.toString());}
  async function adopt(boundaries=false){await leave(async()=>{const result=await post<Project>(`/projects/${initial.id}/adopt-prediction`,{version_id:comparison,base_version_id:version.current,prepare_boundaries:boundaries});replace(result);});}
  async function prepare(){try{if(viewer.current?.drawing())throw Error('Сначала завершите или отмените текущий контур.');const next=await post<AnnotationDocument>(`/projects/${initial.id}/prepare-boundaries`,docRef.current);viewer.current?.restore(next);change(next);setSelected(next.contours[0]?.id||'');tool('contour');}catch(e){setError(String(e));}}
  async function trace(){try{if(draftScope)viewer.current!.focusSlice(draftScope.axis,draftScope.index);const plane=viewer.current!.plane();const contours=await post<Contour[]>(`/projects/${initial.id}/trace-contours`,{document:docRef.current,label,...plane});if(!contours.length){setError('На этом срезе нет выбранного класса.');return;}const next=copy(docRef.current);const removed=new Set(next.contours.filter(c=>c.axis===plane.axis&&c.slice===plane.slice&&c.label===label).map(c=>c.id));next.contours=next.contours.filter(c=>!removed.has(c.id)).concat(contours);next.operations=next.operations.filter(o=>!removed.has(o.contour_id||''));next.operations.push(...contours.map(c=>({kind:'contour' as const,contour_id:c.id})));viewer.current?.restore(next);change(next);setSelected(contours[0].id);tool('contour');}catch(e){setError(String(e));}}
  function flag(){const plane=viewer.current!.plane();const next=copy(docRef.current);next.flags=[...(next.flags||[]),{id:crypto.randomUUID(),label,...plane,kind:'uncertain'}];change(next);}
  async function show3D(){if(activeScope)return;await leave(()=>{const id=comparison==='draft'?version.current:comparison;if(id)setSurfaceVersion(id);});}
  const modes:Array<[Mode,string,string]>=[['contour','⌖','Контур'],['brush','◉','Кисть'],['erase','▱','Ластик'],['pan','✥','Сдвиг'],['zoom','⌕','Масштаб'],['window','◐','Окно']];
  const sourceAxialIndex=ready&&activeScope?(viewer.current?.plane(0).slice??activeScope.index):slices[0];
  return <div className="editor"><header><button className="quiet back" onClick={()=>void leave(back)}>← Кабинет</button><Brand/><div className="study-heading"><strong>{project.patient_id}</strong><small>{project.study_id} · {project.synthetic?'синтетическая':'исследовательская'} КТ</small></div><span className={'status '+project.status}>{status[project.status]}</span><div className="editor-actions"><span data-testid="save-status" className={dirty?'unsaved':''}>{saveState}</span><button disabled={!ready||locked||!dirty} onClick={()=>void save().catch(()=>{})}>Сохранить</button><button className="quiet" onClick={()=>void leave(logout)}>Выйти</button></div></header>
    {error&&<div role="alert" className="error editor-error">{error}<button className="quiet" onClick={()=>setError('')}>×</button></div>}
    <PredictionControls project={{...project,document}} comparison={comparison} locked={locked} ready={ready} canAdopt={user.role==='resident'||user.role==='admin'} select={value=>void compare(value).catch(e=>setError(String(e)))} adopt={boundaries=>void adopt(boundaries).catch(e=>setError(String(e)))} prepare={()=>void prepare()} trace={()=>void trace()} flag={flag} focusFlag={(axis,index)=>viewer.current?.focusSlice(axis,index)} show3D={()=>void show3D().catch(e=>setError(String(e)))}/>
    <div className="anatomy-entry"><button disabled={!ready} onClick={()=>void openAnatomy().catch(e=>setError(String(e)))} aria-expanded={reviewOpen}>Анатомический разбор CT</button><span>Наблюдения в AX / COR / SAG · ход структур · решение по целевому срезу</span></div>
    {user.role==='surgeon'&&project.version_id&&<ReviewControls project={project} ready={ready} updated={replace}/>}
    {surfaceVersion&&<SurfaceView project={project} versionId={surfaceVersion} close={()=>setSurfaceVersion('')}/>}
    <div className={'editor-body'+(reviewOpen?' review-open':'')}><aside className="tools"><div className="eyebrow">ИНСТРУМЕНТЫ</div>{modes.map(([id,icon,name])=><button key={id} disabled={!ready||(locked&&['contour','brush','erase'].includes(id))} aria-label={name} aria-pressed={mode===id} className={mode===id?'tool active':'tool'} onClick={()=>tool(id)}><b>{icon}</b>{name}</button>)}<div className="tool-separator"/><button disabled={locked||!history.length} onClick={()=>undo()}>↶ Отменить</button><button disabled={locked||!future.length} onClick={()=>undo(true)}>↷ Повторить</button><label className="compact-label">Кисть · {brush} мм<input aria-label="Размер кисти" type="range" min="2" max="40" value={brush} onChange={e=>{setBrush(+e.target.value);viewer.current?.brush(+e.target.value);}}/></label><div className="tool-tip"><b>Контрольные точки</b><p>Клик — новая точка.<br/>Первая точка — замкнуть.<br/>Потяните точку — изменить.</p><p>Shift + клик — добавить.<br/>Ctrl + клик — удалить.</p><small>H — сдвиг · Z — масштаб<br/>V — контур · I — изоляция<br/>Пробел + потянуть — сдвиг<br/>Колесо — срез<br/>Средняя кнопка — сдвиг<br/>Правая — масштаб</small></div></aside>
      <section className="view-area"><div className="view-toolbar"><span><i className="live-dot"/> MPR · исходная сетка</span><label>W <input aria-label="Ширина окна" type="number" min="1" max="4000" value={width} onChange={e=>{const w=Math.max(1,+e.target.value);setWidth(w);viewer.current?.window(w,level);}}/></label><label>L <input aria-label="Уровень окна" type="number" value={level} onChange={e=>{setLevel(+e.target.value);viewer.current?.window(width,+e.target.value);}}/></label><span className="grid-info">{project.geometry.spacing.map(n=>n.toFixed(1)).join(' × ')} мм</span></div>
      <div className="mpr-grid">{['Аксиальная','Корональная','Сагиттальная'].map((name,i)=>{
        const sourceAxial=!!activeScope&&i===0;
        const sliceIndex=sourceAxial?sourceAxialIndex:slices[i];
        const sliceCount=ready?(sourceAxial?project.geometry.shape[2]:(viewer.current?.count(i)||1)):2;
        return <div key={name} className={'pane pane-'+i}><div className="pane-label"><span>{['AX','COR','SAG'][i]}</span> {name}<small>{ready?`${sliceIndex+1} / ${sliceCount}`:'Загрузка…'}</small></div>{activeScope&&<div className="scope-pane-note">{i===0?`Правка: исходный аксиальный срез ${activeScope.index+1}`:'Только контекст'}</div>}<div className="viewport" data-testid={'viewport-'+i} ref={refs[i]} onPointerEnter={()=>{overViewport.current=true;}} onPointerLeave={()=>{overViewport.current=false;}} onContextMenu={e=>e.preventDefault()}/><input aria-label={'Срез '+name} className="slice-slider" type="range" min="0" max={sliceCount-1} value={sliceIndex} disabled={!ready} onChange={e=>{if(sourceAxial)viewer.current?.focusSlice(2,+e.target.value);else void viewer.current?.jump(i,+e.target.value);}}/><div className="orientation"><span>{i===2?'A':'R'}</span><span>{i===0?'A':'S'}</span></div></div>;
      })}</div>
      <div className="view-footer"><span>{ready?'● Геометрия проверена':'Подготовка объёма…'}</span><span>{project.geometry.shape.join(' × ')} вокселей · LPS / мм</span><span>{project.synthetic?'Синтетические данные':'Исследовательские данные · AI не утверждено'}</span></div></section>
      <aside className="structures"><div className="eyebrow">АНАТОМИЯ СЕРДЦА</div><h2>{activeScope?'Классы схемы':'Структуры'} <span>{project.labels.length}</span></h2>{activeScope&&<small className="scope-class-note">Наличие всех классов на одном срезе не подтверждено.</small>}<div className="structure-visibility"><button aria-pressed={solo} disabled={!ready} onClick={toggleSolo}>Только выбранная</button><button disabled={!ready} onClick={()=>{setSolo(false);showLabels([]);}}>Показать все</button></div><small className="visibility-help">I — изоляция выбранной структуры. ◉ скрывает маску и её контуры.</small><div className="label-list">{project.labels.map(l=><div key={l.id} className={'label-row '+(label===l.id?'selected':'')} style={{'--label-color':l.color} as React.CSSProperties}><button className="label-choice" onClick={()=>chooseLabel(l.id)}><i/><span><strong>{l.code}</strong><small>{l.name}</small></span><em>{document.contours.filter(c=>c.label===l.id).length||'—'}</em></button><button className="eye" aria-label={'Видимость '+l.code} aria-pressed={!hidden.includes(l.id)} onClick={()=>toggleLabel(l.id)}>{hidden.includes(l.id)?'○':'◉'}</button></div>)}</div>
      <label className="compact-label">Прозрачность маски <span>{opacity}%</span><input aria-label="Прозрачность маски" type="range" min="0" max="80" value={opacity} onChange={e=>{setOpacity(+e.target.value);viewer.current?.opacity(+e.target.value/100);}}/></label>
      <div className="contour-heading"><h3>Контуры <span data-testid="contour-count">{document.contours.length}</span></h3><button className="quiet" disabled={!selected||locked} onClick={remove}>Удалить</button></div><div className="contours">{document.contours.map((c,i)=><button key={c.id} data-testid="contour-item" className={c.id===selected?'contour selected':'contour'} onClick={()=>{setSelected(c.id);chooseLabel(c.label);void viewer.current?.focus(c.id);tool('contour');}}><i style={{background:project.labels.find(l=>l.id===c.label)?.color}}/><span>{project.labels.find(l=>l.id===c.label)?.code} · контур {i+1}<small>{c.points.length} точек · срез {c.slice+1}</small></span><b>{c.applied?'✓':'○'}</b></button>)}{!document.contours.length&&<p className="empty-contour">{draftScope?`Выберите класс и обведите его на аксиальном срезе ${draftScope.index+1}.`:'Выберите структуру и обведите её на любом срезе.'}</p>}</div><button className="primary apply" disabled={locked||!document.contours.length} onClick={apply}>✓ Применить контур</button><small className="help">{!locked&&!!document.contours.length?'Выберите контур в списке и перетаскивайте его точки. Shift + клик — добавить, Ctrl + клик — удалить точку. Правки сохраняются автоматически.':'Маска пересчитывается в исходной сетке КТ. Точки остаются редактируемыми.'}</small>
      <div className="export"><span>Версия {project.version_number||'—'} · {initial.label_schema}</span><a aria-disabled={!project.version_id||dirty} title={dirty?'Дождитесь сохранения перед экспортом':undefined} className={!project.version_id||dirty?'disabled':''} href={project.version_id&&!dirty?`/api/projects/${project.id}/export/${project.version_id}`:undefined} download>↓ Экспорт NIfTI</a><button disabled={locked||!(document.operations.length||document.base_prediction_id)} onClick={async()=>{try{await save();await anatomyReview.current?.flush();await post(`/projects/${project.id}/submit`,{version_id:version.current});setProject(p=>({...p,status:'submitted'}));viewer.current?.lock(true);setMode('pan');}catch(e){setError(String(e));}}}>Передать на проверку →</button></div></aside>
      <AnatomyReview ref={anatomyReview} project={project} open={reviewOpen} close={closeAnatomy} readOnly={workflowLocked} viewerReady={ready} masksHidden={comparison==='ct'}
        getView={i=>{if(!viewer.current||comparison!=='ct')throw Error('Откройте только CT без масок.');return viewer.current.viewSnapshot(i,width,level);}}
        locate={async locator=>{if(!viewer.current)throw Error('CT ещё не готова.');if(comparison!=='ct'&&!await compare('ct'))throw Error('Не удалось открыть CT без масок.');viewer.current.showContours(false);viewer.current.focusViewSnapshot(locator);setWidth(locator.window_width);setLevel(locator.window_level);}}/>
    </div></div>;
}
createRoot(window.document.getElementById('root')!).render(<App/>);
