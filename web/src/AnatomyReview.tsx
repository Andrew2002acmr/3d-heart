import {forwardRef,useEffect,useImperativeHandle,useRef,useState} from 'react';
import {api,copy,post,type AnatomyConfidence,type AnatomyDecision,type AnatomyDocument,type AnatomyLocator,
  type AnatomyObservation,type AnatomyRelation,type AnatomyReviewVersion,type AnatomyTrack,type Project} from './api';

export type AnatomyReviewHandle = {flush:()=>Promise<void>,dirty:()=>boolean};
type Props = {project:Project,open:boolean,close:()=>void,readOnly:boolean,viewerReady:boolean,masksHidden:boolean,
  getView:(i:number)=>AnatomyLocator,locate:(locator:AnatomyLocator)=>void|Promise<void>};
const emptyDocument=():AnatomyDocument=>({schema_version:1,target:null,observations:[],tracks:[],decisions:[]});
const confidenceNames:Record<AnatomyConfidence,string>={low:'Низкая',medium:'Средняя',high:'Высокая'};
const sourceNames:Record<AnatomyObservation['source'],string>={human:'Человек',local_model:'Локальная модель',cloud_model:'Облачная модель'};
const decisionNames:Record<AnatomyDecision['state'],string>={unassessed:'Не оценено',delineated:'Граница намечена',uncertain:'Неуверенно',not_visible:'Не видно',not_assessable:'Оценить невозможно'};
const lines=(value:string)=>value.split(/\n/).map(s=>s.trim()).filter(Boolean);
const toggle=(values:string[],id:string)=>values.includes(id)?values.filter(v=>v!==id):[...values,id];
function Confidence({value,onChange}:{value:AnatomyConfidence,onChange:(value:AnatomyConfidence)=>void}){
  return <label>Уверенность<select value={value} onChange={e=>onChange(e.target.value as AnatomyConfidence)}>{Object.entries(confidenceNames).map(([id,name])=><option key={id} value={id}>{name}</option>)}</select></label>;
}
export const AnatomyReview=forwardRef<AnatomyReviewHandle,Props>(function AnatomyReview({project,open,close,readOnly,viewerReady,masksHidden,getView,locate},ref){
  const [document,setDocument]=useState<AnatomyDocument>(emptyDocument),[version,setVersion]=useState<AnatomyReviewVersion|null>(null);
  const [loaded,setLoaded]=useState(false),[dirty,setDirty]=useState(false),[busy,setBusy]=useState(false),[error,setError]=useState(''),[conflict,setConflict]=useState(false);
  const [tab,setTab]=useState<'observations'|'tracks'|'decisions'>('observations'),[view,setView]=useState(0),[target,setTarget]=useState('1');
  const docRef=useRef(document),versionRef=useRef<string|null>(null),savedRef=useRef(JSON.stringify(document)),jobRef=useRef<Promise<void>|null>(null),alive=useRef(true);
  const [capture,setCapture]=useState<AnatomyLocator|null>(null),[editing,setEditing]=useState('');
  const [kind,setKind]=useState<AnatomyObservation['kind']>('observation'),[structure,setStructure]=useState(0),[statement,setStatement]=useState(''),[features,setFeatures]=useState('');
  const [confidence,setConfidence]=useState<AnatomyConfidence>('low'),[uncertainty,setUncertainty]=useState(''),[evidence,setEvidence]=useState<string[]>([]);
  const [trackClass,setTrackClass]=useState(1),[trackIds,setTrackIds]=useState<string[]>([]),[relations,setRelations]=useState<AnatomyRelation[]>([]),[trackEditing,setTrackEditing]=useState('');
  const [from,setFrom]=useState(''),[to,setTo]=useState(''),[relationKind,setRelationKind]=useState<AnatomyRelation['kind']>('continuation'),[gap,setGap]=useState('');
  const [decisionClass,setDecisionClass]=useState(1),[decision,setDecision]=useState<AnatomyDecision>(()=>newDecision(1));
  const observationBase=useRef(''),pendingForms=useRef({observation:false,track:false,decision:false});
  const observationForm={kind,structure,statement,features,confidence,uncertainty,evidence};
  const originalTrack=document.tracks.find(t=>t.id===trackEditing);
  const pendingObservation=!!capture&&JSON.stringify(observationForm)!==observationBase.current;
  const pendingTrack=JSON.stringify({structure:trackClass,observation_ids:trackIds,relations})!==JSON.stringify({structure:originalTrack?.structure??trackClass,observation_ids:originalTrack?.observation_ids??[],relations:originalTrack?.relations??[]})||!!from||!!to||!!gap||relationKind!=='continuation';
  const pendingDecision=JSON.stringify(decision)!==JSON.stringify(newDecision(decisionClass,document));
  const hasPendingForm=pendingObservation||pendingTrack||pendingDecision;
  pendingForms.current={observation:pendingObservation,track:pendingTrack,decision:pendingDecision};
  const writable=loaded&&!readOnly;
  const editReferenced=!!editing&&(document.observations.some(n=>n.evidence_refs.includes(editing))||document.tracks.some(t=>t.observation_ids.includes(editing))||document.decisions.some(d=>d.supporting_observation_ids.includes(editing)||d.counterevidence_ids.includes(editing)));
  const editTrackClasses=document.tracks.filter(t=>t.observation_ids.includes(editing)).map(t=>t.structure);
  const trackReferenced=!!trackEditing&&document.decisions.some(d=>d.track_ids.includes(trackEditing));
  const targetChangesDecisions=!!document.target&&document.target.index!==+target-1&&document.decisions.length>0;
  const labelName=(id:number)=>id===0?'Неизвестная структура':project.labels.find(l=>l.id===id)?.code||String(id);
  const noteName=(id:string)=>{const n=document.observations.find(n=>n.id===id);return n?`${n.kind==='hypothesis'?'Гипотеза':'Наблюдение'} · ${n.locator.view} · ${n.statement.slice(0,64)}`:'Запись не найдена';};
  function pendingMessage(form?:keyof typeof pendingForms.current){const pending=pendingForms.current;
    if((!form||form==='observation')&&pending.observation)return 'Добавьте или отмените текущую запись';
    if((!form||form==='track')&&pending.track)return 'Запишите или отмените текущий ход структуры';
    if((!form||form==='decision')&&pending.decision)return 'Запишите или отмените текущее решение';
    return '';
  }
  function blockPending(form?:keyof typeof pendingForms.current){const message=pendingMessage(form);if(message)setError(message);return !!message;}
  async function flush(){let message=pendingMessage();if(message){setError(message);throw Error(message);}await save();message=pendingMessage();if(message){setError(message);throw Error(message);}}
  function update(next:AnatomyDocument){if(!writable)return;docRef.current=next;setDocument(next);setDirty(JSON.stringify(next)!==savedRef.current);}
  async function load(){
    if(blockPending())return;
    setError('');
    try{const result=await api<AnatomyReviewVersion>(`/projects/${project.id}/anatomy-review`);
      if(result.ct_sha256!==project.ct_sha256)throw Error('Журнал относится к другой CT.');
      if(!alive.current)return;
      docRef.current=copy(result.document);versionRef.current=result.version_id;savedRef.current=JSON.stringify(result.document);
      setDocument(copy(result.document));setVersion(result);setTarget(String((result.document.target?.index??0)+1));setDirty(false);setConflict(false);setLoaded(true);
      setCapture(null);setEditing('');clearTrack();setDecision(newDecision(decisionClass,result.document));
    }catch(e){if(alive.current)setError(String(e));}
  }
  async function save():Promise<void>{
    if(jobRef.current){await jobRef.current;if(JSON.stringify(docRef.current)!==savedRef.current)return save();return;}
    if(JSON.stringify(docRef.current)===savedRef.current)return;
    if(readOnly)throw Error('Журнал доступен только для чтения.');
    if(conflict)throw Error('Сначала разрешите конфликт версий журнала.');
    const snapshot=copy(docRef.current),json=JSON.stringify(snapshot);setBusy(true);setError('');
    const job=(async()=>{try{
      const result=await post<AnatomyReviewVersion>(`/projects/${project.id}/anatomy-review`,{base_version_id:versionRef.current,ct_sha256:project.ct_sha256,document:snapshot});
      if(result.ct_sha256!==project.ct_sha256)throw Error('Сохранённая версия относится к другой CT.');
      versionRef.current=result.version_id;savedRef.current=json;
      if(alive.current){setVersion(result);setDirty(JSON.stringify(docRef.current)!==json);setConflict(false);}
    }catch(e){if(alive.current){setError(String(e));if(String(e).startsWith('Error: 409:'))setConflict(true);}throw e;
    }finally{jobRef.current=null;if(alive.current)setBusy(false);}})();
    jobRef.current=job;await job;
  }
  useImperativeHandle(ref,()=>({flush,dirty:()=>JSON.stringify(docRef.current)!==savedRef.current||!!pendingMessage()}),[readOnly,conflict]);
  useEffect(()=>{alive.current=true;void load();return()=>{alive.current=false;};},[project.id]);
  useEffect(()=>{if(!dirty||!writable||conflict)return;const timer=setTimeout(()=>void save().catch(()=>{}),2200);return()=>clearTimeout(timer);},[document,dirty,writable,conflict]);
  useEffect(()=>{const guard=(e:BeforeUnloadEvent)=>{if(JSON.stringify(docRef.current)!==savedRef.current||jobRef.current||pendingMessage()){e.preventDefault();e.returnValue='';}};window.addEventListener('beforeunload',guard);return()=>window.removeEventListener('beforeunload',guard);},[]);
  function downloadLocal(){const url=URL.createObjectURL(new Blob([JSON.stringify({project_id:project.id,ct_sha256:project.ct_sha256,base_version_id:versionRef.current,document:docRef.current},null,2)],{type:'application/json'}));const link=window.document.createElement('a');link.href=url;link.download='anatomy-review-local.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  function startCapture(){if(blockPending())return;try{if(!writable||!masksHidden)throw Error('Для наблюдения откройте только CT без масок.');setCapture(getView(view));setEditing('');observationBase.current=JSON.stringify({kind:'observation',structure:0,statement:'',features:'',confidence:'low',uncertainty:'',evidence:[]});setKind('observation');setStructure(0);setStatement('');setFeatures('');setConfidence('low');setUncertainty('');setEvidence([]);setError('');}catch(e){setError(String(e));}}
  async function editNote(note:AnatomyObservation){if(!writable||!masksHidden||blockPending())return;try{await locate(note.locator);setCapture(copy(note.locator));setEditing(note.id);observationBase.current=JSON.stringify({kind:note.kind,structure:note.structure,statement:note.statement,features:note.features.join('\n'),confidence:note.confidence,uncertainty:note.uncertainty_reason,evidence:[...note.evidence_refs]});setKind(note.kind);setStructure(note.structure);setStatement(note.statement);setFeatures(note.features.join('\n'));setConfidence(note.confidence);setUncertainty(note.uncertainty_reason);setEvidence([...note.evidence_refs]);setError('');}catch(e){setError(String(e));}}
  function addNote(){if(!capture||!masksHidden||!statement.trim()||!writable)return;
    if(lines(features).length>20||lines(features).some(f=>f.length>300)){setError('Укажите до 20 признаков, каждый до 300 символов.');return;}
    const note:AnatomyObservation={id:editing||crypto.randomUUID(),kind,structure,statement:statement.trim(),features:lines(features),confidence,uncertainty_reason:uncertainty.trim(),evidence_refs:kind==='hypothesis'?evidence:[],source:'human',locator:copy(capture)};
    update({...document,observations:editing?document.observations.map(n=>n.id===editing?note:n):[...document.observations,note]});setCapture(null);setEditing('');setStatement('');setFeatures('');setError('');
  }
  function addRelation(){if(!from||!to||from===to||!trackIds.includes(from)||!trackIds.includes(to)||(relationKind==='unresolved'&&!gap.trim()))return;
    setRelations([...relations.filter(r=>!(r.from_id===from&&r.to_id===to)),{from_id:from,to_id:to,kind:relationKind,gap_reason:gap.trim()}]);setFrom('');setTo('');setRelationKind('continuation');setGap('');setError('');
  }
  function saveTrack(){if(!writable||trackIds.length<2||!relations.length)return;const track:AnatomyTrack={id:trackEditing||crypto.randomUUID(),structure:trackClass,observation_ids:trackIds,relations};
    if(from||to||gap){setError('Добавьте или отмените незаписанную связь между наблюдениями.');return;}
    update({...document,tracks:trackEditing?document.tracks.map(t=>t.id===trackEditing?track:t):[...document.tracks,track]});clearTrack();setError('');
  }
  function clearTrack(){setTrackIds([]);setRelations([]);setTrackEditing('');setFrom('');setTo('');setRelationKind('continuation');setGap('');}
  function editTrack(track:AnatomyTrack){if(blockPending('track'))return;setTrackEditing(track.id);setTrackClass(track.structure);setTrackIds([...track.observation_ids]);setRelations(copy(track.relations));setFrom('');setTo('');setRelationKind('continuation');setGap('');setError('');}
  function changeTrackClass(id:number){if(blockPending('track'))return;setTrackClass(id);clearTrack();}
  function changeDecision(id:number){if(blockPending('decision'))return;setDecisionClass(id);setDecision(newDecision(id,document));setError('');}
  function chooseTarget(){if(blockPending())return;if(!Number.isInteger(+target)||+target<1||+target>project.geometry.shape[2])return;const next={...document,target:{axis:2 as const,index:+target-1},decisions:targetChangesDecisions?[]:document.decisions};update(next);if(targetChangesDecisions)setDecision(newDecision(decisionClass));}
  function saveDecision(){if(!writable)return;const next={...decision,label:decisionClass,unresolved_questions:lines(decision.unresolved_questions.join('\n'))};if(next.unresolved_questions.length>20||next.unresolved_questions.some(q=>q.length>1500)){setError('Укажите до 20 вопросов, каждый до 1500 символов.');return;}setDecision(next);update({...document,decisions:[...document.decisions.filter(d=>d.label!==decisionClass),next]});setError('');}
  if(!open)return null;
  return <aside className="anatomy-review" aria-label="Анатомический разбор CT">
    <div className="anatomy-heading"><div><div className="eyebrow">ЛОКАЛЬНЫЙ ЖУРНАЛ</div><h2>Анатомический разбор CT</h2></div><button className="quiet" onClick={close} aria-label="Закрыть анатомический разбор">×</button></div>
    <p className="anatomy-banner">Журнал хранится локально; источник каждой записи указан отдельно. {loaded&&!document.observations.length?'Журнал пока пуст. ':''}Наличие AI-предсказаний не является анатомическим доказательством.</p>
    {readOnly&&<p className="anatomy-banner">Журнал доступен только для чтения.</p>}
    {!masksHidden&&<p role="alert" className="anatomy-warning">Маски видимы. Переключите «Показать» на «Только CT», чтобы записать наблюдение.</p>}
    {error&&<div role="alert" className="anatomy-warning">{error}{conflict&&<><p>Локальные записи сохранены в открытом окне. Сначала выгрузите их, затем загрузите последнюю версию.</p><button disabled={busy} onClick={()=>{downloadLocal();void load();}}>Скачать локальную копию и загрузить последнюю</button></>}</div>}
    {!loaded?<div className="anatomy-empty">Загрузка журнала… <button onClick={()=>void load()}>Повторить</button></div>:<>
      <fieldset disabled={!writable} className="anatomy-target"><legend>Целевой срез исходной CT</legend><label>Аксиальный индекс (1–{project.geometry.shape[2]})<input aria-label="Целевой аксиальный срез" type="number" min={1} max={project.geometry.shape[2]} value={target} onChange={e=>setTarget(e.target.value)}/></label><button disabled={!Number.isInteger(+target)||+target<1||+target>project.geometry.shape[2]} onClick={chooseTarget}>{targetChangesDecisions?'Сменить срез и сбросить решения':'Выбрать'}</button><small>{document.target?`Выбран срез ${document.target.index+1}`:'Целевой срез ещё не выбран'}. Наблюдения можно записывать по всему объёму.</small></fieldset>
      <div className="anatomy-tabs" role="tablist" aria-label="Разделы разбора">{([['observations','Записи'],['tracks','Ход структур'],['decisions','Решения']] as const).map(([id,name])=><button key={id} role="tab" aria-selected={tab===id} className={tab===id?'active':''} onClick={()=>setTab(id)}>{name}</button>)}</div>
      <div className="anatomy-content">
        {tab==='observations'&&<>
          <div className="anatomy-capture"><label>Текущий вид<select value={view} onChange={e=>setView(+e.target.value)}>{['AX','COR','SAG'].map((v,i)=><option key={v} value={i}>{v}</option>)}</select></label><button disabled={!writable||!viewerReady||!masksHidden} onClick={startCapture}>Записать текущий вид</button></div>
          {capture&&<fieldset disabled={!writable||!masksHidden} className="anatomy-form"><legend>{editing?'Изменить запись':'Новая запись'} · {capture.view}</legend><small>Плоскость зафиксирована при нажатии кнопки. {capture.native_axis!==undefined?`Исходная ось ${capture.native_axis}, срез ${capture.native_index!+1}.`:'Физическая плоскость без привязки к исходному срезу.'}</small>
            <label>Тип записи<select value={kind} disabled={editReferenced} onChange={e=>{setKind(e.target.value as AnatomyObservation['kind']);setEvidence([]);}}><option value="observation">Наблюдаемый признак</option><option value="hypothesis">Гипотеза об анатомии</option></select></label>{editReferenced&&<small>На наблюдение уже ссылаются другие записи. Для новой гипотезы создайте отдельную запись.</small>}
            <label>Предполагаемая структура<select value={structure} onChange={e=>setStructure(+e.target.value)}><option value={0}>Неизвестная структура</option>{project.labels.map(l=><option key={l.id} value={l.id} disabled={editTrackClasses.some(id=>id!==l.id)}>{l.code} · {l.name}</option>)}</select></label>
            <label>{kind==='observation'?'Что видно на CT':'Содержание гипотезы'}<textarea maxLength={3000} value={statement} onChange={e=>setStatement(e.target.value)} placeholder={kind==='observation'?'Опишите видимый признак своими словами':'Укажите предположение отдельно от наблюдаемого признака'}/></label>
            <label>Видимые признаки · по одному на строке<textarea value={features} onChange={e=>setFeatures(e.target.value)}/></label>
            <Confidence value={confidence} onChange={setConfidence}/><label>Причина неопределённости<textarea maxLength={1500} value={uncertainty} onChange={e=>setUncertainty(e.target.value)}/></label>
            {kind==='hypothesis'&&<details open><summary>Наблюдения в поддержку гипотезы</summary>{document.observations.filter(n=>n.id!==editing&&n.kind==='observation'&&n.source==='human').map(n=><label className="anatomy-check" key={n.id}><input type="checkbox" checked={evidence.includes(n.id)} onChange={()=>setEvidence(toggle(evidence,n.id))}/>{noteName(n.id)}</label>)}</details>}
            <div className="anatomy-row"><button className="primary" disabled={!statement.trim()} onClick={addNote}>{editing?'Обновить запись':'Добавить запись'}</button><button onClick={()=>{setCapture(null);setEditing('');setError('');}}>Отмена</button></div>
          </fieldset>}
          <div className="anatomy-notes">{document.observations.map((n,index)=><article key={n.id}><button className="anatomy-note-locate" disabled={!viewerReady} onClick={async()=>{if(blockPending())return;try{await locate(n.locator);setError('');}catch(e){setError(String(e));}}}><strong>{index+1}. {n.kind==='hypothesis'?'Гипотеза':'Наблюдение'} · {n.locator.view}</strong><span>{n.statement}</span><small>{labelName(n.structure)} · {confidenceNames[n.confidence]} уверенность · {sourceNames[n.source]}</small></button>{!!n.features.length&&<p>{n.features.join(' · ')}</p>}{n.uncertainty_reason&&<p>Неопределённость: {n.uncertainty_reason}</p>}{!!n.evidence_refs.length&&<small>Связи: {n.evidence_refs.map(id=>String(document.observations.findIndex(o=>o.id===id)+1)).join(', ')}</small>}{writable&&n.source==='human'&&<button className="quiet" disabled={!masksHidden} onClick={()=>void editNote(n)}>Изменить</button>}{n.source!=='human'&&<small>Предложение AI сохранено с исходным источником. Собственное наблюдение врача добавляется отдельной записью.</small>}</article>)}{!document.observations.length&&<p className="anatomy-empty">Наблюдения и гипотезы ещё не добавлены. Выберите AX, COR или SAG и зафиксируйте текущий вид без масок.</p>}</div>
        </>}
        {tab==='tracks'&&<><p className="anatomy-explain">Свяжите наблюдения вдоль структуры. Непрерывность, ветвление и разрыв указываются явно; гипотезы остаются отдельными записями.</p><fieldset disabled={!writable} className="anatomy-form"><legend>{trackEditing?'Изменить ход структуры':'Новый ход структуры'}</legend><label>Структура<select disabled={trackReferenced} value={trackClass} onChange={e=>changeTrackClass(+e.target.value)}>{project.labels.map(l=><option key={l.id} value={l.id}>{l.code}</option>)}</select></label>
          {document.observations.filter(n=>n.kind==='observation'&&(n.structure===0||n.structure===trackClass)).map(n=><label key={n.id} className="anatomy-check"><input type="checkbox" checked={trackIds.includes(n.id)} onChange={()=>{const ids=toggle(trackIds,n.id);setTrackIds(ids);setRelations(relations.filter(r=>ids.includes(r.from_id)&&ids.includes(r.to_id)));if(!ids.includes(from))setFrom('');if(!ids.includes(to))setTo('');}}/>{noteName(n.id)}</label>)}
          {!document.observations.some(n=>n.kind==='observation')&&<small>Сначала добавьте наблюдения в нескольких положениях.</small>}
          <label>От записи<select value={from} onChange={e=>setFrom(e.target.value)}><option value="">Выберите запись</option>{trackIds.map(id=><option key={id} value={id}>{noteName(id)}</option>)}</select></label><label>К записи<select value={to} onChange={e=>setTo(e.target.value)}><option value="">Выберите запись</option>{trackIds.map(id=><option key={id} value={id}>{noteName(id)}</option>)}</select></label>
          <label>Связь<select value={relationKind} onChange={e=>setRelationKind(e.target.value as AnatomyRelation['kind'])}><option value="continuation">Продолжение</option><option value="branch">Ветвление</option><option value="unresolved">Связь не установлена</option></select></label><label>Причина разрыва или сомнения<textarea maxLength={1500} value={gap} onChange={e=>setGap(e.target.value)}/></label>
          <button disabled={!from||!to||from===to||(relationKind==='unresolved'&&!gap.trim())} onClick={addRelation}>Добавить связь</button>{relations.map((r,index)=><div className="anatomy-relation" key={index}><span>{noteName(r.from_id)} → {noteName(r.to_id)}<small>{r.kind==='continuation'?'Продолжение':r.kind==='branch'?'Ветвление':'Связь не установлена'}{r.gap_reason?` · ${r.gap_reason}`:''}</small></span><button onClick={()=>setRelations(relations.filter((_,i)=>i!==index))} aria-label="Удалить связь">×</button></div>)}
          <div className="anatomy-row"><button className="primary" disabled={trackIds.length<2||!relations.length} onClick={saveTrack}>Записать ход структуры</button>{(trackEditing||pendingTrack)&&<button onClick={()=>{clearTrack();setError('');}}>Отменить ход структуры</button>}</div></fieldset>
          {document.tracks.map((t,index)=><article className="anatomy-track" key={t.id}><strong>{index+1}. {labelName(t.structure)} · {t.observation_ids.length} наблюдений</strong>{t.relations.map((r,j)=><p key={j}>{noteName(r.from_id)} → {noteName(r.to_id)}<small>{r.kind==='continuation'?'Продолжение':r.kind==='branch'?'Ветвление':'Связь не установлена'}{r.gap_reason?` · ${r.gap_reason}`:''}</small></p>)}<button disabled={!writable} onClick={()=>editTrack(t)}>Изменить</button></article>)}{!document.tracks.length&&<p className="anatomy-empty">Ход структур ещё не описан.</p>}</>}
        {tab==='decisions'&&<><p className="anatomy-explain">Решение относится к выбранному целевому срезу. Отметьте свидетельства, противоречия, правило границы и оставшиеся вопросы.</p><div className="anatomy-decision-list">{project.labels.map(l=><button key={l.id} className={decisionClass===l.id?'active':''} onClick={()=>changeDecision(l.id)}>{l.code}<small>{decisionNames[document.decisions.find(d=>d.label===l.id)?.state||'unassessed']}</small></button>)}</div>
          <fieldset disabled={!writable} className="anatomy-form"><legend>Решение · {labelName(decisionClass)}</legend><label>Оценка<select value={decision.state} onChange={e=>setDecision({...decision,state:e.target.value as AnatomyDecision['state']})}>{Object.entries(decisionNames).map(([id,name])=><option key={id} value={id}>{name}</option>)}</select></label><Confidence value={decision.confidence} onChange={value=>setDecision({...decision,confidence:value})}/>
            <details open><summary>Поддерживающие наблюдения</summary>{document.observations.filter(n=>n.kind==='observation').map(n=><label className="anatomy-check" key={n.id}><input type="checkbox" checked={decision.supporting_observation_ids.includes(n.id)} onChange={()=>setDecision({...decision,supporting_observation_ids:toggle(decision.supporting_observation_ids,n.id)})}/>{noteName(n.id)}</label>)}</details>
            <details><summary>Противоречащие наблюдения</summary>{document.observations.filter(n=>n.kind==='observation').map(n=><label className="anatomy-check" key={n.id}><input type="checkbox" checked={decision.counterevidence_ids.includes(n.id)} onChange={()=>setDecision({...decision,counterevidence_ids:toggle(decision.counterevidence_ids,n.id)})}/>{noteName(n.id)}</label>)}</details>
            <details><summary>Связанный ход структуры</summary>{document.tracks.filter(t=>t.structure===decisionClass).map((t,index)=><label className="anatomy-check" key={t.id}><input type="checkbox" checked={decision.track_ids.includes(t.id)} onChange={()=>setDecision({...decision,track_ids:toggle(decision.track_ids,t.id)})}/>{index+1}. {labelName(t.structure)} · {t.observation_ids.length} наблюдений</label>)}</details>
            <label>Правило проведения границы<textarea maxLength={3000} value={decision.boundary_rule} onChange={e=>setDecision({...decision,boundary_rule:e.target.value})}/></label><label>Обоснование решения<textarea maxLength={3000} value={decision.rationale} onChange={e=>setDecision({...decision,rationale:e.target.value})}/></label><label>Нерешённые вопросы · по одному на строке<textarea value={decision.unresolved_questions.join('\n')} onChange={e=>setDecision({...decision,unresolved_questions:e.target.value.split('\n')})}/></label>
            <div className="anatomy-row"><button className="primary" onClick={saveDecision}>Записать решение</button>{pendingDecision&&<button onClick={()=>{setDecision(newDecision(decisionClass,document));setError('');}}>Отменить решение</button>}</div>
          </fieldset></>}
      </div>
      <div className="anatomy-save"><div role="status">{hasPendingForm?'Есть незаписанная форма · добавьте или отмените её':busy?'Сохранение журнала…':dirty?'Есть несохранённые записи':version?.version_id?`Журнал сохранён · версия ${version.number}`:'Сохранённых версий журнала нет'}</div><div className="anatomy-row"><button className="primary" disabled={!writable||busy||(!dirty&&!hasPendingForm)||conflict} onClick={()=>void flush().catch(()=>{})}>Сохранить журнал</button><button onClick={downloadLocal}>Локальная копия JSON</button><a className={!version?.version_id||dirty?'disabled':''} aria-disabled={!version?.version_id||dirty} href={version?.version_id&&!dirty?`/api/projects/${project.id}/anatomy-review/export`:undefined} download>Экспорт версии JSON</a></div>
        <details><summary>{dirty||hasPendingForm?'Полнота будет проверена после сохранения':version?.readiness.ready?'Журнал заполнен':'Журнал пока не заполнен'}</summary><p>Это проверка полноты записей, а не подтверждение правильности анатомии.</p>{!dirty&&!hasPendingForm&&version?.readiness.issues.map((issue,i)=><p key={i}>{issue}</p>)}</details>
      </div>
    </>}
  </aside>;
});
function newDecision(label:number,document?:AnatomyDocument):AnatomyDecision{return copy(document?.decisions.find(d=>d.label===label)||{label,state:'unassessed',supporting_observation_ids:[],counterevidence_ids:[],track_ids:[],boundary_rule:'',rationale:'',unresolved_questions:[],confidence:'low'});}
