import {useState} from 'react';
import {api,post,type Project} from './api';

export function ReviewControls({project,ready,updated}:{project:Project,ready:boolean,updated:(project:Project)=>void}){
  const [note,setNote]=useState(''),[busy,setBusy]=useState(false),[error,setError]=useState('');
  async function decide(decision:'approved'|'needs_revision'){
    setBusy(true);setError('');
    try{await post(`/projects/${project.id}/reviews`,{version_id:project.version_id,decision,note});updated(await api<Project>(`/projects/${project.id}`));}
    catch(e){setError(String(e));}finally{setBusy(false);}
  }
  return <section className="review-bar"><strong>Проверка хирургом · человеческая версия {project.version_number}</strong>
    <label>Комментарий к разметке<input aria-label="Комментарий хирурга" value={note} maxLength={2000} onChange={e=>setNote(e.target.value)} placeholder="Без персональных данных"/></label>
    <button disabled={!ready||busy||project.status!=='submitted'} onClick={()=>void decide('needs_revision')}>Вернуть на доработку</button>
    <button disabled={!ready||busy||project.status!=='submitted'} onClick={()=>void decide('approved')}>Утвердить экспертную версию</button>
    {error&&<span role="alert">{error}</span>}
  </section>;
}
