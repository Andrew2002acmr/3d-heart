// Explicit local save of an already-authored notebook; never interprets CT.
// Credentials use hidden stdin; no notebook text or response bodies are logged.
import {request} from '@playwright/test';
import fs from 'node:fs/promises';
import {isDeepStrictEqual} from 'node:util';

const [origin,alias,documentPath,exportPath]=process.argv.slice(2);
const exportExisting=process.argv.includes('--export-existing');
const base=new URL(origin);
if(base.protocol!=='http:'||base.hostname!=='127.0.0.1'||base.origin!==origin||!/^C-[0-9]{3}$/.test(alias||'')||!documentPath||!exportPath)throw Error('Exact loopback origin, pseudonym, private notebook and new export path required');
const document=JSON.parse(await fs.readFile(documentPath,'utf8'));
await fs.access(exportPath).then(()=>{throw Error('Choose a new private export path');},error=>{if(error.code!=='ENOENT')throw error;});
if(process.stdin.isTTY)process.stdin.setRawMode(true);
console.log('READY_FOR_PASSWORD_STDIN (hidden)');
let password=await new Promise(resolve=>{let value='';const read=chunk=>{value+=chunk.toString();if(/[\r\n]/.test(value)){
  process.stdin.off('data',read);process.stdin.pause();if(process.stdin.isTTY)process.stdin.setRawMode(false);resolve(value.split(/[\r\n]/)[0]);
}};process.stdin.on('data',read);process.stdin.resume();});
const client=await request.newContext({baseURL:origin});
let csrf='';
const checked=async response=>{if(!response.ok())throw Error('Local API operation rejected (status '+response.status()+'); body withheld');return response;};
try{
  const auth=await (await checked(await client.post('/api/auth/login',{data:{username:'resident',password},maxRedirects:0}))).json();password='';csrf=auth.csrf;
  const projects=await (await checked(await client.get('/api/projects',{maxRedirects:0}))).json();
  const project=projects.find(p=>p.study_id===alias);if(!project)throw Error('Assigned study not found');
  const endpoint='/api/projects/'+project.id+'/anatomy-review';
  const existing=await (await checked(await client.get(endpoint,{maxRedirects:0}))).json();
  // Never silently replace a human/model notebook already present.
  if(exportExisting&&(!existing.version_id||!isDeepStrictEqual(existing.document,document)))throw Error('Existing notebook differs; refused to export another document');
  if(!exportExisting&&(existing.version_id||existing.document.observations.length||existing.document.tracks.length||existing.document.decisions.length))throw Error('Notebook already exists; merge explicitly instead of overwriting');
  const saved=exportExisting?existing:await (await checked(await client.post(endpoint,{maxRedirects:0,headers:{'X-CSRF-Token':csrf},data:{base_version_id:existing.version_id,ct_sha256:project.ct_sha256,document}}))).json();
  // Preserve server bytes: JS reserialization changes 1000.0 to 1000 and would
  // invalidate the server's Python-canonical export digest.
  const exportText=await (await checked(await client.get(endpoint+'/export?version_id='+encodeURIComponent(saved.version_id),{maxRedirects:0}))).text();
  const exported=JSON.parse(exportText);
  if(exported.version_id!==saved.version_id||exported.ct_sha256!==project.ct_sha256)throw Error('Export identity mismatch');
  await fs.writeFile(exportPath,exportText,{flag:'wx'});
  console.log(JSON.stringify({saved:!exportExisting,exported:true,study:alias,observations:saved.document.observations.length,tracks:saved.document.tracks.length,decisions:saved.document.decisions.length,documentation_complete:saved.readiness.ready,expert_validated:false,cloud_model_provenance_preserved:true}));
}catch(error){
  console.error(error.message.startsWith('Local API')||error.message.startsWith('Notebook already')?error.message:'Notebook save failed; clinical text, DOM and response bodies withheld.');process.exitCode=1;
}finally{
  password='';if(csrf)await client.post('/api/auth/logout',{maxRedirects:0,headers:{'X-CSRF-Token':csrf}}).catch(()=>{});await client.dispose();
}
