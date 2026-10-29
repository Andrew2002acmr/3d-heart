// Read-only clinical UI check. Pixels and notebook text never leave local Chrome.
import {chromium} from '@playwright/test';
import fs from 'node:fs/promises';
import {installLocalNetworkGuard} from './local-network-guard.mjs';
import {readMPRStats} from './render-stats.mjs';

const [origin,alias,report]=process.argv.slice(2);
if(!origin||!/^C-[0-9]{3}$/.test(alias||'')||!report)throw Error('Exact local origin, pseudonym and private report path required');
const requireTrue=(value,message)=>{if(!value)throw Error(message);};
const until=async(check,message)=>{
  const deadline=Date.now()+45000;
  while(Date.now()<deadline){if(await check())return;await new Promise(resolve=>setTimeout(resolve,200));}
  throw Error(message);
};
if(process.stdin.isTTY)process.stdin.setRawMode(true);
console.log('READY_FOR_PASSWORD_STDIN (hidden)');
let password=await new Promise(resolve=>{let value='';const read=chunk=>{value+=chunk.toString();if(/[\r\n]/.test(value)){
  process.stdin.off('data',read);process.stdin.pause();if(process.stdin.isTTY)process.stdin.setRawMode(false);resolve(value.split(/[\r\n]/)[0]);
}};process.stdin.on('data',read);process.stdin.resume();});
const browser=await chromium.launch({channel:'chrome',headless:true,args:['--force_high_performance_gpu','--use-angle=d3d11','--disable-extensions','--disable-background-networking']});
try{
  const context=await browser.newContext({serviceWorkers:'block',acceptDownloads:false,viewport:{width:1600,height:1050}});
  let blockedExternal=0,finishedExternal=0,pageErrors=0,maskRequests=0,clinicalWrites=0;
  await installLocalNetworkGuard(context,origin,()=>{blockedExternal++;});
  const page=await context.newPage();page.setDefaultTimeout(90000);
  page.on('pageerror',()=>{pageErrors++;});
  page.on('requestfinished',request=>{if(new URL(request.url()).origin!==origin)finishedExternal++;});
  page.on('request',request=>{
    const url=new URL(request.url());
    if(url.origin!==origin)return;
    if(/\/api\/projects\/[^/]+\/(mask|preview)(\/|$)/.test(url.pathname))maskRequests++;
    if(url.pathname.startsWith('/api/projects/')&&!['GET','HEAD','OPTIONS'].includes(request.method()))clinicalWrites++;
  });
  await page.goto(origin+'/?view=anatomy',{waitUntil:'domcontentloaded'});
  await page.getByLabel('Логин',{exact:true}).fill('resident');
  await page.getByLabel('Пароль',{exact:true}).fill(password);password='';
  await page.getByRole('button',{name:'Войти в рабочее пространство'}).click();
  await page.locator('article').filter({hasText:alias}).getByRole('button',{name:'Открыть разметку'}).click();
  await page.getByText('● Геометрия проверена').waitFor({timeout:180000});
  const panel=page.getByRole('complementary',{name:'Анатомический разбор CT'});
  await panel.waitFor({state:'visible'});
  const listResponse=await page.request.get(origin+'/api/projects');requireTrue(listResponse.ok(),'Project list not available');
  const project=(await listResponse.json()).find(p=>p.study_id===alias);requireTrue(!!project,'Pseudonymous study not found');
  const endpoint=origin+`/api/projects/${project.id}`;
  const before=await (await page.request.get(endpoint)).json();
  const notebook=await (await page.request.get(endpoint+'/anatomy-review')).json();
  await until(async()=> (await readMPRStats(page)).every(s=>s.gray>500&&s.grayLevels>=16&&s.colored===0),'Three CT-only MPR views did not render');
  const renderStats=await readMPRStats(page);
  requireTrue(new URL(page.url()).searchParams.get('view')==='anatomy','CT-only mode missing from URL');
  const capturedViews=[];
  for(const [index,name] of [['0','AX'],['1','COR'],['2','SAG']]){
    await panel.getByRole('combobox',{name:'Текущий вид',exact:true}).selectOption(index);
    await panel.getByRole('button',{name:'Записать текущий вид',exact:true}).click();
    requireTrue(await panel.getByRole('textbox',{name:'Что видно на CT',exact:true}).isVisible(),'Physical view capture unavailable');
    requireTrue(await panel.getByRole('textbox',{name:'Что видно на CT',exact:true}).inputValue()==='','Unexpected observation content');
    await panel.getByRole('button',{name:'Отмена',exact:true}).click();capturedViews.push(name);
  }
  await panel.getByRole('tab',{name:'Ход структур',exact:true}).click();
  const tracksAvailable=await panel.getByRole('button',{name:'Записать ход структуры',exact:true}).isVisible();
  await panel.getByRole('tab',{name:'Решения',exact:true}).click();
  const decisionsAvailable=await panel.getByRole('combobox',{name:'Оценка',exact:true}).isVisible();
  await panel.getByRole('tab',{name:'Записи',exact:true}).click();
  const after=await (await page.request.get(endpoint)).json();
  const notebookAfter=await (await page.request.get(endpoint+'/anatomy-review')).json();
  requireTrue(after.version_id===before.version_id&&after.status===before.status&&after.revision===before.revision,'Segmentation state changed during read-only check');
  requireTrue(notebookAfter.version_id===notebook.version_id,'Notebook changed during read-only check');
  requireTrue(pageErrors===0&&finishedExternal===0&&clinicalWrites===0&&maskRequests===0,'Read-only/local/CT-only guard failed');
  const result={passed:true,study:alias,CT_only:true,rendered_MPR:renderStats,captured_empty_views:capturedViews,
    tracks_controls:tracksAvailable,decision_controls:decisionsAvailable,patient_anatomy_review_performed:false,
    observations_created:0,segmentation_edits:0,notebook_writes:0,mask_preview_requests:maskRequests,
    existing_segmentation_unchanged:true,existing_notebook_unchanged:true,page_errors:pageErrors,
    finished_external_requests:finishedExternal,blocked_external_attempts:blockedExternal,screenshots:false,traces:false};
  await fs.writeFile(report,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
  await page.getByRole('button',{name:'Выйти',exact:true}).click();
}catch{
  // Avoid Playwright snapshots, page text and response bodies in clinical logs.
  console.error('Local anatomy viewer verification failed; no image, DOM, notebook text or response body emitted.');
  process.exitCode=1;
}finally{password='';await browser.close();}
