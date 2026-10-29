// Clinical pixels stay inside local Chrome. Only numerical render checks leave it.
import {chromium,expect} from '@playwright/test';
import fs from 'node:fs/promises';
import {installLocalNetworkGuard} from './local-network-guard.mjs';
import {readMPRStats} from './render-stats.mjs';
const [origin,alias,report]=process.argv.slice(2);
const anatomy=process.argv.includes('--anatomy');
if(!origin||!alias||!report)throw Error('origin alias private-report required');
if(process.stdin.isTTY)process.stdin.setRawMode(true);
console.log('READY_FOR_PASSWORD_STDIN (hidden)');
let password=await new Promise(resolve=>{let value='';const read=chunk=>{value+=chunk.toString();if(/[\r\n]/.test(value)){
  process.stdin.off('data',read);process.stdin.pause();if(process.stdin.isTTY)process.stdin.setRawMode(false);resolve(value.split(/[\r\n]/)[0]);
}};process.stdin.on('data',read);process.stdin.resume();});
const browser=await chromium.launch({channel:'chrome',headless:true,args:['--force_high_performance_gpu','--use-angle=d3d11','--disable-extensions','--disable-background-networking']});
try{
  const context=await browser.newContext({serviceWorkers:'block',acceptDownloads:false,viewport:{width:1500,height:1000}});
  const blocked={};let completedExternal=0;const errors=[];
  await installLocalNetworkGuard(context,origin,host=>{blocked[host]=(blocked[host]||0)+1;});
  const page=await context.newPage();page.setDefaultTimeout(90000);page.on('pageerror',e=>errors.push(e.message));
  page.on('requestfinished',r=>{if(new URL(r.url()).origin!==origin)completedExternal++;});
  await page.goto(origin,{waitUntil:'domcontentloaded'});await page.getByLabel('Логин',{exact:true}).fill('resident');await page.getByLabel('Пароль',{exact:true}).fill(password);password='';
  await page.getByRole('button',{name:'Войти в рабочее пространство'}).click();await page.locator('article').filter({hasText:alias}).getByRole('button',{name:'Открыть разметку'}).click();
  await expect(page.getByText('● Геометрия проверена')).toBeVisible({timeout:90000});
  const projects=await (await page.request.get(origin+'/api/projects')).json(),project=projects.find(p=>p.study_id===alias);
  const proposal=project.predictions.find(v=>v.source===(anatomy?'AI anatomy slice draft':'AI-assisted slice draft'));expect(proposal).toBeTruthy();
  const target=proposal.provenance.scope.index;
  await expect(page.getByLabel('Показать маску')).toHaveValue(proposal.id);
  await expect(page.getByLabel('Срез Аксиальная',{exact:true})).toHaveValue(String(target));
  await expect(page.locator('.pane-0 .pane-label small')).toHaveText(`${target+1} / ${project.geometry.shape[2]}`);
  await expect(page.locator('.slice-scope')).toContainText('не экспертная GT');
  await expect(page.getByRole('button',{name:'3D-реконструкция'})).toBeDisabled();
  await expect(page.getByRole('button',{name:'Взять черновик за основу'})).toBeEnabled();
  await expect(page.getByRole('button',{name:'Кисть',exact:true})).toBeDisabled();
  await expect.poll(async()=>{const s=await readMPRStats(page);return s.every(p=>p.gray>500&&p.grayLevels>=16)&&s[0].colored>100;},{timeout:20000}).toBe(true);
  const initial=await readMPRStats(page);
  const physical=await page.evaluate(async()=>{const c=await import('/node_modules/.vite/deps/@cornerstonejs_core.js'),v=c.getRenderingEngines()[0].getViewports()[0],volume=c.cache.getVolume(v.getVolumeId());return {source_z:volume.imageData.worldToIndex(v.getCamera().focalPoint)[2],renderer:c.getRenderingCapabilities().renderer};});
  expect(Math.abs(physical.source_z-target)).toBeLessThan(.01);
  await page.getByLabel('Показать маску').selectOption('ct');
  await expect.poll(async()=> (await readMPRStats(page)).every(p=>p.gray>500&&p.grayLevels>=16&&p.colored===0)).toBe(true);
  await page.getByLabel('Показать маску').selectOption(proposal.id);
  await expect.poll(async()=> (await readMPRStats(page))[0].colored).toBeGreaterThan(100);
  await page.getByLabel('Прозрачность маски',{exact:true}).focus();await page.keyboard.press('ArrowRight');
  await page.getByLabel('Видимость RV',{exact:true}).click();await page.getByLabel('Видимость RV',{exact:true}).click();
  const after=await (await page.request.get(origin+`/api/projects/${project.id}`)).json();expect(after.version_id).toBe(project.version_id);expect(after.status).toBe(project.status);
  if(anatomy){
    const notebook=await (await page.request.get(origin+`/api/projects/${project.id}/anatomy-review`)).json();
    expect(notebook.version_id).toBe(proposal.provenance.notebook_version_id);
    expect(notebook.document.target.index).toBe(target);
    expect(notebook.document.observations.some(n=>n.source==='cloud_model')).toBe(true);
    expect(proposal.provenance.observer_sources).toContain('cloud_model');
    expect(proposal.provenance.medical_correctness_verified).toBe(false);
  }
  expect(errors).toEqual([]);expect(completedExternal).toBe(0);
  const result={passed:true,study:alias,source:proposal.source,notebook_link_verified:anatomy,source_index:target,display_number:target+1,rendered_MPR_pixels:initial,physical_plane:physical,CT_only_verified:true,opacity_and_visibility_verified:true,AI_mask_readonly:true,adoption_button_available:true,partial_3D_disabled:true,real_study_edits:false,JS_errors:errors,finished_external_requests:completedExternal,blocked_external_attempts:blocked,screenshots:false,traces:false};
  await fs.writeFile(report,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
  await page.getByRole('button',{name:'Выйти',exact:true}).click();
}finally{await browser.close();}
