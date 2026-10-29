// Read-only local clinical smoke check. No screenshots, videos, traces, exports or edits.
// Password arrives only on stdin and is never logged or persisted.
import {chromium,expect} from '@playwright/test';
import fs from 'node:fs/promises';
import {installLocalNetworkGuard} from './local-network-guard.mjs';
import {readMPRStats} from './render-stats.mjs';

const [origin,alias,report] = process.argv.slice(2);
if (!origin || !/^[A-Z0-9-]{3,40}$/.test(alias||'') || !report) throw Error('origin, pseudonymous alias, private report path required');
if (process.stdin.isTTY) process.stdin.setRawMode(true);
console.log('READY_FOR_PASSWORD_STDIN (hidden)');
let password = await new Promise(resolve => {
  let value='';const read=chunk=>{value+=chunk.toString();if(/[\r\n]/.test(value)){
    process.stdin.off('data',read);process.stdin.pause();if(process.stdin.isTTY)process.stdin.setRawMode(false);resolve(value.split(/[\r\n]/)[0]);
  }};process.stdin.on('data',read);process.stdin.resume();
});
const browser=await chromium.launch({channel:'chrome',headless:process.env.HEART3D_RENDER_HEADFUL!=='1',args:['--force_high_performance_gpu','--use-angle=d3d11','--disable-extensions','--disable-background-networking']});
try {
  const context=await browser.newContext({serviceWorkers:'block',acceptDownloads:false,viewport:{width:1500,height:1000}});
  const blocked={},errors=[];let nonlocalFinished=0;
  await installLocalNetworkGuard(context,origin,host=>{blocked[host]=(blocked[host]||0)+1;});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  if(process.env.HEART3D_RENDER_DIAGNOSTICS==='1')page.on('console',msg=>{if(['error','warning'].includes(msg.type()))console.log('RENDER_LOG',msg.text().slice(0,500));});
  page.on('requestfinished',r=>{if(new URL(r.url()).origin!==origin)nonlocalFinished++;});
  page.setDefaultTimeout(180000);
  await page.goto(origin,{waitUntil:'domcontentloaded'});console.log('Local page loaded');await page.getByLabel('Логин',{exact:true}).fill('resident');await page.getByLabel('Пароль',{exact:true}).fill(password);password='';
  await page.getByRole('button',{name:'Войти в рабочее пространство'}).click();
  await page.locator('article').filter({hasText:alias}).getByRole('button',{name:'Открыть разметку'}).click();
  await expect(page.getByText('● Геометрия проверена')).toBeVisible({timeout:180000});
  console.log('C-001 original CT and initial AI mask loaded');
  await page.waitForTimeout(1500);
  console.log('MPR_RENDER_STATS',JSON.stringify(await readMPRStats(page)));
  const initialRender=await readMPRStats(page);
  expect(initialRender.every(p=>p.gray>500&&p.grayLevels>=16&&p.colored>100)).toBe(true);
  if(process.env.HEART3D_RENDER_DIAGNOSTICS==='1'){
    await page.getByLabel('Показать маску').selectOption('ct');await page.waitForTimeout(1200);
    console.log('CT_ONLY_RENDER_STATS',JSON.stringify(await readMPRStats(page)));
    console.log('RENDER_DIAGNOSTICS',JSON.stringify(await page.evaluate(async()=>{
      const core=await import('/node_modules/.vite/deps/@cornerstonejs_core.js');
      return {capabilities:core.getRenderingCapabilities(),views:core.getRenderingEngines().map(e=>e.getViewports().map(v=>({camera:v.getCamera(),bounds:v.getDefaultActor()?.actor.getBounds(),properties:v.getProperties(),actors:v.getActors().length})))};
    })));
    await browser.close();process.exit(0);
  }
  const project=await (await page.request.get(origin+'/api/projects')).json();const p=project.find(p=>p.study_id===alias);
  if(!p||p.synthetic||p.version_id||p.status!=='not_started'||p.predictions.length!==2)throw Error('Expected untouched unapproved clinical AI pair');
  const models=[];
  for(const prediction of p.predictions){
    if(await page.getByLabel('Показать маску').inputValue()!==prediction.id){
      const response=page.waitForResponse(r=>r.url().includes('/mask?version_id='+prediction.id));
      await page.getByLabel('Показать маску').selectOption(prediction.id);await (await response).finished();
    }
    await expect(page.getByRole('button',{name:'Взять '+prediction.provenance.variant+' за основу'})).toBeVisible();
    await expect(page.getByRole('button',{name:'Кисть',exact:true})).toBeDisabled();
    await expect.poll(async()=> (await readMPRStats(page)).every(p=>p.gray>500&&p.grayLevels>=16&&p.colored>100)).toBe(true);
    for(const axis of ['Аксиальная','Корональная','Сагиттальная']){await page.getByLabel('Срез '+axis,{exact:true}).focus();await page.keyboard.press('ArrowRight');}
    await page.getByLabel('Прозрачность маски',{exact:true}).focus();await page.keyboard.press('ArrowRight');
    await page.getByLabel('Видимость PA',{exact:true}).click();await page.getByLabel('Видимость PA',{exact:true}).click();
    await page.getByRole('button',{name:'3D-реконструкция'}).click();
    console.log(prediction.provenance.variant+' local reconstruction requested');
    await expect(page.locator('.surface-main aside')).toContainText('LV ·',{timeout:180000});
    const response=await page.request.get(origin+`/api/projects/${p.id}/reconstruction/${prediction.id}`);const mesh=await response.json();
    expect(mesh.structures.length).toBe(7);expect(mesh.absent_labels).toEqual([]);
    await expect(page.getByTestId('surface-canvas').locator('canvas')).toBeVisible();
    const box=await page.getByTestId('surface-canvas').boundingBox();await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();await page.mouse.move(box.x+box.width/2+20,box.y+box.height/2+15,{steps:5});await page.mouse.up();
    models.push({variant:prediction.provenance.variant,structures:mesh.structures.map(s=>({class:s.short_name,triangles:s.triangles,components:s.components_26_connected,touches_scan_boundary:s.touches_scan_boundary}))});
    await page.getByRole('button',{name:'Закрыть 3D'}).click();
    expect((await readMPRStats(page)).every(p=>p.gray>500&&p.grayLevels>=16)).toBe(true);
    console.log(prediction.provenance.variant+' MPR and 3D complete');
  }
  const after=await (await page.request.get(origin+`/api/projects/${p.id}`)).json();expect(after.version_id).toBeNull();expect(after.status).toBe('not_started');
  expect(errors).toEqual([]);expect(nonlocalFinished).toBe(0);
  await page.getByLabel('Показать маску').selectOption('ct');
  await expect.poll(async()=> (await readMPRStats(page)).every(p=>p.gray>500&&p.grayLevels>=16&&p.colored===0)).toBe(true);
  const result={study:alias,source:'actual deidentified CT',read_only:true,screenshots:false,traces:false,edits:false,original_grid_shape:p.geometry.shape,MPR:true,rendered_MPR_pixels_verified:true,initial_MPR_pixel_statistics:initialRender,CT_only_pixels_verified:true,opacity:true,class_visibility:true,AI_comparison:true,interactive_3D:true,models,JS_errors:errors,nonlocal_requests_finished:nonlocalFinished,blocked_nonlocal_attempts:blocked};
  await fs.writeFile(report,JSON.stringify(result,null,2)+'\n');
  console.log(JSON.stringify({passed:true,study:alias,models:models.map(x=>x.variant),nonlocal_requests_finished:nonlocalFinished,AI_versions_unchanged:true}));
  await page.getByRole('button',{name:'Выйти',exact:true}).click();
} finally {await browser.close();}
