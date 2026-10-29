// Ephemeral local Chrome: no extensions, service workers, remote requests or traces.
import {chromium} from '@playwright/test';
import {installLocalNetworkGuard} from './local-network-guard.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {randomUUID} from 'node:crypto';

const origin = process.argv[2] || 'http://127.0.0.1:5173';
const alias = process.argv[3];
const anatomyMode=process.argv.includes('--anatomy');
const editBoundaries=process.argv.includes('--edit-boundaries');
let password='';
if(process.argv.includes('--password-stdin')){
  if(!/^[A-Z0-9-]{3,40}$/.test(alias||''))throw Error('Pseudonymous study alias required');
  if(process.stdin.isTTY)process.stdin.setRawMode(true);
  console.log('READY_FOR_PASSWORD_STDIN (hidden)');
  password=await new Promise(resolve=>{let value='';const read=chunk=>{value+=chunk.toString();if(/[\r\n]/.test(value)){
    process.stdin.off('data',read);process.stdin.pause();if(process.stdin.isTTY)process.stdin.setRawMode(false);resolve(value.split(/[\r\n]/)[0]);
  }};process.stdin.on('data',read);process.stdin.resume();});
}
const root=process.env.HEART3D_WEB_DATA||path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..','.web-data');
const browser = await chromium.launch({channel:'chrome',headless:false,
  args:['--force_high_performance_gpu','--use-angle=d3d11','--start-maximized','--disable-extensions','--disable-background-networking','--no-first-run']});
try {
  const context = await browser.newContext({serviceWorkers:'block',acceptDownloads:true,viewport:null});
  let blocked = 0;
  await installLocalNetworkGuard(context, origin, () => { blocked++; });
  context.on('page',page=>page.on('download',async download=>{
    const url=new URL(download.url());
    const mask=url.origin===origin&&/^\/api\/projects\/[a-z0-9-]+\/export\/[a-z0-9-]+$/.test(url.pathname);
    const notebook=url.origin===origin&&/^\/api\/projects\/[a-z0-9-]+\/anatomy-review\/export$/.test(url.pathname);
    const recovery=url.protocol==='blob:'&&url.origin===origin&&download.suggestedFilename()==='anatomy-review-local.json';
    if(!mask&&!notebook&&!recovery)return download.cancel();
    try{
      const directory=path.join(root,'exports');await fs.mkdir(directory,{recursive:true});
      await download.saveAs(path.join(directory,(mask?'mask-':'anatomy-review-')+randomUUID()+(mask?'.nii.gz':'.json')));
      console.log('Export saved in the private web data exports directory.');
    }catch{console.log('Private export failed; the saved notebook/version remains available in the application.');}
  }));
  const page = await context.newPage();
  console.log('Local research viewer: only '+origin+' is permitted. Close Chrome to stop.');
  const closed = new Promise(resolve => browser.once('disconnected', resolve));
  await page.goto(origin+(anatomyMode?'/?view=anatomy':''),{waitUntil:'domcontentloaded'});
  if(password){
    await page.getByLabel('Логин',{exact:true}).fill('resident');await page.getByLabel('Пароль',{exact:true}).fill(password);password='';
    await page.getByRole('button',{name:'Войти в рабочее пространство'}).click();
    await page.locator('article').filter({hasText:alias}).getByRole('button',{name:'Открыть разметку'}).click();
    await page.getByText('● Геометрия проверена').waitFor({timeout:180000});
    if(editBoundaries){
      const button=page.getByRole('button',{name:'Редактировать границы',exact:true});
      // Initialize an untouched project or add handles to an existing draft
      // without contours. Existing mask edits are preserved by the endpoint;
      // drafts with editable contours are never adopted/replaced automatically.
      const projects=await (await page.request.get(origin+'/api/projects')).json();
      const project=projects.find(p=>p.study_id===alias);
      if(project&&(!project.version_id||!project.document.contours.length)&&await button.isVisible()){
        await button.click();await page.getByText('● Геометрия проверена').waitFor({timeout:180000});
        await page.waitForFunction(()=>Number(document.querySelector('[data-testid="contour-count"]')?.textContent)>0,{},{timeout:90000});
        console.log('Editable boundary copy prepared; original AI version unchanged.');
      }else console.log('Existing draft preserved; no automatic adoption performed.');
    }
    await page.evaluate(({alias,anatomyMode})=>{document.title='Heart3D · '+alias+' · '+(anatomyMode?'анатомический разбор':'NVIDIA');},{alias,anatomyMode});
    await page.bringToFront();
    console.log(alias+(anatomyMode?' is ready: CT-only anatomy notebook; no findings generated.':' is ready: CT + AI predictions; no expert approval.'));
  }
  await closed;
  console.log('Blocked nonlocal request attempts: '+blocked+'; no request URLs or patient data logged.');
} finally {
  await browser.close();
}
