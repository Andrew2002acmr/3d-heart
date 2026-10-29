import {test,expect} from '@playwright/test';
import {installLocalNetworkGuard} from '../scripts/local-network-guard.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import {randomUUID} from 'node:crypto';
import {readMPRStats} from '../scripts/render-stats.mjs';

test('compare immutable AI versions, adopt, edit native boundary, flag, persist and render interactive 3D',async({page,context})=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  const external:string[]=[];await installLocalNetworkGuard(context,'http://127.0.0.1:5174',(host:string)=>external.push(host));
  // This controlled probe is aborted before DNS/network access.
  const probe=await context.newPage();await expect(probe.goto('https://heart3d-egress-check.invalid/')).rejects.toThrow();await probe.close();
  expect(external).toEqual(['heart3d-egress-check.invalid']);external.length=0;
  // A dev server must not bypass the authenticated API through its file serving.
  const probeFile=path.resolve('.web-data',randomUUID()+'.txt');await fs.mkdir(path.dirname(probeFile),{recursive:true});
  try{await fs.writeFile(probeFile,'synthetic private static denial probe');
    for(const url of ['/.web-data/'+path.basename(probeFile),'/@fs/'+probeFile.replaceAll('\\','/')]){
      const response=await page.request.get(url);expect(response.status()).toBe(403);expect(await response.text()).not.toContain('synthetic private static denial probe');
    }
  }finally{await fs.unlink(probeFile);}
  await page.goto('/');await page.getByLabel('Логин',{exact:true}).fill('ai-resident');await page.getByLabel('Пароль',{exact:true}).fill(process.env.HEART3D_E2E_PASSWORD!);await page.getByRole('button',{name:'Войти в рабочее пространство'}).click();
  await page.getByRole('button',{name:'Открыть разметку'}).click();await expect(page.getByText('● Геометрия проверена')).toBeVisible({timeout:45000});
  await expect.poll(async()=> (await readMPRStats(page)).every(p=>p.gray>100&&p.grayLevels>=16)).toBe(true);
  let view=(await (await page.request.get('/api/projects')).json())[0];const id=view.id;
  const v2=view.predictions.find((p:any)=>p.provenance.variant==='v2').id;
  const original=await (await page.request.get(`/api/projects/${id}/export/${v2}`)).body();
  await expect(page.getByRole('button',{name:'Кисть',exact:true})).toBeDisabled();
  await page.getByLabel('Показать маску').selectOption(v2);await expect(page.getByRole('button',{name:'Взять v2 за основу'})).toBeVisible();
  await page.getByLabel('Прозрачность маски').focus();await page.keyboard.press('ArrowRight');
  await page.getByRole('button',{name:'Взять v2 за основу'}).click();await expect(page.getByRole('button',{name:'Контур маски на срезе'})).toBeEnabled({timeout:45000});
  view=await (await page.request.get('/api/projects/'+id)).json();expect(view.document.base_prediction_id).toBe(v2);
  // The first trace can load the local contouring stack. Wait for that operation,
  // then check hydration; a generic UI assertion can expire while it is pending.
  const tracedResponse=page.waitForResponse(response=>response.url().endsWith(`/api/projects/${id}/trace-contours`)&&response.request().method()==='POST',{timeout:45000});
  await page.getByRole('button',{name:'Контур маски на срезе'}).click();
  const traced=await tracedResponse;expect(traced.ok()).toBe(true);expect(await traced.json()).toHaveLength(1);
  await expect(page.getByTestId('contour-count')).toHaveText('1');
  await page.getByTestId('contour-item').click();
  const handle=page.getByTestId('viewport-0').locator('svg circle').first();await expect(handle).toBeVisible();
  const b=(await handle.boundingBox())!;await page.mouse.move(b.x+b.width/2,b.y+b.height/2);await page.mouse.down();await page.mouse.move(b.x+b.width/2-12,b.y+b.height/2+6,{steps:8});await page.mouse.up();
  await page.getByRole('button',{name:'⚑ Подозрительный срез'}).click();
  await expect.poll(async()=>{const p=await (await page.request.get('/api/projects/'+id)).json();return p.document.flags.length;},{timeout:15000}).toBe(1);
  await expect(page.getByTestId('save-status')).toContainText('Сохранено');
  const after=await (await page.request.get(`/api/projects/${id}/export/${v2}`)).body();expect(after.equals(original)).toBe(true);
  await page.getByRole('button',{name:'3D-реконструкция'}).click();await expect(page.locator('.surface-main aside')).toContainText('LV ·',{timeout:45000});await expect(page.locator('.surface-main aside')).toContainText('RV ·');
  await expect(page.getByTestId('surface-canvas').locator('canvas')).toBeVisible();
  const surface=(await page.getByTestId('surface-canvas').boundingBox())!;await page.mouse.move(surface.x+surface.width/2,surface.y+surface.height/2);await page.mouse.down();await page.mouse.move(surface.x+surface.width/2+30,surface.y+surface.height/2+20,{steps:6});await page.mouse.up();
  await page.screenshot({path:'../outputs/clinical_integration_3D_SYNTHETIC.png'});await page.getByRole('button',{name:'Закрыть 3D'}).click();
  await page.getByLabel('Показать маску').selectOption('ct');await expect(page.getByRole('button',{name:'Кисть',exact:true})).toBeDisabled();
  await expect.poll(async()=> (await readMPRStats(page)).every(p=>p.gray>100&&p.grayLevels>=16&&p.colored===0)).toBe(true);
  await page.getByLabel('Показать маску').selectOption('draft');await expect(page.getByRole('button',{name:'Кисть',exact:true})).toBeEnabled();
  await page.reload();await expect(page.getByText('● Геометрия проверена')).toBeVisible({timeout:45000});await expect(page.getByTestId('contour-count')).toHaveText('1');
  view=await (await page.request.get('/api/projects/'+id)).json();expect(view.document.base_prediction_id).toBe(v2);expect(view.document.flags).toHaveLength(1);
  const versions=await (await page.request.get(`/api/projects/${id}/versions`)).json();expect(versions.filter((v:any)=>v.source==='AI prediction')).toHaveLength(2);expect(versions.every((v:any)=>!v.approved)).toBe(true);
  await page.getByRole('button',{name:'Передать на проверку →'}).click();await expect(page.locator('.editor header .status')).toHaveText('На проверке');
  await page.getByRole('button',{name:'Выйти',exact:true}).click();
  await page.getByLabel('Логин',{exact:true}).fill('ai-surgeon');await page.getByLabel('Пароль',{exact:true}).fill(process.env.HEART3D_E2E_PASSWORD!);await page.getByRole('button',{name:'Войти в рабочее пространство'}).click();
  await page.getByRole('button',{name:'Открыть разметку'}).click();await expect(page.getByText('● Геометрия проверена')).toBeVisible({timeout:45000});
  await page.getByLabel('Комментарий хирурга').fill('Synthetic fixture review');await page.getByRole('button',{name:'Утвердить экспертную версию'}).click();
  await expect(page.locator('.editor header .status')).toHaveText('Утверждено');
  const approved=await (await page.request.get(`/api/projects/${id}/versions`)).json();
  expect(approved.filter((v:any)=>v.approved)).toHaveLength(1);
  expect(approved.filter((v:any)=>v.source==='AI prediction').every((v:any)=>!v.approved)).toBe(true);
  expect(errors).toEqual([]);
  expect(external.filter(host=>host!=='gc.kis.v2.scr.kaspersky-labs.com')).toEqual([]);
  test.info().annotations.push({type:'blocked_host_antivirus_requests',description:String(external.length)});
});
