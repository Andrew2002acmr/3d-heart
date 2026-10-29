import {test,expect,type Page} from '@playwright/test';
import {installLocalNetworkGuard} from '../scripts/local-network-guard.mjs';
async function login(page:Page){await page.goto('/');await page.getByLabel('Логин',{exact:true}).fill('resident');await page.getByLabel('Пароль',{exact:true}).fill(process.env.HEART3D_E2E_PASSWORD!);await page.getByRole('button',{name:'Войти в рабочее пространство'}).click();await expect(page.getByRole('heading',{name:'Мои исследования'})).toBeVisible();}
async function open(page:Page){await page.getByRole('button',{name:'Открыть разметку'}).click();await expect(page.getByText('● Геометрия проверена')).toBeVisible({timeout:45000});}
test('editable native points, zoom/pan, brush, autosave, relogin, NIfTI and object authorization',async({page,context})=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  const blockedExternal:string[]=[];
  await installLocalNetworkGuard(context,'http://127.0.0.1:5174',(host:string)=>blockedExternal.push(host));
  await login(page);await open(page);
  const canvas=page.getByTestId('viewport-0');const box=(await canvas.boundingBox())!;
  const points=[[.40,.42],[.60,.42],[.60,.58],[.40,.58]].map(([x,y])=>({x:box.x+x*box.width,y:box.y+y*box.height}));
  for(const p of points)await page.mouse.click(p.x,p.y);
  await page.mouse.click(points[0].x,points[0].y);
  await expect(page.getByTestId('contour-count')).toHaveText('1');
  await expect(page.getByTestId('save-status')).toContainText('Сохранено',{timeout:15000});
  let projects=await (await page.request.get('/api/projects')).json();const id=projects[0].id;
  const original=await (await page.request.get('/api/projects/'+id)).json();
  // Move a real Cornerstone control point through pointer events.
  await page.mouse.move(points[1].x,points[1].y);await page.mouse.down();await page.mouse.move(points[1].x+17,points[1].y-12,{steps:8});await page.mouse.up();
  await page.waitForTimeout(400);
  // Native add/delete control-point bindings.
  await page.keyboard.down('Shift');await page.mouse.click((points[0].x+points[1].x)/2,points[0].y-6);await page.keyboard.up('Shift');
  await expect(page.getByTestId('contour-item')).toContainText('5 точек');
  await page.keyboard.down('Control');await page.mouse.click((points[0].x+points[1].x)/2,points[0].y-6);await page.keyboard.up('Control');
  await expect(page.getByTestId('contour-item')).toContainText('4 точек');
  await page.getByRole('button',{name:/Отменить/}).click();await expect(page.getByTestId('contour-item')).toContainText('5 точек');
  await page.getByRole('button',{name:/Повторить/}).click();await expect(page.getByTestId('contour-item')).toContainText('4 точек');
  await page.getByRole('button',{name:'Применить контур'}).click();
  await expect(page.getByTestId('save-status')).toContainText('Сохранено',{timeout:15000});
  let saved=await (await page.request.get('/api/projects/'+id)).json();
  expect(saved.document.contours[0].points.length).toBe(4);
  expect(saved.document.contours[0].points).not.toEqual(original.document.contours[0].points);
  expect(saved.document.contours[0].applied).toBe(true);
  const before=saved.document.contours[0].points;
  const mask=await (await page.request.get(`/api/projects/${id}/mask`)).body();expect([...mask].filter(v=>v===1).length).toBeGreaterThan(10);
  await page.locator('.label-choice').filter({hasText:'RV'}).click();await page.getByRole('button',{name:'Кисть',exact:true}).click();
  const bx=box.x+box.width*.66,by=box.y+box.height*.55;
  await page.mouse.move(bx,by);await page.mouse.down();await page.mouse.move(bx+25,by,{steps:5});await page.mouse.up();
  await expect(page.getByTestId('save-status')).toContainText('Сохранено',{timeout:15000});
  saved=await (await page.request.get('/api/projects/'+id)).json();expect(saved.document.operations.some((o:any)=>o.kind==='brush'&&o.values.includes(2))).toBe(true);
  await page.getByRole('button',{name:'Ластик',exact:true}).click();await page.mouse.click(bx,by);
  await expect.poll(async()=>{const p=await (await page.request.get('/api/projects/'+id)).json();return p.document.operations.some((o:any)=>o.kind==='brush'&&o.values.includes(0));},{timeout:15000}).toBe(true);
  await expect(page.getByTestId('save-status')).toContainText('Сохранено',{timeout:15000});
  saved=await (await page.request.get('/api/projects/'+id)).json();expect(saved.document.operations.some((o:any)=>o.kind==='brush'&&o.values.includes(0))).toBe(true);
  await page.getByRole('button',{name:'Масштаб',exact:true}).click();await page.mouse.move(box.x+box.width*.5,box.y+box.height*.5);await page.mouse.down();await page.mouse.move(box.x+box.width*.5,box.y+box.height*.5+40,{steps:10});await page.mouse.up();
  await page.getByRole('button',{name:'Сдвиг',exact:true}).click();await page.mouse.move(box.x+box.width*.5,box.y+box.height*.5);await page.mouse.down();await page.mouse.move(box.x+box.width*.5+20,box.y+box.height*.5,{steps:8});await page.mouse.up();
  // Camera operations must not mutate physical control points.
  saved=await (await page.request.get('/api/projects/'+id)).json();expect(saved.document.contours[0].points).toEqual(before);
  await page.screenshot({path:'../outputs/web_annotation_editor.png',fullPage:true});
  await page.getByRole('button',{name:'Выйти',exact:true}).click();await expect(page.getByRole('heading',{name:'Вход в кабинет'})).toBeVisible();
  expect((await page.request.get('/api/projects')).status()).toBe(401);
  await login(page);await open(page);await expect(page.getByTestId('contour-count')).toHaveText('1');
  saved=await (await page.request.get('/api/projects/'+id)).json();expect(saved.document.contours[0].points).toEqual(before);
  await page.getByTestId('contour-item').click();
  await page.mouse.move(points[0].x,points[0].y);await page.mouse.down();await page.mouse.move(points[0].x-8,points[0].y+8,{steps:6});await page.mouse.up();
  await expect.poll(async()=>{const p=await (await page.request.get('/api/projects/'+id)).json();return p.document.contours[0].points;},{timeout:15000}).not.toEqual(before);
  await expect(page.getByTestId('save-status')).toContainText('Сохранено',{timeout:15000});
  saved=await (await page.request.get('/api/projects/'+id)).json();expect(saved.document.contours[0].points).not.toEqual(before);
  await page.reload();await expect(page.getByText('● Геометрия проверена')).toBeVisible({timeout:45000});await expect(page.getByTestId('contour-count')).toHaveText('1');
  const download=await Promise.all([page.waitForEvent('download'),page.getByRole('link',{name:'Экспорт NIfTI'}).click()]);await download[0].saveAs('../outputs/web_annotation_export.nii.gz');
  // A different assigned user sees a distinct project, inaccessible to this session.
  const foreign=await context.request.post('/api/auth/login',{data:{username:'foreign-resident',password:process.env.HEART3D_E2E_PASSWORD}});expect(foreign.status()).toBe(200);
  projects=await (await context.request.get('/api/projects')).json();const foreignId=projects[0].id;
  await context.request.post('/api/auth/login',{data:{username:'resident',password:process.env.HEART3D_E2E_PASSWORD}});
  for(const suffix of ['','/volume','/mask','/versions'])expect((await context.request.get(`/api/projects/${foreignId}${suffix}`)).status()).toBe(404);
  expect(errors).toEqual([]);
  // The host antivirus rewrites HTTP/CSP and injects this script. Every such
  // request was aborted above; any application-originated external URL fails.
  expect(blockedExternal.filter(host=>host!=='gc.kis.v2.scr.kaspersky-labs.com')).toEqual([]);
  test.info().annotations.push({type:'blocked_host_antivirus_requests',description:String(blockedExternal.length)});
});
