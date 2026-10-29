// Real browser interactions in an isolated temporary Chrome profile.
const {chromium}=require('../web/node_modules/playwright');
const fs=require('node:fs');const path=require('node:path');let browser;
(async()=>{
 browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1600,height:1200}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8510');
 await page.getByText('Learnable Feature Explorer',{exact:true}).first().click({timeout:90000});
 const frame=page.frameLocator('iframe[title*="spatial_three_plane"]');
 await frame.locator('canvas').first().waitFor({timeout:90000});
 const records=[];
 for(const plane of ['Axial XY','Coronal XZ','Sagittal YZ']){
  const canvas=frame.locator(`canvas[data-plane="${plane}"][data-ready="true"]`);await canvas.waitFor();await canvas.scrollIntoViewIfNeeded();await page.waitForTimeout(300);
  const before=(await frame.locator('body').getAttribute('data-xyz')).split(',').map(Number);
  const metadata=await canvas.evaluate(c=>({axes:c.dataset.axes.split(',').map(Number),width:c.width,height:c.height}));
  await canvas.evaluate(c=>c.addEventListener('click',e=>{const r=c.getBoundingClientRect(),axes=c.dataset.axes.split(',').map(Number),xyz=document.body.dataset.xyz.split(',').map(Number);
    xyz[axes[0]]=Math.floor((e.clientX-r.left)/r.width*c.width);xyz[axes[1]]=c.height-1-Math.floor((e.clientY-r.top)/r.height*c.height);window.__expectedSpatialClick=xyz;},{capture:true,once:true}));
  const box=await canvas.boundingBox();
  await page.mouse.click(box.x+box.width*.4,box.y+box.height*.45);
  const expected=await frame.locator('body').evaluate(()=>window.__expectedSpatialClick);
  await frame.locator('body').filter({has:frame.locator('canvas')}).waitFor();
  try{await frame.locator(`body[data-xyz="${expected.join(',')}"]`).waitFor({timeout:30000});}
  catch(e){console.error('PLANE FAILURE',{plane,before,expected,actual:await frame.locator('body').getAttribute('data-xyz'),errors});throw e;}
  const after=(await frame.locator('body').getAttribute('data-xyz')).split(',').map(Number);
  if(after.join(',')!==expected.join(','))throw new Error('Crosshair mismatch '+plane);
  records.push({plane,expected,actual:after});
  console.log('Verified',plane);
 }
 // Select measured scatter points with an actual mouse lasso.
 await page.evaluate(()=>{const p=[...document.querySelectorAll('.js-plotly-plot')].find(p=>p.data?.some(t=>t.customdata?.length));
 p.id='spatial-e2e-scatter';p.on('plotly_selected',e=>{window.__spatialSelection=(e?.points||[]).map(p=>p.customdata);});});
 const scatter=page.locator('#spatial-e2e-scatter');await scatter.scrollIntoViewIfNeeded();
 const box=await scatter.evaluate(p=>{const r=p.getBoundingClientRect(),l=p._fullLayout;return{x:r.left+l.xaxis._offset,y:r.top+l.yaxis._offset,w:l.xaxis._length,h:l.yaxis._length}});
 await page.mouse.move(box.x+2,box.y+2);await page.mouse.down();
 for(const [x,y] of [[box.x+2,box.y+box.h-2],[box.x+box.w-2,box.y+box.h-2],[box.x+box.w-2,box.y+2],[box.x+2,box.y+2]])await page.mouse.move(x,y,{steps:12});
 await page.mouse.up();
 await frame.locator('body:not([data-selected-count="0"])').waitFor({timeout:30000});
 const selected=await page.evaluate(()=>window.__spatialSelection);
 const xyz=(await frame.locator('body').getAttribute('data-xyz')).split(',').map(Number);
 if(!selected.some(p=>p.slice(0,3).join(',')===xyz.join(',')))throw new Error('Crosshair not at a selected source voxel');
 const selectedCount=+(await frame.locator('body').getAttribute('data-selected-count'));
 if(selectedCount!==new Set(selected.map(p=>p[3])).size)throw new Error('Region voxel ID count mismatch');
 const root=path.resolve(__dirname,'../outputs/learnable_feature_03');
 await page.locator('[data-testid="stSelectbox"]').filter({hasText:'Learnable model'}).getByRole('combobox').click();
 await page.getByRole('option',{name:'CE',exact:true}).click();
 const inspector=page.locator('[data-testid="stSelectbox"]').filter({hasText:'Embedding inspector layer'});
 await inspector.getByRole('combobox').waitFor({timeout:90000});
 await inspector.getByRole('combobox').click();await page.getByRole('option',{name:'embedding',exact:true}).click();
 console.log('Waiting for actual embedding chart');
 try{await page.waitForFunction(()=>[...document.querySelectorAll('.js-plotly-plot')].some(p=>p._fullLayout?.title?.text?.replace(/<[^>]*>/g,'')==='embedding'),{},{timeout:90000});}
 catch(e){console.error(await page.evaluate(()=>[...document.querySelectorAll('.js-plotly-plot')].map(p=>({title:p._fullLayout?.title?.text,count:p.calcdata?.[0]?.length}))));throw e;}
 const activationCount=await page.evaluate(()=>[...document.querySelectorAll('.js-plotly-plot')].find(p=>p._fullLayout?.title?.text?.replace(/<[^>]*>/g,'')==='embedding')?.calcdata?.[0]?.length);
 if(activationCount!==16)throw new Error('Expected 16 embedding bars, got '+activationCount);
 await inspector.scrollIntoViewIfNeeded();await page.screenshot({path:path.join(root,'browser_embedding.png'),fullPage:false});
 await page.evaluate(()=>document.querySelector('[data-testid="stMain"]').scrollTo(0,0));
 await page.waitForTimeout(400);await page.screenshot({path:path.join(root,'browser_learnable.png'),fullPage:false});
 fs.writeFileSync(path.join(root,'browser_verification.json'),JSON.stringify({plane_clicks:records,selectedCount,crosshair:xyz,scatter_crosshair_matches:true,neural_model:'CE e16',inspected_layer:'embedding',activationCount,page_errors:errors},null,2));
 if(errors.length)throw new Error(errors.join('\n'));
 console.log('LEARNABLE BROWSER OK',records.length,'planes; selected',selectedCount,'; CE embedding inspector');
})().catch(e=>{console.error(e);process.exitCode=1}).finally(async()=>{if(browser)await browser.close()});
