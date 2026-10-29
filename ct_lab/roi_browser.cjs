// Real clicks in an isolated temporary Chrome profile, with native-coordinate assertions.
const {chromium}=require('../web/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path');let browser;
(async()=>{
 browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1600,height:1200}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8510');await page.getByText('Heart Localization & ROI Segmentation',{exact:true}).first().click({timeout:90000});
 const frame=page.frameLocator('iframe[title*="spatial_three_plane"]');
 await frame.locator('canvas').first().waitFor({timeout:90000});const records=[];
 if(+(await frame.locator('body').getAttribute('data-box-count'))!==1)throw Error('Missing predicted ROI');
 for(const plane of ['Axial XY','Coronal XZ','Sagittal YZ']){
  const canvas=frame.locator(`canvas[data-plane="${plane}"][data-ready="true"]`);await canvas.waitFor();await canvas.scrollIntoViewIfNeeded();await page.waitForTimeout(300);
  await canvas.evaluate(c=>c.addEventListener('click',e=>{const r=c.getBoundingClientRect(),axes=c.dataset.axes.split(',').map(Number),xyz=document.body.dataset.xyz.split(',').map(Number);
   xyz[axes[0]]=Math.floor((e.clientX-r.left)/r.width*c.width);xyz[axes[1]]=c.height-1-Math.floor((e.clientY-r.top)/r.height*c.height);window.__roiExpected=xyz;},{capture:true,once:true}));
  const b=await canvas.boundingBox();await page.mouse.click(b.x+b.width*.41,b.y+b.height*.47);
  const expected=await frame.locator('body').evaluate(()=>window.__roiExpected);
  await frame.locator(`body[data-xyz="${expected.join(',')}"]`).waitFor({timeout:30000});
  records.push({plane,expected,actual:(await frame.locator('body').getAttribute('data-xyz')).split(',').map(Number)});console.log('Verified',plane,expected);
 }
 async function select(label,value){await page.locator('[data-testid="stSelectbox"]').filter({hasText:label}).getByRole('combobox').click();await page.getByRole('option',{name:value,exact:true}).click();await page.waitForTimeout(500);await page.locator('[data-testid="stApp"][data-test-script-state="notRunning"]').waitFor({timeout:90000});}
 for(const mode of ['False positives','False negatives','Class confusion','Segmentation']){await select('ROI error view',mode);await page.waitForTimeout(500);await frame.locator('canvas[data-ready="true"]').first().waitFor();}
 const root=path.resolve(__dirname,'../outputs/heart_localization_roi_05');
 const meshMethod=fs.existsSync(path.join(root,'surfaces/ct_1001/D_predicted_256/index.json'))?'D_predicted_256':'GT';
 await select('ROI segmentation model',meshMethod);
 const expander=page.locator('[data-testid="stExpander"]').filter({hasText:'3D: bbox и Heart3D surfaces'});
 const details=expander.locator('details');if(!(await details.evaluate(d=>d.open)))await expander.locator('summary').click();
 await page.waitForFunction(()=>[...document.querySelectorAll('.js-plotly-plot')].some(p=>p.data?.some(t=>t.type==='mesh3d')&&p.data?.some(t=>t.name==='Selected voxel')),{},{timeout:60000});
 const position=await page.evaluate(()=>{const p=[...document.querySelectorAll('.js-plotly-plot')].find(p=>p.data?.some(t=>t.name==='Selected voxel'));const t=p.data.find(t=>t.name==='Selected voxel');return[t.x[0],t.y[0],t.z[0]];});
 const xyz=records.at(-1).actual;
 // ImageCHD68 ct_1001 header: unit diagonal RAS, origin [1,1,1], audited in source analysis.
 if(position.some((v,i)=>v!==xyz[i]+1))throw Error('3D point is not source affine XYZ');
 await page.locator('iframe[title*="spatial_three_plane"]').evaluate(e=>e.scrollIntoView({block:'start'}));
 await page.evaluate(()=>document.querySelector('[data-testid="stMain"]').scrollBy(0,-80));await page.waitForTimeout(300);
 await page.screenshot({path:path.join(root,'browser_roi_planes.png'),fullPage:false});
 await expander.locator('.js-plotly-plot').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(root,'browser_roi_surfaces.png'),fullPage:false});
 const alerts=await page.locator('[data-testid="stException"]').allTextContents();if(alerts.length)throw Error(alerts.join('\n'));
 fs.writeFileSync(path.join(root,'browser_verification.json'),JSON.stringify({plane_clicks:records,predicted_box_visible:true,error_views:4,selected_native_voxel:xyz,selected_3D_header_position:position,Heart3D_meshes:true,mesh_method:meshMethod,page_errors:errors},null,2));
 if(errors.length)throw Error(errors.join('\n'));console.log('ROI BROWSER OK');
})().catch(e=>{console.error(e);process.exitCode=1}).finally(async()=>{if(browser)await browser.close()});
