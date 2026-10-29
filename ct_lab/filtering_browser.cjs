// Headless E2E in an isolated browser context; no Heart3D auth or personal profile.
const {chromium}=require('../web/node_modules/playwright');
const path=require('node:path');
const fs=require('node:fs');
let activeBrowser;
(async()=>{
  const browser=await chromium.launch({channel:'chrome',headless:true});
  activeBrowser=browser;
  const page=await browser.newPage({viewport:{width:1600,height:1200}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8510',{waitUntil:'domcontentloaded'});
  await page.getByText('Filtering & Feature Analysis',{exact:true}).first().click({timeout:90000});
  await page.getByText('Выделение точек связано с КТ',{exact:false}).first().waitFor({timeout:90000});
  await page.waitForFunction(()=>document.querySelectorAll('.js-plotly-plot').length>=8,{timeout:30000});
  const before=await page.locator('.js-plotly-plot').count();
  if(before<8)throw new Error('Expected CT/scatter/projection figures');
  const output=path.resolve(__dirname,'../outputs/filtering_feature_01');
  fs.mkdirSync(output,{recursive:true});
  await page.screenshot({path:path.join(output,'browser_filtering.png'),fullPage:true});
  fs.writeFileSync(path.join(output,'browser_trace_shapes.json'),JSON.stringify(await page.evaluate(()=>[...document.querySelectorAll('.js-plotly-plot')].map(p=>(p.data||[]).map(t=>({name:t.name,type:t.type,customKeys:Object.keys(t.customdata||{}).slice(0,8),customShape:t.customdata?.shape,length:t.x?.length})))),null,2));
  await page.evaluate(()=>{
    const plot=[...document.querySelectorAll('.js-plotly-plot')].find(p=>p.data?.some(t=>t.customdata?.length));
    if(!plot)throw new Error('No sampled scatter found');
    plot.id='feature-scatter-e2e';
    plot.on('plotly_selected',e=>{if(e?.points?.length){window.__featureSelected=e.points[0].customdata;window.__featureAllSelected=e.points.map(p=>p.customdata);}});
  });
  await page.locator('#feature-scatter-e2e').scrollIntoViewIfNeeded();
  const target=await page.evaluate(()=>{
    const plot=document.getElementById('feature-scatter-e2e');
    const trace=plot.data.find(t=>t.customdata?.length);
    const rect=plot.getBoundingClientRect(),layout=plot._fullLayout;
    return {x:rect.left+layout.xaxis._offset,y:rect.top+layout.yaxis._offset,width:layout.xaxis._length,height:layout.yaxis._length,dragmode:layout.dragmode};
  });
  await page.mouse.move(target.x+2,target.y+2);await page.mouse.down();
  await page.mouse.move(target.x+2,target.y+target.height-2,{steps:12});
  await page.mouse.move(target.x+target.width-2,target.y+target.height-2,{steps:12});
  await page.mouse.move(target.x+target.width-2,target.y+2,{steps:12});
  await page.mouse.move(target.x+2,target.y+2,{steps:12});await page.mouse.up();
  await page.waitForFunction(()=>[...document.querySelectorAll('.js-plotly-plot')].some(p=>(p.data||[]).some(t=>t.name==='Выбрано на графике'&&t.x?.length)),{timeout:30000});
  const selected=await page.evaluate(()=>window.__featureSelected);
  if(!selected)throw new Error('No actual Plotly selection event');
  await page.waitForTimeout(1000);
  const markers=await page.evaluate(()=>[...document.querySelectorAll('.js-plotly-plot')].flatMap(p=>p.data||[]).filter(t=>t.name==='Выбрано на графике').map(t=>({x:Array.from(t.x),y:Array.from(t.y)})));
  if(markers.length<2 || !markers.every(m=>m.x[0]===selected[2]&&m.y[0]===selected[1]))throw new Error('Selection did not map back to source CT');
  const allSelected=await page.evaluate(()=>window.__featureAllSelected);
  const expected=new Set(allSelected.map(p=>`${p[2]},${p[1]}`));
  if(!markers.every(m=>m.x.length===expected.size&&m.x.every((x,i)=>expected.has(`${x},${m.y[i]}`))))throw new Error('Area selection coordinate set differs');
  await page.locator('[data-testid="stMainBlockContainer"]').screenshot({path:path.join(output,'browser_filtering.png')});
  fs.writeFileSync(path.join(output,'browser_verification.json'),JSON.stringify({figures:before,selection_mode:'actual mouse lasso',selected_point:selected,
    selected_count:expected.size,ct_marker_counts:markers.map(m=>m.x.length),all_coordinate_sets_match:true,page_errors:errors},null,2));
  await browser.close();if(errors.length)throw new Error(errors.join('\n'));
  console.log('BROWSER OK: real Plotly point maps to source CT',selected);
})().catch(e=>{console.error(e);process.exitCode=1}).finally(async()=>{if(activeBrowser)await activeBrowser.close()});
