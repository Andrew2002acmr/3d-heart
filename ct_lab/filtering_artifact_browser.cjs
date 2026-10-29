const {chromium}=require('../web/node_modules/playwright');
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
let browser;
(async()=>{
  browser=await chromium.launch({channel:'chrome',headless:true});
  const page=await browser.newPage({viewport:{width:1600,height:1100}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const root=path.resolve(__dirname,'../outputs/filtering_feature_01');
  await page.goto(pathToFileURL(path.join(root,'linked_ct1001.html')).href);
  await page.waitForFunction(()=>document.getElementById('linked_0')?.data?.length===8);
  const before=await page.evaluate(()=>document.getElementById('linked_0').data[6].x[0]);
  await page.selectOption('#config_0','2');
  await page.waitForFunction(initial=>document.getElementById('linked_0').data[6].x[0]!==initial,before);
  const result=await page.evaluate(()=>{
    const plot=document.getElementById('linked_0'),trace=plot.data[6];
    const custom=trace.customdata[0];
    plot.emit('plotly_selected',{points:[{curveNumber:6,customdata:custom}]});
    return {before_filter_intensity:null,after_filter_intensity:trace.x[0],selected_customdata:custom,
            source_markers:plot.data[1].x,processed_markers:plot.data[3].x,
            pca_after_points:document.getElementById('pca_0').data[2].x.length,
            umap_present:!!document.getElementById('umap_0')};
  });
  await page.waitForTimeout(200);
  if(result.source_markers[0]!==result.selected_customdata[2]||result.processed_markers[0]!==result.selected_customdata[2])throw new Error('Offline report selection did not map to CT');
  result.before_filter_intensity=before;result.page_errors=errors;
  if(errors.length)throw new Error(errors.join('\n'));
  fs.writeFileSync(path.join(root,'offline_browser_verification.json'),JSON.stringify(result,null,2));
  console.log('OFFLINE LINKED REPORT OK',JSON.stringify(result));
})().catch(e=>{console.error(e);process.exitCode=1}).finally(async()=>{if(browser)await browser.close()});
