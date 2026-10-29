const {chromium}=require('../web/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');let browser;
(async()=>{
 const root=path.resolve(__dirname,'../outputs/training_dynamics_06'),run=path.join(root,'runs/baseline_s20261009_n48');
 const files=['step000480.pt','step003840.pt'].map(n=>path.join(run,n));
 const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');const before=files.map(sha);
 browser=await chromium.launch({channel:'chrome',headless:true});const page=await browser.newPage({viewport:{width:1600,height:1200}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8510');await page.getByText('Training Dynamics Lab',{exact:true}).first().click({timeout:90000});
 await page.getByText('Checkpoint Explorer',{exact:true}).click();
 const slider=page.getByRole('slider',{name:'Dynamics native Z',exact:true});await slider.waitFor({timeout:90000});await slider.press('ArrowRight');
 await page.waitForTimeout(500);await page.locator('[data-testid="stApp"][data-test-script-state="notRunning"]').waitFor({timeout:90000});const z=await slider.getAttribute('aria-valuenow');
 async function select(value){await page.getByText('Checkpoint Explorer',{exact:true}).click();await page.locator('[data-testid="stSelectbox"]').filter({hasText:'Dynamics checkpoint'}).getByRole('combobox').click();await page.getByRole('option',{name:value,exact:true}).click();await page.waitForTimeout(500);await page.locator('[data-testid="stApp"][data-test-script-state="notRunning"]').waitFor({timeout:90000});await page.getByText('Checkpoint Explorer',{exact:true}).click();}
 const changes=[];
 for(const checkpoint of ['step000480','step003840','CHD68 v1 historical']){
  await select(checkpoint);const after=await slider.getAttribute('aria-valuenow');if(after!==z)throw Error('Native Z changed with checkpoint');changes.push({checkpoint,nativeZ:+after});
 }
 await select('step003840');await page.screenshot({path:path.join(root,'browser_checkpoint_explorer.png'),fullPage:false});
 await page.getByText('Learning Curves',{exact:true}).click();await page.screenshot({path:path.join(root,'browser_learning_curves.png'),fullPage:false});
 if((await page.locator('[data-testid="stException"]').count())||errors.length)throw Error('UI errors '+errors.join(';'));
 if(before.some((v,i)=>v!==sha(files[i])))throw Error('Checkpoint mutated during UI interactions');
 fs.writeFileSync(path.join(root,'browser_verification.json'),JSON.stringify({changes,nativeZ_preserved:true,checkpoint_SHA256_before:before,checkpoint_SHA256_after:files.map(sha),page_errors:errors},null,2));console.log('DYNAMICS BROWSER OK',changes);
})().catch(e=>{console.error(e);process.exitCode=1}).finally(async()=>{if(browser)await browser.close()});
