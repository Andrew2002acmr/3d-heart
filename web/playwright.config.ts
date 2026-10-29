import {defineConfig} from '@playwright/test';
import {randomBytes} from 'node:crypto';
import path from 'node:path';
const root=path.resolve('..');
process.env.HEART3D_E2E_PASSWORD ||= randomBytes(24).toString('base64url');
// Never inherit the operator's real HEART3D_WEB_DATA into test servers.
process.env.HEART3D_E2E_ROOT ||= path.join(root,'outputs','web-e2e-private',randomBytes(8).toString('hex'));
process.env.HEART3D_WEB_DATA = process.env.HEART3D_E2E_ROOT;
process.env.HEART3D_ALLOWED_ORIGINS='http://127.0.0.1:5174';
export default defineConfig({testDir:'./e2e',workers:1,fullyParallel:false,timeout:90000,
  globalSetup:'./e2e/setup.ts',reporter:[['list'],['json',{outputFile:'../outputs/web_e2e_results.json'}]],
  use:{baseURL:'http://127.0.0.1:5174',channel:'chrome',headless:true,serviceWorkers:'block',actionTimeout:15000,viewport:{width:1500,height:1000},
    trace:'retain-on-failure',screenshot:'only-on-failure',launchOptions:{args:['--force_high_performance_gpu','--use-angle=d3d11']}},
  webServer:[{command:'..\\.venv\\Scripts\\python.exe -m uvicorn heart3d.web.app:create_app --factory --host 127.0.0.1 --port 8001 --no-access-log',
    cwd:'.',env:{PYTHONPATH:root},url:'http://127.0.0.1:8001/api/health',reuseExistingServer:false},
    {command:'npm.cmd run dev -- --port 5174',url:'http://127.0.0.1:5174',
      env:{HEART3D_API_TARGET:'http://127.0.0.1:8001'},reuseExistingServer:false}]});
