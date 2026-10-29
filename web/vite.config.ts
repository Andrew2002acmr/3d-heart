import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({plugins:[react()], server:{host:'127.0.0.1', port:5173, strictPort:true,
  fs:{strict:true,deny:['.env','.env.*','*.{crt,pem}','**/.git/**','**/.web-data/**','**/private_clinical/**','**/srcreens_22_10_2026/**','**/outputs/**','**/datasets/**','**/checkpoints/**','**/*.dcm','**/*.DCM','**/*.nii','**/*.nii.gz','**/*.npy','**/*.npz','**/*.pt','**/*.pth','**/*.sqlite3*','**/*.private.json']},
  headers:{'Content-Security-Policy':"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self' ws://127.0.0.1:5173 ws://127.0.0.1:5174; img-src 'self' data: blob:; worker-src 'self' blob:; font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'",'Referrer-Policy':'no-referrer'},
  proxy:{'/api':{target:process.env.HEART3D_API_TARGET || 'http://127.0.0.1:8000',changeOrigin:false}}},
  resolve:{alias:{events:'events/',url:'url/'}},
  build:{chunkSizeWarningLimit:2500}});
