# CFOCR - 4399 OCR for Cloudflare

This project is a direct OCR service for Cloudflare Workers.

Features:
- Website OCR: open `/` and upload file
- API OCR: send `POST /api/ocr`
- D1 persistence: each successful OCR result is saved to D1 if `OCR_DB` exists
- 4399 model support: uses the model from boluoreg/4399Register

Quick start:
1. Install dependencies:
   npm install
2. Download the 4399 model:
   bash scripts/download_4399.sh
3. Create a D1 database in Cloudflare Dashboard
4. Fill the `database_id` in `wrangler.toml`
5. Deploy:
   npx wrangler login
   npx wrangler deploy

API:
curl -F "file=@test.png" https://YOUR_WORKER.example.com/api/ocr

Website:
Open https://YOUR_WORKER.example.com/

Notes:
- This project is built for the 4399 OCR ONNX model.
- The model is fetched from GitHub raw URL by default.
- If you see WASM issues, enable Node compatibility in Cloudflare Worker or deploy in a normal runtime.
