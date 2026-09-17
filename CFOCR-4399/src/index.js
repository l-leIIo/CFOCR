import * as ort from "onnxruntime-web";

const CHARSET = [
  " ", "6", "t", "y", "w", "J", "K", "k", "p", "7", "8", "9", "n", "j", "P", "q",
  "D", "G", "c", "N", "v", "X", "H", "Y", "5", "0", "h", "R", "f", "r", "4", "d",
  "A", "E", "M", "l", "V", "m", "a", "F", "s", "i", "z", "U", "g", "x", "u", "o",
  "3", "Q", "b", "e", "T", "1", "2"
];

const DEFAULT_MODEL = "https://raw.githubusercontent.com/boluoreg/4399Register/4b71346e21a23f211e2756f8ad43e7a0a5157456/4399ocr/4399ocr.onnx";

let sessionPromise = null;

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "GET,POST,OPTIONS",
      "access-control-allow-headers": "content-type"
    }
  });
}

async function getSession(env) {
  if (!sessionPromise) {
    const modelUrl = env.MODEL_URL_4399 || DEFAULT_MODEL;
    sessionPromise = fetch(modelUrl)
      .then(r => {
        if (!r.ok) throw new Error(`model download failed: ${r.status}`);
        return r.arrayBuffer();
      })
      .then(buf => ort.InferenceSession.create(buf, { executionProviders: ["wasm"] }));
  }
  return sessionPromise;
}

async function readRequestBytes(request) {
  const contentType = request.headers.get("content-type") || "";

  if (contentType.includes("multipart/form-data")) {
    const form = await request.formData();
    const file = form.get("file") || form.get("image");
    if (!file || typeof file.arrayBuffer !== "function") {
      throw new Error("send image in form field 'file'");
    }
    return file.arrayBuffer();
  }

  if (contentType.includes("application/json")) {
    const body = await request.json();
    const raw = body.image || body.b64;
    if (!raw) throw new Error("JSON requires image or b64");
    const base64 = raw.includes(",") ? raw.split(",").pop() : raw;
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes.buffer;
  }

  return request.arrayBuffer();
}

async function decodeImage(bytes) {
  const bitmap = await createImageBitmap(new Blob([bytes]));
  const baseH = 32;
  const targetW = Math.max(32, Math.min(512, Math.round(bitmap.width * baseH / Math.max(1, bitmap.height))));
  const canvas = new OffscreenCanvas(targetW, baseH);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, targetW, baseH);
  ctx.drawImage(bitmap, 0, 0, targetW, baseH);

  const imageData = ctx.getImageData(0, 0, targetW, baseH);
  const { data } = imageData;

  const floatData = new Float32Array(baseH * targetW);
  for (let y = 0; y < baseH; y++) {
    for (let x = 0; x < targetW; x++) {
      const i = (y * targetW + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
      floatData[y * targetW + x] = gray;
    }
  }

  return new ort.Tensor("float32", floatData, [1, 1, baseH, targetW]);
}

function decodeCTC(outputTensor) {
  const dims = outputTensor.dims.map(Number);

  let time;
  let classes;
  if (dims.length === 3) {
    [, time, classes] = dims;
  } else if (dims.length === 2) {
    [time, classes] = dims;
  } else {
    throw new Error(`unsupported output shape: ${dims.join("x")}`);
  }

  const data = outputTensor.data;
  const blankIndex = 0;
  let prev = -1;
  let text = "";
  let scoreSum = 0;
  let count = 0;

  for (let t = 0; t < time; t++) {
    let bestIndex = 0;
    let bestValue = -Infinity;

    for (let c = 0; c < classes; c++) {
      const idx = t * classes + c;
      const val = Number(data[idx]);
      if (val > bestValue) {
        bestValue = val;
        bestIndex = c;
      }
    }

    if (bestIndex !== blankIndex && bestIndex !== prev) {
      const mapped = Math.max(0, bestIndex - 1);
      if (mapped < CHARSET.length) {
        text += CHARSET[mapped];
        scoreSum += bestValue;
        count += 1;
      }
    }
    prev = bestIndex;
  }

  return {
    text: text.trim(),
    confidence: count > 0 ? Math.min(0.9999, Math.max(0.0, scoreSum / count)) : 0
  };
}

async function saveResult(env, result) {
  if (!env.OCR_DB) return;
  await env.OCR_DB.prepare(
    "INSERT INTO results (id, created_at, text_result, confidence, model) VALUES (?, ?, ?, ?, ?)"
  ).bind(
    crypto.randomUUID(),
    new Date().toISOString(),
    result.text,
    result.confidence,
    "4399ocr"
  ).run();
}

const PAGE_HTML = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>4399 OCR</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 18px; }
    .card { border: 1px solid #dfe3e8; border-radius: 14px; padding: 24px; background: #fff; box-shadow: 0 8px 20px rgba(0,0,0,0.04); }
    input[type=file] { margin-bottom: 12px; }
    button { padding: 10px 16px; border: 0; border-radius: 10px; background: #2563eb; color: #fff; cursor: pointer; }
    img { max-width: 100%; max-height: 260px; display: block; margin-top: 12px; }
    pre { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; white-space: pre-wrap; }
  </style>
</head>
<body>
  <div class="card">
    <h1>4399 OCR</h1>
    <p>涓婁紶鍥剧墖锛岀洿鎺ヨ瘑鍒枃鏈紙鏀寔 API 鍜岀綉椤电洿鎺ヤ娇鐢級</p>

    <input id="file" type="file" accept="image/*" />
    <button id="go">寮€濮嬭瘑鍒?/button>

    <img id="preview" alt="preview" hidden />

    <pre id="output">绛夊緟涓婁紶鍥剧墖鈥?/pre>
  </div>

  <script>
    const fileInput = document.getElementById('file');
    const output = document.getElementById('output');
    const preview = document.getElementById('preview');
    const btn = document.getElementById('go');

    fileInput.addEventListener('change', () => {
      const file = fileInput.files[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      preview.src = url;
      preview.hidden = false;
    });

    btn.addEventListener('click', async () => {
      const file = fileInput.files[0];
      if (!file) {
        output.textContent = '璇峰厛閫夋嫨鍥剧墖';
        return;
      }
      output.textContent = '璇嗗埆涓€?;
      const form = new FormData();
      form.append('file', file);

      try {
        const resp = await fetch('/api/ocr', { method: 'POST', body: form });
        const data = await resp.json();
        if (!resp.ok || !data.ok) {
          throw new Error(data.error || '璇锋眰澶辫触');
        }
        const text = data.result.text || '';
        const conf = (data.result.confidence || 0).toFixed(3);
        output.textContent = '璇嗗埆缁撴灉锛歕\n' + text + '\\n\\n缃俊搴︼細' + conf;
      } catch (e) {
        output.textContent = '璇嗗埆澶辫触锛? + e.message;
      }
    });
  </script>
</body>
</html>`;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "access-control-allow-origin": "*",
          "access-control-allow-methods": "GET,POST,OPTIONS",
          "access-control-allow-headers": "content-type"
        }
      });
    }

    if (url.pathname === "/api/health") {
      return new Response(JSON.stringify({ ok: true, model: "4399ocr" }), {
        headers: { "content-type": "application/json; charset=utf-8", "access-control-allow-origin": "*" }
      });
    }

    if (url.pathname === "/api/ocr" && request.method === "POST") {
      try {
        const bytes = await readRequestBytes(request);
        if (!bytes.byteLength || bytes.byteLength > 8 * 1024 * 1024) {
          return new Response(JSON.stringify({ ok: false, error: "image must be 1 byte to 8 MB" }), {
            status: 400,
            headers: { "content-type": "application/json; charset=utf-8", "access-control-allow-origin": "*" }
          });
        }

        const session = await getSession(env);
        const inputTensor = await decodeImage(bytes);

        const inputName = session.inputNames[0];
        const outputs = await session.run({ [inputName]: inputTensor });
        const firstOutput = outputs[Object.keys(outputs)[0]];
        const result = decodeCTC(firstOutput);

        await saveResult(env, result);

        return new Response(JSON.stringify({
          ok: true,
          model: "4399ocr",
          result
        }), {
          headers: { "content-type": "application/json; charset=utf-8", "access-control-allow-origin": "*" }
        });
      } catch (err) {
        return new Response(JSON.stringify({ ok: false, error: err.message }), {
          status: 500,
          headers: { "content-type": "application/json; charset=utf-8", "access-control-allow-origin": "*" }
        });
      }
    }

    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(PAGE_HTML, {
        headers: { "content-type": "text/html; charset=utf-8" }
      });
    }

    return new Response(JSON.stringify({
      ok: true,
      endpoints: [
        "GET /api/health",
        "POST /api/ocr"
      ]
    }), {
      headers: { "content-type": "application/json; charset=utf-8", "access-control-allow-origin": "*" }
    });
  }
};
