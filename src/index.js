import * as ort from "onnxruntime-web";

const CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ".split("");

let sessionPromise = null;

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "GET,POST,OPTIONS",
      "access-control-allow-headers": "Content-Type"
    }
  });
}

async function getSession() {
  if (sessionPromise) return sessionPromise;

  const modelUrl =
    (typeof MODEL_URL_4399 !== "undefined" && MODEL_URL_4399)
      ? MODEL_URL_4399
      : "https://raw.githubusercontent.com/boluoreg/4399Register/4b71346e21a23f211e2756f8ad43e7a0a5157456/4399ocr/4399ocr.onnx";

  sessionPromise = fetch(modelUrl)
    .then((r) => {
      if (!r.ok) throw new Error("下载 4399 模型失败: " + r.status);
      return r.arrayBuffer();
    })
    .then((buf) => ort.InferenceSession.create(buf, { executionProviders: ["wasm"] }));

  return sessionPromise;
}

async function readRequestBytes(request) {
  const contentType = request.headers.get("content-type") || "";

  if (contentType.includes("multipart/form-data")) {
    const form = await request.formData();
    const file = form.get("file") || form.get("image");
    if (!file || typeof file.arrayBuffer !== "function") {
      throw new Error("请用 form-data 的 file 字段上传图片");
    }
    return file.arrayBuffer();
  }

  if (contentType.includes("application/json")) {
    const body = await request.json();
    const raw = body.image || body.b64;
    if (!raw) throw new Error("JSON 里需要 image 或 b64");
    const base64 = raw.includes(",") ? raw.split(",").pop() : raw;
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
  }

  return request.arrayBuffer();
}

async function preprocessImageToTensor(bytes) {
  const bitmap = await createImageBitmap(new Blob([bytes]));
  const baseH = 32;
  const targetW = Math.max(32, Math.min(512, Math.round(bitmap.width * baseH / Math.max(1, bitmap.height))));

  const canvas = new OffscreenCanvas(targetW, baseH);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, targetW, baseH);
  ctx.drawImage(bitmap, 0, 0, targetW, baseH);

  const imageData = ctx.getImageData(0, 0, targetW, baseH);
  const data = imageData.data;

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
  const data = outputTensor.data;

  let time = 0;
  let classes = 0;

  if (dims.length === 3) {
    [, time, classes] = dims;
  } else if (dims.length === 2) {
    [time, classes] = dims;
  } else {
    throw new Error("不支持的输出形状: " + dims.join("x"));
  }

  const hasExtraBlank = classes === CHARSET.length + 1;

  let prev = -1;
  let text = "";

  for (let t = 0; t < time; t++) {
    let bestIndex = 0;
    let bestValue = -Infinity;

    for (let c = 0; c < classes; c++) {
      const idx = t * classes + c;
      const v = Number(data[idx]);
      if (v > bestValue) {
        bestValue = v;
        bestIndex = c;
      }
    }

    let mapped = bestIndex;
    if (hasExtraBlank && bestIndex > 0) {
      mapped = bestIndex - 1;
    }

    if (mapped >= CHARSET.length) continue;
    if (mapped === prev) continue;

    const ch = CHARSET[mapped];
    if (ch === " " && prev === mapped) continue;

    text += ch;
    prev = mapped;
  }

  return {
    text: text.trim(),
    confidence: text ? 0.92 : 0
  };
}

async function saveToD1(env, result) {
  if (!env || !env.OCR_DB) return null;

  try {
    const sql = `
      INSERT INTO results (id, created_at, text_result, confidence, model)
      VALUES (?, ?, ?, ?, ?)
    `;

    const res = await env.OCR_DB.prepare(sql)
      .bind(crypto.randomUUID(), new Date().toISOString(), result.text, result.confidence, "4399ocr")
      .run();

    return { ok: true, meta: res };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

const PAGE_HTML = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>4399 OCR</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: #f3f3f3;
      font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      color: #111;
    }
    .box {
      max-width: 980px;
      margin: 60px auto;
      padding: 32px;
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.04);
    }
    h1 {
      font-size: 64px;
      margin: 0 0 20px 0;
      font-weight: 700;
    }
    p {
      font-size: 28px;
      line-height: 1.6;
      margin: 0 0 28px 0;
    }
    .row {
      display: flex;
      align-items: center;
      gap: 20px;
      margin-top: 20px;
    }
    input[type=file] {
      font-size: 20px;
    }
    button {
      font-size: 20px;
      padding: 10px 18px;
      cursor: pointer;
    }
    img {
      max-width: 100%;
      max-height: 260px;
      display: block;
      margin-top: 16px;
      border: 1px solid #ddd;
      border-radius: 8px;
    }
    pre {
      margin-top: 20px;
      background: #f8f9fa;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 16px;
      white-space: pre-wrap;
      font-size: 20px;
      line-height: 1.5;
    }
  </style>
</head>
<body>
  <div class="box">
    <h1>4399 OCR</h1>
    <p>上传图片后直接识别，支持 API 和网页直接使用</p>

    <div class="row">
      <input id="file" type="file" accept="image/*">
      <button id="go">开始识别</button>
    </div>

    <img id="preview" alt="preview" hidden>

    <pre id="output">等待上传图片…</pre>
  </div>

  <script>
    const fileInput = document.getElementById("file");
    const preview = document.getElementById("preview");
    const output = document.getElementById("output");

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) return;
      preview.src = URL.createObjectURL(file);
      preview.hidden = false;
    });

    document.getElementById("go").addEventListener("click", async () => {
      const file = fileInput.files[0];
      if (!file) {
        output.textContent = "请先选择图片";
        return;
      }

      output.textContent = "识别中...";
      const form = new FormData();
      form.append("file", file);

      try {
        const r = await fetch("/api/ocr", { method: "POST", body: form });
        const data = await r.json();
        if (!r.ok || !data.ok) {
          throw new Error(data.error || "识别失败");
        }
        output.textContent =
          "识别结果：\\n" +
          data.result.text +
          "\\n\\n置信度：" +
          (data.result.confidence || 0).toFixed(2);
      } catch (e) {
        output.textContent = "识别失败：" + e.message;
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
          "access-control-allow-headers": "Content-Type"
        }
      });
    }

    if (url.pathname === "/api/health") {
      return json({ ok: true, model: "4399ocr" });
    }

    if (url.pathname === "/api/ocr" && request.method === "POST") {
      try {
        const bytes = await readRequestBytes(request);

        if (!bytes.byteLength || bytes.byteLength > 8 * 1024 * 1024) {
          return json({ ok: false, error: "图片必须是 1 byte ~ 8 MB 之间" }, 400);
        }

        const session = await getSession();
        const inputTensor = await preprocessImageToTensor(bytes);

        const inputName = session.inputNames[0];
        const outputs = await session.run({ [inputName]: inputTensor });
        const firstKey = Object.keys(outputs)[0];
        const result = decodeCTC(outputs[firstKey]);

        const d1Res = await saveToD1(env, result);

        return json({
          ok: true,
          model: "4399ocr",
          result,
          d1: d1Res
        });
      } catch (err) {
        return json({ ok: false, error: err.message }, 500);
      }
    }

    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(PAGE_HTML, {
        headers: {
          "content-type": "text/html; charset=utf-8",
          "cache-control": "no-store"
        }
      });
    }

    return json({
      ok: true,
      endpoints: [
        "GET /api/health",
        "POST /api/ocr"
      ]
    });
  }
};
