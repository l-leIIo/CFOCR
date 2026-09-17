import * as ort from "onnxruntime-web";

/*
 * 关键修复：
 * ONNX Runtime 在 Cloudflare Worker 中无法自动找到 WASM 文件，
 * 所以必须手动指定 WASM 文件所在目录。
 */
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

// 4399ocr.json 中的原始字符集
const CHARSET = [
  " ", "6", "t", "y", "w", "J", "K", "k", "p", "7",
  "8", "9", "n", "j", "P", "q", "D", "G", "c", "N",
  "v", "X", "H", "Y", "5", "0", "h", "R", "f", "r",
  "4", "d", "A", "E", "M", "l", "V", "m", "a", "F",
  "s", "i", "z", "U", "g", "x", "u", "o", "3", "Q",
  "b", "e", "T", "1", "2"
];

const DEFAULT_MODEL_URL =
  "https://raw.githubusercontent.com/boluoreg/4399Register/4b71346e21a23f211e2756f8ad43e7a0a5157456/4399ocr/4399ocr.onnx";

let sessionPromise = null;

/* 返回 JSON */
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

/* 加载并缓存 ONNX 模型 */
async function getSession(env) {
  if (sessionPromise) {
    return sessionPromise;
  }

  const modelUrl = env.MODEL_URL_4399 || DEFAULT_MODEL_URL;

  sessionPromise = fetch(modelUrl)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`模型下载失败，HTTP 状态码：${response.status}`);
      }

      return response.arrayBuffer();
    })
    .then((modelBuffer) => {
      return ort.InferenceSession.create(modelBuffer, {
        executionProviders: ["wasm"]
      });
    });

  return sessionPromise;
}

/* 读取上传的图片 */
async function readImageBytes(request) {
  const contentType = request.headers.get("content-type") || "";

  // multipart/form-data
  if (contentType.includes("multipart/form-data")) {
    const formData = await request.formData();
    const file = formData.get("file") || formData.get("image");

    if (!file || typeof file.arrayBuffer !== "function") {
      throw new Error("没有找到图片，请使用 file 字段上传图片");
    }

    return file.arrayBuffer();
  }

  // JSON Base64
  if (contentType.includes("application/json")) {
    const body = await request.json();
    const base64Text = body.b64 || body.image;

    if (!base64Text) {
      throw new Error("JSON 请求必须包含 b64 或 image 字段");
    }

    const base64 = base64Text.includes(",")
      ? base64Text.split(",").pop()
      : base64Text;

    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);

    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }

    return bytes.buffer;
  }

  // 直接上传图片二进制
  return request.arrayBuffer();
}

/* 图片预处理：灰度、缩放到高度 32 */
async function imageToTensor(imageBytes) {
  const imageBlob = new Blob([imageBytes]);
  const bitmap = await createImageBitmap(imageBlob);

  const targetHeight = 32;

  const targetWidth = Math.max(
    32,
    Math.min(
      512,
      Math.round(
        (bitmap.width * targetHeight) / Math.max(1, bitmap.height)
      )
    )
  );

  const canvas = new OffscreenCanvas(targetWidth, targetHeight);
  const context = canvas.getContext("2d", {
    willReadFrequently: true
  });

  if (!context) {
    throw new Error("无法创建图片处理画布");
  }

  // 白色背景
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, targetWidth, targetHeight);

  context.drawImage(
    bitmap,
    0,
    0,
    targetWidth,
    targetHeight
  );

  const imageData = context.getImageData(
    0,
    0,
    targetWidth,
    targetHeight
  );

  const pixels = imageData.data;
  const floatData = new Float32Array(
    targetWidth * targetHeight
  );

  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const pixelIndex = (y * targetWidth + x) * 4;

      const red = pixels[pixelIndex];
      const green = pixels[pixelIndex + 1];
      const blue = pixels[pixelIndex + 2];

      // RGB 转灰度并归一化到 0~1
      const gray =
        (0.299 * red + 0.587 * green + 0.114 * blue) / 255;

      floatData[y * targetWidth + x] = gray;
    }
  }

  return new ort.Tensor(
    "float32",
    floatData,
    [1, 1, targetHeight, targetWidth]
  );
}

/* CTC 解码 */
function decodeOutput(outputTensor) {
  const dimensions = outputTensor.dims.map(Number);
  const data = outputTensor.data;

  let time;
  let classes;

  if (dimensions.length === 3) {
    // [1, time, classes]
    [, time, classes] = dimensions;
  } else if (dimensions.length === 2) {
    // [time, classes]
    [time, classes] = dimensions;
  } else {
    throw new Error(
      `模型输出形状不支持：${dimensions.join(" x ")}`
    );
  }

  /*
   * 如果输出类别数量比字符集多 1，
   * 说明第 0 类是 blank。
   */
  const hasBlankClass = classes === CHARSET.length + 1;

  let previousIndex = -1;
  let text = "";
  let confidenceTotal = 0;
  let characterCount = 0;

  for (let t = 0; t < time; t++) {
    let bestIndex = 0;
    let bestValue = -Infinity;

    for (let c = 0; c < classes; c++) {
      const value = Number(data[t * classes + c]);

      if (value > bestValue) {
        bestValue = value;
        bestIndex = c;
      }
    }

    // CTC 重复字符只保留一次
    if (bestIndex === previousIndex) {
      continue;
    }

    // 处理 blank 类
    if (hasBlankClass && bestIndex === 0) {
      previousIndex = bestIndex;
      continue;
    }

    const charsetIndex = hasBlankClass
      ? bestIndex - 1
      : bestIndex;

    if (
      charsetIndex >= 0 &&
      charsetIndex < CHARSET.length
    ) {
      text += CHARSET[charsetIndex];
      characterCount++;

      // 将 logits 粗略转换为 0~1 的数值
      const probability =
        bestValue >= 0 && bestValue <= 1
          ? bestValue
          : 1 / (1 + Math.exp(-bestValue));

      confidenceTotal += probability;
    }

    previousIndex = bestIndex;
  }

  const cleanText = text.trim();

  return {
    text: cleanText,
    confidence: characterCount > 0
      ? Number((confidenceTotal / characterCount).toFixed(4))
      : 0
  };
}

/* 保存到 D1，可选 */
async function saveToD1(env, result) {
  if (!env.OCR_DB) {
    return null;
  }

  try {
    await env.OCR_DB
      .prepare(
        `INSERT INTO results
        (id, created_at, text_result, confidence, model)
        VALUES (?, ?, ?, ?, ?)`
      )
      .bind(
        crypto.randomUUID(),
        new Date().toISOString(),
        result.text,
        result.confidence,
        "4399ocr"
      )
      .run();

    return {
      ok: true
    };
  } catch (error) {
    return {
      ok: false,
      error: error.message
    };
  }
}

/*
 * Worker
 */
export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // 处理跨域预检请求
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

    // 健康检查
    if (
      url.pathname === "/api/health" &&
      request.method === "GET"
    ) {
      return json({
        ok: true,
        model: "4399ocr",
        wasm: "configured"
      });
    }

    // OCR API
    if (
      url.pathname === "/api/ocr" &&
      request.method === "POST"
    ) {
      try {
        const imageBytes = await readImageBytes(request);

        if (!imageBytes || imageBytes.byteLength === 0) {
          return json(
            {
              ok: false,
              error: "图片为空"
            },
            400
          );
        }

        if (imageBytes.byteLength > 8 * 1024 * 1024) {
          return json(
            {
              ok: false,
              error: "图片不能超过 8 MB"
            },
            400
          );
        }

        const session = await getSession(env);
        const inputTensor = await imageToTensor(imageBytes);

        const inputName = session.inputNames[0];

        if (!inputName) {
          throw new Error("模型没有找到输入名称");
        }

        const outputs = await session.run({
          [inputName]: inputTensor
        });

        const outputNames = Object.keys(outputs);

        if (outputNames.length === 0) {
          throw new Error("模型没有返回输出");
        }

        const outputTensor = outputs[outputNames[0]];
        const result = decodeOutput(outputTensor);
        const d1 = await saveToD1(env, result);

        return json({
          ok: true,
          model: "4399ocr",
          result,
          d1
        });
      } catch (error) {
        return json(
          {
            ok: false,
            error: error.message
          },
          500
        );
      }
    }

    // 根路径提示
    if (url.pathname === "/") {
      return new Response(
        "4399 OCR API is running. Use POST /api/ocr.",
        {
          status: 200,
          headers: {
            "content-type": "text/plain; charset=utf-8",
            "cache-control": "no-store"
          }
        }
      );
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
