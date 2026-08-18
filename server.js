/* 加贝英语台 · 后端代理
 * 职责：
 *   POST /ocr  —— 照片 → 视觉模型 → [{en,phonetic,pos,zh,ex}]（无 key 时回退本地 Tesseract）
 *   POST /asr  —— 录音 → 通义 Paraformer → {text}
 *   GET  /     —— 自检页（显示 key/服务商/接口状态）
 * 关键：API Key 只存在本服务端（.env），绝不进前端，避免泄露。
 */
require('dotenv').config();
// 把任何未捕获的崩溃打到 Render 日志，避免“静默 502”
process.on('unhandledRejection', e => console.error('[CRASH] UNHANDLED_REJECTION:', e && e.stack || e));
process.on('uncaughtException', e => console.error('[CRASH] UNCAUGHT_EXCEPTION:', e && e.stack || e));
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json({ limit: '20mb' }));
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 20 * 1024 * 1024 } });

const PORT = process.env.PORT || 3000;
const VISION_PROVIDER = (process.env.VISION_PROVIDER || 'qwen').toLowerCase();
const ASR_PROVIDER = (process.env.ASR_PROVIDER || 'qwen').toLowerCase();
const API_KEY = process.env.API_KEY || process.env.DASHSCOPE_API_KEY || '';
const VISION_MODEL = process.env.VISION_MODEL || 'qwen3-vl-plus';
const ASR_MODEL = process.env.ASR_MODEL || 'paraformer-v2';
const ARK_ENDPOINT = process.env.ARK_ENDPOINT || '';

const OCR_PROMPT = `你是一个英语单词卡片识别器。图片里是单词学习卡片，每行形如「单词 /音标/ 词性. 中文意思」（音标和词性可能缺失）。
请识别并返回 JSON 数组，每个元素包含字段：en(英文单词)、phonetic(国际音标，不要斜杠，如 nekst)、pos(词性缩写如 n./v./adj./adv.)、zh(中文意思)、ex(英文例句，没有就填空字符串)。
只返回 JSON 数组本身，不要任何解释、不要 markdown 代码块标记。`;

const sleep = ms => new Promise(r => setTimeout(r, ms));

/* ---------- 解析模型返回的 JSON ---------- */
function extractWords(content) {
  if (!content) return [];
  let s = String(content).trim();
  // 去掉 ```json ... ``` 或 ``` ... ```
  s = s.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/i, '').trim();
  // 截取第一个 [ 到最后一个 ]
  const a = s.indexOf('['), b = s.lastIndexOf(']');
  if (a >= 0 && b > a) s = s.slice(a, b + 1);
  let arr;
  try { arr = JSON.parse(s); } catch (e) {
    // 退路：逐行解析
    return parseOcrText(content);
  }
  if (!Array.isArray(arr)) {
    if (Array.isArray(arr.words)) arr = arr.words;
    else return [];
  }
  return arr.map(x => ({
    en: String(x.en || x.word || x.term || '').trim(),
    phonetic: String(x.phonetic || x.ph || '').replace(/[\/]/g, '').trim(),
    pos: String(x.pos || x.part || '').trim(),
    zh: String(x.zh || x.meaning || x.chinese || '').trim(),
    ex: String(x.ex || x.example || x.sentence || '').trim()
  })).filter(x => x.en);
}

/* ---------- 无 key 时的本地解析（兜底，处理「en /ph/ pos. zh」或「en 中文」）---------- */
function parseOcrText(text) {
  const rows = [];
  for (let raw of String(text || '').split(/\r?\n/)) {
    raw = raw.trim();
    if (!raw) continue;
    const m = raw.match(/^([a-zA-Z][a-zA-Z'\-]+)\s*(?:\/\s*([^\/]+?)\s*\/)?\s*(?:([a-z]+\.)?\s*)?[\u4e00-\u9fff]/);
    const en = (raw.match(/^[a-zA-Z][a-zA-Z'\-]+/g) || [])[0] || '';
    const cjk = (raw.match(/[\u4e00-\u9fff][\u4e00-\u9fff，、；：！？\sA-Za-z0-9\-]*\u4e00-\u9fff|[一-鿿][一-鿿，、；：！？\sA-Za-z0-9\-]*/) || [''])[0] || '';
    if (en) rows.push({ en: en.toLowerCase(), phonetic: '', pos: '', zh: cjk.trim(), ex: '' });
  }
  return rows;
}

/* ---------- 视觉模型调用 ---------- */
async function visionOCR(buf, mime) {
  const b64 = buf.toString('base64');
  const dataUri = `data:${mime || 'image/jpeg'};base64,${b64}`;
  if (VISION_PROVIDER === 'doubao') return callDoubaoVision(dataUri);
  return callQwenVision(dataUri);
}

async function callQwenVision(dataUri) {
  const url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions';
  const body = {
    model: VISION_MODEL,
    messages: [{ role: 'user', content: [{ image: dataUri }, { text: OCR_PROMPT }] }]
  };
  const r = await fetch(url, {
    method: 'POST',
    signal: AbortSignal.timeout(30000),
    headers: { 'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const j = await r.json();
  if (j.error) throw new Error(j.error.message || JSON.stringify(j.error));
  const content = j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content || '';
  return extractWords(content);
}

async function callDoubaoVision(dataUri) {
  if (!ARK_ENDPOINT) throw new Error('豆包需配置 ARK_ENDPOINT');
  const url = 'https://ark.cn-beijing.volces.com/api/v3/chat/completions';
  const body = {
    model: ARK_ENDPOINT,
    messages: [{ role: 'user', content: [{ type: 'image_url', image_url: { url: dataUri } }, { type: 'text', text: OCR_PROMPT }] }]
  };
  const r = await fetch(url, {
    method: 'POST',
    signal: AbortSignal.timeout(30000),
    headers: { 'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const j = await r.json();
  if (j.error) throw new Error(j.error.message || JSON.stringify(j.error));
  const content = j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content || '';
  return extractWords(content);
}

/* ---------- 无 key 本地 Tesseract 兜底 ---------- */
let _tesseract = null;
async function localOCR(buf, mime) {
  try {
    if (!_tesseract) _tesseract = require('tesseract.js');
  } catch (e) {
    throw new Error('未安装 tesseract.js（可选依赖），且无 API Key，无法做本地识别。请配置云端 Key，或用「批量导入」粘贴 AI 文本。');
  }
  const { data } = await _tesseract.recognize(buf, 'eng+chi_sim', { logger: () => {} });
  return parseOcrText(data.text);
}

/* ---------- 语音识别：通义 Paraformer（异步轮询）---------- */
async function asrQwen(buf, ext) {
  const form = new FormData();
  form.append('file', new Blob([buf], { type: mimeFromExt(ext) }), 'rec.' + ext);
  form.append('task', JSON.stringify({ model: ASR_MODEL, input: { file_urls: [] }, parameters: { language_hints: ['zh', 'en'] } }));
  const r = await fetch('https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription', {
    method: 'POST',
    signal: AbortSignal.timeout(30000),
    headers: { 'Authorization': 'Bearer ' + API_KEY, 'X-DashScope-Async': 'enable' },
    body: form
  });
  const j = await r.json();
  if (j.code) throw new Error(j.message || JSON.stringify(j));
  const taskId = j.output && j.output.task_id;
  if (!taskId) throw new Error('创建识别任务失败：' + JSON.stringify(j));
  for (let i = 0; i < 40; i++) {
    await sleep(1000);
    const s = await fetch('https://dashscope.aliyuncs.com/api/v1/tasks/' + taskId, {
      signal: AbortSignal.timeout(10000),
      headers: { 'Authorization': 'Bearer ' + API_KEY }
    });
    const sj = await s.json();
    const st = sj.output && sj.output.task_status;
    if (st === 'SUCCEEDED') {
      const url = sj.output.results && sj.output.results[0] && sj.output.results[0].transcription_url;
      const t = await (await fetch(url)).json();
      return (t.transcripts && t.transcripts[0] && t.transcripts[0].text) || t.text || '';
    }
    if (st === 'FAILED') throw new Error('识别失败');
  }
  throw new Error('识别超时');
}

function mimeFromExt(ext) {
  return ({ wav: 'audio/wav', mp3: 'audio/mpeg', pcm: 'audio/pcm', flac: 'audio/flac', m4a: 'audio/mp4', webm: 'audio/webm', ogg: 'audio/ogg' }[ext] || 'audio/webm');
}

/* ---------- 路由 ---------- */
app.post('/ocr', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: '未收到图片' });
    let words, source;
    if (API_KEY && VISION_PROVIDER !== 'none') {
      try {
        words = await visionOCR(req.file.buffer, req.file.mimetype);
        source = 'vision:' + VISION_PROVIDER;
      } catch (e) {
        // 视觉失败，回退本地
        words = await localOCR(req.file.buffer, req.file.mimetype);
        source = 'local-tesseract(fallback:' + e.message + ')';
      }
    } else {
      words = await localOCR(req.file.buffer, req.file.mimetype);
      source = 'local-tesseract(no-key)';
    }
    res.json({ words: words || [], source });
  } catch (e) {
    res.status(500).json({ error: e.message || String(e) });
  }
});

app.post('/asr', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: '未收到音频' });
    if (!API_KEY || ASR_PROVIDER === 'none') {
      return res.status(400).json({ error: 'no-key', message: '未配置云端语音识别，请改用浏览器识别或在 .env 配置 ASR_PROVIDER=qwen 与 API_KEY。' });
    }
    const ext = (req.file.originalname.split('.').pop() || 'webm').toLowerCase();
    const text = await asrQwen(req.file.buffer, ext);
    res.json({ text: text || '' });
  } catch (e) {
    res.status(500).json({ error: e.message || String(e) });
  }
});

app.get('/', (req, res) => {
  const hasKey = !!API_KEY;
  res.type('html').send(`<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>加贝后端自检</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>body{font-family:system-ui;max-width:640px;margin:40px auto;padding:0 16px;color:#234}
  h1{font-size:20px}.ok{color:#1a9e5f;font-weight:700}.no{color:#e35d6b;font-weight:700}
  code{background:#f1f5f2;padding:2px 6px;border-radius:4px}.card{border:1px solid #e3e9e5;border-radius:10px;padding:14px 16px;margin:12px 0}</style></head>
  <body><h1>加贝英语台 · 后端自检</h1>
  <div class="card">
    <p>API Key：${hasKey ? '<span class="ok">已配置</span>' : '<span class="no">未配置（OCR 走本地 Tesseract 兜底，ASR 需走浏览器）</span>'}</p>
    <p>视觉服务商：<code>${VISION_PROVIDER}</code>　语音服务商：<code>${ASR_PROVIDER}</code></p>
    <p>接口：<code>POST /ocr</code>（照片→4字段）　<code>POST /asr</code>（录音→文本）</p>
    <p>状态：<span class="ok">服务正常</span></p>
  </div>
  <p class="muted">把本地址（含端口）填到「加贝英语台 → 设置 → 云端服务地址」即可。本服务已托管在云端，手机直接访问，无需内网穿透。</p>
  </body></html>`);
});

// 兜底错误中间件：任何未处理异常都返回 JSON，避免进程崩溃导致 502
app.use((err, req, res, next) => {
  console.error('[CRASH] EXPRESS_ERROR:', err && err.stack || err);
  if (!res.headersSent) res.status(500).json({ error: err && err.message || String(err) });
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log('[加贝后端] 已启动 http://localhost:' + PORT + '  key=' + (API_KEY ? '已配置' : '未配置') + ' vision=' + VISION_PROVIDER + ' asr=' + ASR_PROVIDER);
  });
}

module.exports = { extractWords, parseOcrText, app };
