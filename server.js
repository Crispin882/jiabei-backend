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
const crypto = require('crypto');
const fs = require('fs');
const http = require('http');
// 实时语音识别依赖 ws；若未安装，主服务仍正常启动，仅 /asr-realtime 不可用（避免整个进程崩溃导致部署失败）
let WebSocketServer = null, WebSocket = null, WS_AVAILABLE = false;
try {
  ({ WebSocketServer, WebSocket } = require('ws'));
  WS_AVAILABLE = true;
} catch (e) {
  console.warn('[WARN] 未检测到 ws 模块，实时语音识别(/asr-realtime)已禁用。如需“边说边出字”，请执行 `npm install ws` 后重新部署。OCR/ASR/TTS 不受影响。');
}

const app = express();
app.use(cors());
app.use(express.json({ limit: '20mb' }));
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 20 * 1024 * 1024 } });

const PORT = process.env.PORT || 3000;
const VISION_PROVIDER = (process.env.VISION_PROVIDER || 'qwen').toLowerCase();
const ASR_PROVIDER = (process.env.ASR_PROVIDER || 'qwen').toLowerCase();
const API_KEY = (process.env.API_KEY || process.env.DASHSCOPE_API_KEY || '').trim();
const VISION_MODEL = process.env.VISION_MODEL || 'qwen3-vl-plus';
const ASR_MODEL = process.env.ASR_MODEL || 'paraformer-v2';
const ARK_ENDPOINT = process.env.ARK_ENDPOINT || '';
// 实时语音识别（paraformer-realtime-v2）业务空间专属域名；WorkspaceId 由用户开通实时服务后提供
const RT_WORKSPACE = (process.env.DASHSCOPE_WORKSPACE || 'ws-8tkk6xjf5kj0refl').trim();
// 实时识别 WebSocket 地址：默认用业务空间专属域名；若需改用百炼标准地址（非 workspace），
// 可在 Render 设 DASHSCOPE_RT_URL=wss://dashscope.aliyuncs.com/api-ws/v1/inference/ 覆盖。
const RT_WS_URL = (process.env.DASHSCOPE_RT_URL || ('wss://' + RT_WORKSPACE + '.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference')).trim();

const OCR_PROMPT = `你是一个英语单词卡片识别器。图片里是单词学习卡片，每行形如「单词 /音标/ 词性. 中文意思」（音标和词性可能缺失）。
请识别并返回 JSON 数组，每个元素包含字段：en(英文单词)、phonetic(国际音标，不要斜杠，如 nekst)、pos(词性缩写如 n./v./adj./adv.)、zh(中文意思)、ex(英文例句，没有就填空字符串)。
只返回 JSON 数组本身，不要任何解释、不要 markdown 代码块标记。`;

// 短文拍照识图提示词：识别英文短文 + 对应中文翻译，按原文段落/分栏顺序组织为 paragraphs
const PASSAGE_OCR_PROMPT = `你是一个英语短文识别器。输入是 1 到多张图片，内容为同一篇英语短文（可能分多张拍摄），可能带有中文翻译。图片可能有以下三种版式之一：
(a) 英文句子与中文翻译逐行交替（一句英文，紧接着一句中文）；
(b) 左右双栏（左栏英文，右栏中文）；
(c) 先整段英文、再整段中文。
请按图片顺序（第 1 张、第 2 张、第 3 张……）通读所有英文句子及其对应的中文翻译，合并成「一篇完整短文」，严格按原文出现的段落/分栏顺序排列。只输出合法 JSON（不要 markdown、不要任何解释）：
{"title":string, "level":"入门"|"初级"|"中级"|"高级"（无法判断可留空字符串 ""）, "paragraphs":[[{"en":string,"zh":string}]]}
要求：
- paragraphs 是段落数组；每段是句子数组；每句含 en（英文句子）和 zh（对应中文翻译）。
- en 与 zh 必须一一对应；若某句没有对应翻译，zh 设为空字符串 ""。
- 一句英文就是一句，不要把相邻多句错误地合并成一个 en；也不要把一句拆成两句。
- 段落划分遵循原文的换行或分栏；某张图内原文有分段则保留段落；跨页处原文无明确分段则顺接为同一段（必要时可在此处另起一段）。
- 多张图的内容是同一篇短文的不同部分，请按顺序拼接，不要重复、不要遗漏。
- 忠实原图，不要编造原文没有的句子；英文拼写尽量准确。`;

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
  let content;
  if (VISION_PROVIDER === 'doubao') content = await callDoubaoVision(dataUri, OCR_PROMPT);
  else content = await callQwenVision(dataUri, OCR_PROMPT);
  return extractWords(content);
}

// 短文拍照识图：复用视觉模型，换用 PASSAGE_OCR_PROMPT，返回 {title, level, paragraphs}
// 支持多张图片（同一篇短文分多张拍）：buffers/mimes 为数组，按顺序合并成一篇完整短文
async function visionPassageOCR(buffers, mimes) {
  const dataUris = (buffers || []).map(function (b, i) {
    return `data:${mimes[i] || 'image/jpeg'};base64,${b.toString('base64')}`;
  });
  let content;
  if (VISION_PROVIDER === 'doubao') content = await callDoubaoVisionMulti(dataUris, PASSAGE_OCR_PROMPT);
  else content = await callQwenVisionMulti(dataUris, PASSAGE_OCR_PROMPT);
  return extractPassage(content);
}

// 多图视觉调用：content 数组中按顺序放多张 image_url，再跟文本提示
async function callQwenVisionMulti(dataUris, promptText) {
  const url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions';
  const content = dataUris.map(function (u) { return { type: 'image_url', image_url: { url: u } }; });
  content.push({ type: 'text', text: promptText || PASSAGE_OCR_PROMPT });
  const body = { model: VISION_MODEL, messages: [{ role: 'user', content: content }] };
  const r = await fetch(url, {
    method: 'POST',
    signal: AbortSignal.timeout(90000),
    headers: { 'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const j = await r.json();
  if (j.error) { const c = j.error.code || ''; throw new Error(`云端视觉识别失败[${c}]：${j.error.message || JSON.stringify(j.error)}`); }
  const out = j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content || '';
  try { recordUsage(VISION_MODEL, j.usage || null, false); } catch (_) {}
  return out;
}

async function callDoubaoVisionMulti(dataUris, promptText) {
  if (!ARK_ENDPOINT) throw new Error('豆包需配置 ARK_ENDPOINT');
  const url = 'https://ark.cn-beijing.volces.com/api/v3/chat/completions';
  const content = dataUris.map(function (u) { return { type: 'image_url', image_url: { url: u } }; });
  content.push({ type: 'text', text: promptText || PASSAGE_OCR_PROMPT });
  const body = { model: ARK_ENDPOINT, messages: [{ role: 'user', content: content }] };
  const r = await fetch(url, {
    method: 'POST',
    signal: AbortSignal.timeout(60000),
    headers: { 'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const j = await r.json();
  if (j.error) throw new Error(j.error.message || JSON.stringify(j.error));
  return j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content || '';
}

async function callQwenVision(dataUri, promptText) {
  const url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions';
  // 兼容模式(OpenAI 格式)要求图像用 image_url 结构，不能用原生 {image:...}
  const body = {
    model: VISION_MODEL,
    messages: [{
      role: 'user',
      content: [
        { type: 'image_url', image_url: { url: dataUri } },
        { type: 'text', text: promptText || OCR_PROMPT }
      ]
    }]
  };
  const r = await fetch(url, {
    method: 'POST',
    signal: AbortSignal.timeout(60000),
    headers: { 'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const j = await r.json();
  if (j.error) { const c = j.error.code || ''; throw new Error(`云端视觉识别失败[${c}]：${j.error.message || JSON.stringify(j.error)}`); }
  const content = j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content || '';
  try { recordUsage(VISION_MODEL, j.usage || null, false); } catch (_) {}   // 模型池用量统计
  return content;
}

async function callDoubaoVision(dataUri, promptText) {
  if (!ARK_ENDPOINT) throw new Error('豆包需配置 ARK_ENDPOINT');
  const url = 'https://ark.cn-beijing.volces.com/api/v3/chat/completions';
  const body = {
    model: ARK_ENDPOINT,
    messages: [{ role: 'user', content: [{ type: 'image_url', image_url: { url: dataUri } }, { type: 'text', text: promptText || OCR_PROMPT }] }]
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
  return content;
}

// 解析视觉模型返回的短文 JSON，规范化为 {title, level, paragraphs:[[{en,zh}]]}
function extractPassage(content) {
  if (!content) return { title: '', level: '', paragraphs: [] };
  let s = String(content).trim();
  s = s.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/i, '').trim();
  const a = s.indexOf('{'), b = s.lastIndexOf('}');
  if (a >= 0 && b > a) s = s.slice(a, b + 1);
  let obj;
  try { obj = JSON.parse(s); }
  catch (_) {
    // 容错：去尾逗号 + 补对象/数组元素间逗号后重试一次
    try { obj = JSON.parse(s.replace(/,\s*([\]\}])/g, '$1').replace(/\}\s*\{/g, '},{').replace(/\]\s*\[/g, '],[')); }
    catch (e2) { return { title: '', level: '', paragraphs: [] }; }
  }
  if (!obj || !Array.isArray(obj.paragraphs)) return { title: '', level: '', paragraphs: [] };
  const paragraphs = obj.paragraphs.map(function (para) {
    return (para || []).map(function (s2) {
      if (typeof s2 === 'string') return { en: s2.trim(), zh: '' };
      return { en: (s2.en || '').trim(), zh: (s2.zh || '').trim() };
    }).filter(function (x) { return x.en; });
  }).filter(function (p) { return p.length; });
  return { title: (obj.title || '').trim(), level: (obj.level || '').trim(), paragraphs: paragraphs };
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
async function asrQwen(audioUrl) {
  // dashscope transcription 接口只接受公网 URL（不支持 file://），audioUrl 为后端中转公网地址
  const taskBody = {
    model: ASR_MODEL,
    input: { file_urls: [audioUrl] },
    parameters: { language_hints: ['zh', 'en'] }
  };
  const r = await fetch('https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription', {
    method: 'POST',
    signal: AbortSignal.timeout(30000),
    headers: { 'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json', 'X-DashScope-Async': 'enable' },
    body: JSON.stringify(taskBody)
  });
  const j = await r.json();
  if (j.code) throw new Error(j.message || JSON.stringify(j));
  const taskId = j.output && j.output.task_id;
  if (!taskId) throw new Error('创建识别任务失败：' + JSON.stringify(j));
  for (let i = 0; i < 50; i++) {
    await sleep(1000);
    const s = await fetch('https://dashscope.aliyuncs.com/api/v1/tasks/' + taskId, {
      signal: AbortSignal.timeout(10000),
      headers: { 'Authorization': 'Bearer ' + API_KEY }
    });
    const sj = await s.json();
    const st = sj.output && sj.output.task_status;
    if (st === 'SUCCEEDED') {
      const resultUrl = sj.output.results && sj.output.results[0] && sj.output.results[0].transcription_url;
      if (!resultUrl) throw new Error('转写完成但无结果地址');
      const t = await (await fetch(resultUrl)).json();
      const text = (t.transcripts && t.transcripts[0] && t.transcripts[0].text) || t.text || '';
      try { recordUsage(ASR_MODEL, null, false); } catch (_) {}             // 模型池用量统计
      // 没有有效语音时 Paraformer 会返回字面量 “SUCCESS_WITH_NO_VALID_FRAGMENT”，当作空识别（前端提示“没识别到声音”）
      if (/SUCCESS_WITH_NO_VALID_FRAGMENT|NO_VALID_FRAGMENT/i.test(text)) return '';
      return text;
    }
    if (st === 'FAILED') {
      const detail = (sj.output && (sj.output.message || sj.output.task_status_message)) || sj.message || JSON.stringify(sj).slice(0, 200);
      // 同上：偶尔以 FAILED 形式返回“无片段”，按空识别处理，避免误报成错误
      if (/SUCCESS_WITH_NO_VALID_FRAGMENT|NO_VALID_FRAGMENT/i.test(detail)) return '';
      throw new Error(detail); // 不再加“识别失败：”前缀，由前端统一加，避免双重前缀
    }
  }
  throw new Error('识别超时');
}

function mimeFromExt(ext) {
  return ({ wav: 'audio/wav', mp3: 'audio/mpeg', pcm: 'audio/pcm', flac: 'audio/flac', m4a: 'audio/mp4', webm: 'audio/webm', ogg: 'audio/ogg' }[ext] || 'audio/webm');
}

/* ---------- 异步任务（避免 Render 代理 25-30s 超时导致 502）---------- */
const tasks = new Map();
function newTask() {
  const id = Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
  tasks.set(id, { status: 'processing', startedAt: Date.now() });
  if (tasks.size > 200) tasks.delete(tasks.keys().next().value);
  return id;
}
function finishTask(id, patch) { const t = tasks.get(id); if (t) Object.assign(t, patch); }
function consumeTask(id) { const t = tasks.get(id); if (t) { if (t.status === 'done' || t.status === 'error') tasks.delete(id); } return t; }

// 临时音频中转：录音以随机 token 存内存，暴露公开 GET 路由，供 dashscope 下载转写
// 原因：dashscope transcription 接口只接受公网 URL，不支持 file:// 引用，故用本服务公网地址中转
const audioStore = new Map(); // token -> { buf, ext, ts }
function setAudio(token, buf, ext) { audioStore.set(token, { buf, ext, ts: Date.now() }); if (audioStore.size > 100) { const k = audioStore.keys().next().value; audioStore.delete(k); } }
function getAudio(token) { const a = audioStore.get(token); if (a) audioStore.delete(token); return a; }

/* ---------- 免费数据源薄代理（零成本 · 与付费 OCR/ASR 完全独立） ----------
 * 仅代理免费、无需 Key 的公开数据源，用于「开口说」换一批/更多，不碰 DashScope、不花钱。
 *  - GET /free-sentence ?q=主题词&limit=    → Tatoeba 英中双语例句（真实母语句子+人工中文，服务端缓存，失败回退空）
 *  - GET /free-dialog   ?offset=            → 本地适龄对话子集随机抽（DailyDialog 风格自编）
 *  - GET /free-passage  ?offset=            → 本地分级短文随机抽
 * 前端约定：所有免费源请求都走与 OCR/ASR 相同的 cloudUrl（Render 后端），无需新增配置。
 */
const _freeCache = new Map(); // key -> { ts, data }
const FREE_CACHE_TTL = 6 * 60 * 60 * 1000; // 6 小时
function freeCacheGet(key) {
  const c = _freeCache.get(key);
  if (c && Date.now() - c.ts < FREE_CACHE_TTL) return c.data;
  return null;
}
function freeCacheSet(key, data) {
  _freeCache.set(key, { ts: Date.now(), data });
  if (_freeCache.size > 300) { const k = _freeCache.keys().next().value; _freeCache.delete(k); }
}
function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
// Tatoeba 英中双语例句（真实母语者句子 + 人工中文翻译，权威、适龄、免费、CC-BY）。
// 浏览器直连 CORS 不稳且国内常不可达，故经后端代理并缓存。
// 当前 API：GET /v1/sentences?lang=eng&q="词"&trans:lang=cmn&word_count=3-9&sort=random
//  - q 用双引号包裹 = 精确包含该词；word_count=3-9 只取短句（更适合孩子跟读）
//  - sort=random 增加多样；trans:lang=cmn 只返回带中文翻译的句子
// 响应结构：{ data:[ { text:"英文", translations:[ { text:"中文", lang:"cmn" } ] } ] }
const _BAD_WORDS = /(sex|porn|drug|kill|die|dead|blood|war|fuck|shit|ass|bitch|hell|damn|wtf|rape|weed|alcohol|drunk|smoke|gun|murder|naked|nude|breast|penis|vagina|condom|pregnant|abortion|suicide|stupid|idiot)/i;

// 繁→简转换：Tatoeba 返回的中文翻译是繁体，孩子/家长更习惯简体，故服务端统一转简体。
// opencc-js 为纯 JS、零成本；若未安装则跳过转换（不影响主流程，仅显示繁体，绝不让进程崩溃）。
let _toSimp = null, _convInit = false;
function toSimplified(text) {
  if (!text) return text;
  if (!_convInit) {
    _convInit = true;
    try {
      const OpenCC = require('opencc-js');
      _toSimp = OpenCC.Converter({ from: 'tw', to: 'cn' });
      console.log('[加贝] opencc-js 已加载，Tatoeba 繁体中文将自动转简体。');
    } catch (e) {
      console.warn('[WARN] 未检测到 opencc-js，Tatoeba 繁体中文将不转简体。如需简体，请执行 `npm install opencc-js` 后重新部署。');
      _toSimp = null;
    }
  }
  if (_toSimp) { try { return _toSimp(text); } catch (e) { return text; } }
  return text;
}

async function fetchTatoeba(q, limit) {
  const safeQ = String(q || '').trim().toLowerCase();
  if (!safeQ) return [];
  const url = 'https://api.tatoeba.org/v1/sentences?lang=eng&q=' + encodeURIComponent('"' + safeQ + '"') +
    '&trans:lang=cmn&word_count=3-9&sort=random&limit=' + (parseInt(limit) || 8);
  const r = await fetch(url, { signal: AbortSignal.timeout(12000), headers: { 'Accept': 'application/json' } });
  if (!r.ok) throw new Error('tatoeba http ' + r.status);
  const j = await r.json();
  const rows = (j && Array.isArray(j.data) ? j.data : []).map(function (it) {
    const zhRaw = (it.translations && it.translations[0] && it.translations[0].text) || '';
    return { en: (it.text || '').trim(), zh: toSimplified(zhRaw.trim()) };
  }).filter(function (x) {
    // 只保留：含目标词、有中文翻译、不含不良词（给孩子用的轻量把关）
    return x.en && x.zh && x.en.toLowerCase().includes(safeQ) && !_BAD_WORDS.test(x.en);
  });
  return rows;
}

// 服务端代理 Free Dictionary API：为「主题词库」稳定补全音标/例句/词性（国内可达、零成本）。
// 浏览器直连 dictionaryapi.dev 在国外/家庭网络常超时（表现为“离线”），故改由后端代理并缓存。
async function fetchDictApi(word, acc) {
  const w = String(word || '').trim().toLowerCase();
  const url = 'https://api.dictionaryapi.dev/api/v2/entries/en/' + encodeURIComponent(w);
  const r = await fetch(url, { signal: AbortSignal.timeout(10000), headers: { 'Accept': 'application/json' } });
  if (!r.ok) throw new Error('dict http ' + r.status);
  const d = await r.json();
  if (!Array.isArray(d) || !d[0] || d[0].title) throw new Error('dict notfound');
  // dictionaryapi.dev 返回同一词的多个词条版本（数组），音标/例句常分布在后面版本，
  // 需全量遍历所有版本取第一个非空字段，否则像 cat 这类主词条首义无例句会返回空。
  // 同时：该源的「顶层 phonetic」与「phonetics[0].text」默认是英式（英式 RP），
  // 美式音标藏在 audio 链接含 -us 的 phonetics 条目里。按用户口音优先选对应音标。
  let wordOut = '', phonetic = '', phoneticUS = '', phoneticUK = '', pos = '', example = '';
  const a = (acc || 'en-US').toLowerCase();
  const wantUS = a.indexOf('us') >= 0;
  const wantUK = a.indexOf('gb') >= 0;
  for (const ent of d) {
    if (!wordOut && ent.word) wordOut = ent.word;
    if (!phonetic) {
      if (ent.phonetic) phonetic = ent.phonetic;
      else (ent.phonetics || []).forEach(function (ph) { if (!phonetic && ph.text) phonetic = ph.text; });
    }
    (ent.phonetics || []).forEach(function (ph) {
      if (!ph.text) return;
      const audio = (ph.audio || '').toLowerCase();
      if (!phoneticUS && /(^|[-_])(us|american)/.test(audio)) phoneticUS = ph.text;
      if (!phoneticUK && /(^|[-_])(uk|gb|british)/.test(audio)) phoneticUK = ph.text;
    });
    for (const m of (ent.meanings || [])) {
      if (!pos) pos = m.partOfSpeech || '';
      for (const df of (m.definitions || [])) {
        if (!example && df.example) example = df.example;
      }
    }
    if (phonetic && pos && example) break; // 三个字段都拿到就提前结束
  }
  // 按口音选音标：美式优先用含 -us 的音标，英式优先用含 -uk/-gb 的；取不到则回退默认（英式）
  let chosen = phonetic;
  if (wantUS && phoneticUS) chosen = phoneticUS;
  else if (wantUK && phoneticUK) chosen = phoneticUK;
  return { word: wordOut || w, phonetic: chosen, phoneticUS: phoneticUS, phoneticUK: phoneticUK, pos: pos, example: example };
}
app.get('/free-word', async (req, res) => {
  const w = String(req.query.w || '').slice(0, 40).trim();
  const acc = String(req.query.acc || 'en-US').slice(0, 8).trim() || 'en-US';
  if (!w) return res.json({ ok: true, w: '', phonetic: '', pos: '', example: '' });
  const key = 'word:v3:' + acc + ':' + w;
  const cached = freeCacheGet(key);
  if (cached) return res.json(Object.assign({ ok: true, w: w, cached: true }, cached));
  try {
    const d = await fetchDictApi(w, acc);
    freeCacheSet(key, d);
    res.json(Object.assign({ ok: true, w: w }, d));
  } catch (e) {
    // 代理失败不致命：返回空，前端回退到“仅中文释义”并提示手动填写
    res.json({ ok: false, w: w, phonetic: '', pos: '', example: '', error: (e && e.message || String(e)) });
  }
});

app.get('/free-sentence', async (req, res) => {
  const q = String(req.query.q || '').slice(0, 40).trim();
  const limit = parseInt(req.query.limit) || 8;
  if (!q) return res.json({ ok: true, q: '', items: [], note: 'no-query' });
  const key = 'sent:' + q.toLowerCase();
  const cached = freeCacheGet(key);
  if (cached) return res.json({ ok: true, q: q, items: cached, cached: true });
  try {
    const items = await fetchTatoeba(q, limit);
    freeCacheSet(key, items);
    res.json({ ok: true, q: q, items: items });
  } catch (e) {
    // 代理失败不致命：返回空，前端回退到离线精选集
    res.json({ ok: false, q: q, items: [], error: (e && e.message || String(e)) });
  }
});

// 本地适龄对话 / 短文子集（与前端 assets/data 同源思路，但放在后端便于“换一批”随机且离线）
let _ddSafe = null, _pgSafe = null;
try { _ddSafe = require('./dailydialog_safe.json'); } catch (e) { _ddSafe = []; }
// 内置短文库：直接内联，保证部署 server.js 这一个文件就必然带中文翻译（zh 与 lines 逐句对应）。
// 若外部 passages_safe.json 存在且确实含中文，则优先用外部（便于后续扩展）；否则回退内联版本。
const BUILTIN_PASSAGES = [
  { title:"My Red Cat", level:"入门", lines:["I have a small red cat.","Her name is Mimi.","She likes to sleep on my bed.","Every morning she says meow to me.","I love my little red cat."],
    zh:["我有一只小小的红猫。","她的名字叫咪咪。","她喜欢睡在我的床上。","每天早晨她都对我喵喵叫。","我爱我的小红猫。"] },
  { title:"A Sunny Day", level:"入门", lines:["Today is a sunny day.","The sky is blue and the sun is warm.","I go to the park with my friend.","We run and laugh on the green grass.","We are very happy."],
    zh:["今天是个晴天。","天空蓝蓝的，阳光暖暖的。","我和朋友一起去公园。","我们在绿草地上奔跑欢笑。","我们特别开心。"] },
  { title:"My School", level:"初级", lines:["My school is big and clean.","There are many classrooms and a library.","My teacher is kind and patient.","I read books and make new friends at school.","I like going to school every day."],
    zh:["我的学校又大又干净。","有很多教室和一个图书馆。","我的老师既善良又有耐心。","我在学校读书，还交到了新朋友。","我每天都喜欢去上学。"] },
  { title:"The Little Dog", level:"初级", lines:["A little dog is lost in the street.","He looks sad and hungry.","A boy gives him some water and bread.","The dog wags his tail to say thank you.","Now they are good friends."],
    zh:["一只小狗在街上迷路了。","它看起来又伤心又饿。","一个男孩给了它一些水和面包。","小狗摇着尾巴说谢谢。","现在它们成了好朋友。"] },
  { title:"Apple Tree", level:"初级", lines:["We plant a small apple tree in spring.","We water it when the sun is hot.","In autumn the tree has red apples.","My mother makes a sweet apple pie.","The whole family enjoys the pie together."],
    zh:["春天我们种下一棵小苹果树。","太阳热的时候我们给它浇水。","秋天树上结满了红苹果。","妈妈做了一个香甜的苹果派。","全家人一起分享这个派。"] },
  { title:"My Family", level:"初级", lines:["There are four people in my family.","My father is a doctor and my mother is a teacher.","My sister and I go to the same school.","We eat dinner and talk about our day.","I feel warm and safe with my family."],
    zh:["我家有四口人。","爸爸是医生，妈妈是老师。","姐姐和我在同一所学校。","我们一边吃晚饭，一边聊今天的事。","和家人在一起我觉得温暖又安心。"] },
  { title:"Rainy Morning", level:"中级", lines:["It is raining when I wake up in the morning.","The rain taps softly on the window.","I put on my yellow raincoat and boots.","On the way to school I see a small frog.","Rainy days can be quiet and beautiful too."],
    zh:["早上我醒来时，外面正在下雨。","雨点轻轻敲打着窗户。","我穿上黄色的雨衣和雨靴。","上学路上我看见一只小青蛙。","雨天也可以安静又美好。"] },
  { title:"The Helpful Robot", level:"中级", lines:["My uncle makes a small robot at home.","The robot can sweep the floor and sing songs.","It helps my grandma carry light things.","We teach it to say hello in English.","Technology can be a good friend to people."],
    zh:["叔叔在家里做了一个小机器人。","机器人会扫地，还会唱歌。","它帮奶奶拿轻一点的东西。","我们教它用英语说你好。","科技也能成为人们的好朋友。"] },
  { title:"A Trip to the Zoo", level:"中级", lines:["Last Sunday our class visits the zoo.","We see tall giraffes and funny monkeys.","The panda eats bamboo slowly and looks cute.","Our teacher tells us to protect animals.","I learn a lot and want to come again."],
    zh:["上周日我们班去动物园玩。","我们看到了高高的长颈鹿和调皮的猴子。","熊猫慢吞吞地吃着竹子，样子真可爱。","老师告诉我们都要爱护动物。","我学到了很多，还想再来一次。"] },
  { title:"My Dream", level:"中级", lines:["When I grow up I want to be a pilot.","I will fly to many countries and see the world.","I will learn English well to talk with friends everywhere.","I will also help people who need a ride home.","I believe my dream can come true with hard work."],
    zh:["长大后我想当一名飞行员。","我要飞去很多国家，看看这个世界。","我要学好英语，和各地的朋友聊天。","我也会帮助那些需要回家的人。","我相信只要努力，梦想就能实现。"] },
  { title:"The Lost Star", level:"高级", lines:["One night a little star falls from the sky.","It lands quietly in a small quiet village.","A boy finds it and keeps it in a glass jar.","The star gives a soft light that makes him calm.","He lets it go so it can shine for everyone again."],
    zh:["一天夜里，一颗小星星从天上掉了下来。","它悄悄落在一个安静的小村庄里。","一个男孩发现了它，把它放进玻璃罐里。","星星发出柔和的光，让他感到平静。","他放它飞走，让它重新为所有人发光。"] },
  { title:"The Brave Little Boat", level:"高级", lines:["A small wooden boat sails far across the sea.","It meets big waves but never gives up.","A friendly dolphin shows the way home.","At last it reaches a warm and sunny shore.","The little boat learns that courage brings safe harbor."],
    zh:["一只小木船远远地航行在大海上。","它遇到巨大的海浪，却从不放弃。","一只友善的海豚为它指路回家。","终于，它抵达了温暖而阳光明媚的岸边。","小船明白了：勇气能带来安全的港湾。"] }
];
try {
  const ext = require('./passages_safe.json');
  _pgSafe = (ext && ext.length && ext.some(p => p && p.zh)) ? ext : BUILTIN_PASSAGES;
} catch (e) { _pgSafe = BUILTIN_PASSAGES; }

app.get('/free-dialog', (req, res) => {
  const offset = parseInt(req.query.offset) || 0;
  const pool = shuffle(_ddSafe || []);
  const pick = pool.slice(offset % Math.max(pool.length, 1), (offset % Math.max(pool.length, 1)) + 6);
  res.json({ ok: true, items: pick.length ? pick : (pool.slice(0, 6)) });
});

app.get('/free-passage', (req, res) => {
  const offset = parseInt(req.query.offset) || 0;
  const wantLevel = String(req.query.level || '').trim();
  let pool = shuffle(_pgSafe || []);
  // 按难度过滤（前端传 入门/初级/中级/高级；为空或命中不到则回退全部）
  if (wantLevel) {
    const matched = pool.filter(p => (p.level || '').includes(wantLevel));
    if (matched.length) pool = matched;
  }
  const pick = pool.slice(offset % Math.max(pool.length, 1), (offset % Math.max(pool.length, 1)) + 4);
  res.json({ ok: true, items: pick.length ? pick : (pool.slice(0, 4)) });
});

/* ---------- 路由 ---------- */
/* 诊断接口：返回 API Key 脱敏元信息，不暴露完整 Key，用于排查「Incorrect API key」 */
app.get('/diag', (req, res) => {
  const raw = process.env.API_KEY || process.env.DASHSCOPE_API_KEY || '';
  const k = raw.trim();
  const effSource = process.env.API_KEY ? 'API_KEY'
    : process.env.DASHSCOPE_API_KEY ? 'DASHSCOPE_API_KEY' : 'none';
  res.json({
    hasApiKey: !!k,
    rawKeyLen: raw.length,                 // 原始（可能含 \r\n 等不可见字符）
    keyLen: k.length,                      // 清理首尾空白后
    hasInvisibleChars: raw.length !== k.length || /[\r\n\t]/.test(raw),
    keyPrefix: k.slice(0, 6), keySuffix: k.slice(-4),
    looksLikeDashscope: k.startsWith('sk-'),
    effectiveSource: effSource,
    apiKeySet: !!process.env.API_KEY,
    dashscopeKeySet: !!process.env.DASHSCOPE_API_KEY,
    visionProvider: VISION_PROVIDER, visionModel: VISION_MODEL, asrModel: ASR_MODEL
  });
});

app.post('/ocr', upload.single('file'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: '未收到图片' });
  const id = newTask();
  const buf = req.file.buffer, mime = req.file.mimetype;
  (async () => {
    try {
      let words, source;
      if (API_KEY && VISION_PROVIDER !== 'none') {
        try {
          words = await visionOCR(buf, mime);
          source = 'vision:' + VISION_PROVIDER;
        } catch (e) {
          // 云端优先走视觉模型；失败直接报错，不再 fallback 本地 tesseract（Render 上 wasm 会崩溃）
          finishTask(id, { status: 'error', error: '云端视觉识别失败：' + (e && e.message || e) + '（请检查 API Key 是否有效、模型 ' + VISION_MODEL + ' 是否已开通）' });
          return;
        }
      } else {
        try {
          words = await localOCR(buf, mime);
          source = 'local-tesseract(no-key)';
        } catch (e) {
          finishTask(id, { status: 'error', error: e.message || String(e) });
          return;
        }
      }
      finishTask(id, { status: 'done', words: words || [], source });
    } catch (e) { finishTask(id, { status: 'error', error: e.message || String(e) }); }
  })();
  res.status(202).json({ taskId: id, status: 'processing' });
});

app.get('/ocr/status/:id', (req, res) => {
  const t = tasks.get(req.params.id);
  if (!t) return res.status(404).json({ error: '任务不存在或已过期，请重试' });
  if (t.status === 'processing' && Date.now() - t.startedAt > 120000) {
    tasks.delete(req.params.id);
    return res.status(408).json({ error: '识别超时（模型响应过慢），请用「批量导入」粘贴 AI 文本，或稍后重试' });
  }
  consumeTask(req.params.id);
  res.json(t);
});

// 短文拍照识图：复用 /ocr 的异步上传管道，换用视觉短文提示词，返回 {title, level, paragraphs}
// 支持多张图片（同一篇短文分多张拍）：表单字段名 images，最多 6 张，按顺序合并成一篇完整短文
app.post('/ocr-passage', upload.array('images', 6), (req, res) => {
  const files = (req.files && req.files.length) ? req.files : (req.file ? [req.file] : []);
  if (!files.length) return res.status(400).json({ error: '未收到图片' });
  const id = newTask();
  const buffers = files.map(f => f.buffer), mimes = files.map(f => f.mimetype);
  (async () => {
    try {
      if (!(API_KEY && VISION_PROVIDER !== 'none')) {
        finishTask(id, { status: 'error', error: '未配置视觉识别 Key（VISION_PROVIDER 设为 none 或无 API_KEY），无法做拍照识图。请改用「我自己写」手动粘贴。' });
        return;
      }
      let passage, source;
      try {
        passage = await visionPassageOCR(buffers, mimes);
        source = 'vision:' + VISION_PROVIDER;
      } catch (e) {
        finishTask(id, { status: 'error', error: '云端视觉识别失败：' + (e && e.message || e) + '（可能是免费额度用尽，请稍后重试或改用「我自己写」）' });
        return;
      }
      finishTask(id, { status: 'done', passage: passage || { title: '', level: '', paragraphs: [] }, source: source });
    } catch (e) { finishTask(id, { status: 'error', error: e.message || String(e) }); }
  })();
  res.status(202).json({ taskId: id, status: 'processing' });
});

app.post('/asr', upload.single('file'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: '未收到音频' });
  if (!API_KEY || ASR_PROVIDER === 'none') {
    return res.status(400).json({ error: 'no-key', message: '未配置云端语音识别，请改用浏览器识别或在 .env 配置 ASR_PROVIDER=qwen 与 API_KEY。' });
  }
  const id = newTask();
  const buf = req.file.buffer, ext = (req.file.originalname.split('.').pop() || 'webm').toLowerCase();
  // 以随机 token 落地内存，构造公网 URL 供 dashscope 下载（transcription 仅支持公网 URL）
  const token = crypto.randomBytes(12).toString('hex');
  setAudio(token, buf, ext);
  const audioUrl = 'https://' + req.get('host') + '/audio/' + token;
  (async () => {
    try {
      const text = await asrQwen(audioUrl);
      finishTask(id, { status: 'done', text: text || '' });
    } catch (e) {
      finishTask(id, { status: 'error', error: e.message || String(e) });
    } finally {
      getAudio(token); // 无论成败都清理中转音频
    }
  })();
  res.status(202).json({ taskId: id, status: 'processing' });
});

// dashscope 下载中转音频（token 用后即焚，不可猜）
app.get('/audio/:token', (req, res) => {
  const a = getAudio(req.params.token);
  if (!a) return res.status(404).end();
  res.set('Content-Type', mimeFromExt(a.ext));
  res.send(Buffer.from(a.buf));
});

app.get('/asr/status/:id', (req, res) => {
  const t = tasks.get(req.params.id);
  if (!t) return res.status(404).json({ error: '任务不存在或已过期，请重试' });
  if (t.status === 'processing' && Date.now() - t.startedAt > 90000) {
    tasks.delete(req.params.id);
    return res.status(408).json({ error: '识别超时，请稍后重试或用「我已大声朗读」记录' });
  }
  consumeTask(req.params.id);
  res.json(t);
});

/* ---------- AI 口语陪练：文本对话模型 ----------
 * 新增长连接「和 AI 用英语聊天」：前端维护多轮 messages 发来，后端调文本对话模型接话。
 * 半引导 + 小学词为主偶尔初中词 + 美式 + 自信保护，prompt 由前端在 messages[0] 固化传入。
 *
 * 模型生命周期管理（重要）：阿里百炼会定期下线旧模型（如 qwen-turbo 已于 2026-10-10 下线）。
 * 因此后端内置 CHAT_MODELS 降级列表：按序尝试，遇到模型下线/额度耗尽/不可用等错误自动跳下一个，
 * 调用方无需改代码。新增/替换模型只改下面这个数组即可（顺序=优先级）。
 * 备注（2026-08 核实）：qwen-turbo 已确认 2026-10-10 下线 → 主力改 qwen3.6-flash（长期在架、免费、超快）。
 * 另：paraformer ASR 系列亦于 2026-10-10 下线，需迁 paraformer-realtime-v2 等，属 ASR 改动，不在本段范围。
 */
// 对话模型降级池（顺序=优先级）。可经环境变量 CHAT_MODEL 覆盖首选项；CHAT_MODELS 覆盖整个池。
// 2026-08 官方核实（含下线公告 notice 2009/118434/117861 + 官方计费页 2712341）：
//  - 免费额度：百炼新用户每模型 100 万 tokens（90 天有效，华北2/北京），各版本独立额度 → 可叠加
//  - 已下线：qwen3.5-flash（2026-01-30）、qwen-turbo 旧快照等
//  - 2026-10-10 即将下线：qwen-turbo 全系、qwen3-max-preview、qwen3.6-max-preview、qwen3.6-max、
//    qwen3-max-2026-01-23 / qwen3-max-2025-09-23、qwq-plus、qwen-vl-max/plus 旧版、qwen-coder 系、paraformer 全系、
//    qwen-plus-0112、qwen-plus-1220、qwen-plus-2025-07-28、qwen-plus-2025-09-11、qwen-plus-2025-12-01-us（→ qwen3.7-plus）
//  - 重要：无日期版 qwen-plus 自 2026-01-20 起自动指向 qwen-plus-2025-12-01 快照（官方公告 117861）→ 安全
//  - 中国内地有 100 万免费额度、安全可叠加：qwen3.7-plus（最新最强中端，主力）、qwen3.7-flash（速度型）、
//    qwen-plus（自动=2025-12-01）、qwen-plus-2025-12-01（独立额度）、qwen3.6-plus、qwen3.6-flash（保底）
//  - qwen3.7-max 中国内地 API 免费额度未明示（网页端免费≠API）→ 不选，避免按量扣费
const CHAT_MODELS = (process.env.CHAT_MODELS && process.env.CHAT_MODELS.split(',').map(s=>s.trim()).filter(Boolean))
  || [ process.env.CHAT_MODEL || 'qwen3.7-plus',
       'qwen3.7-plus', 'qwen-plus', 'qwen-plus-2025-12-01',
       'qwen3.6-plus', 'qwen3.7-flash', 'qwen3.6-flash' ]
  .filter((v,i,a)=>a.indexOf(v)===i); // 去重，保留顺序

// 模型生命周期（防"用着用着模型下线"）：status=stable 稳定 / sunset 近期将下线 / retired 已下线。
// 依据 2026-08 官方公告核实（见上面 CHAT_MODELS 注释）：
// 规则：retired/sunset 一律不进实际调用池（"会下线的一律不用"）；前端模型池据此显示状态标签。
const MODEL_LIFECYCLE = {
  'qwen3.5-flash':           { status: 'retired', note: '已下线（2026-01-30）' },
  'qwen-turbo':              { status: 'retired', note: '已下线（2026-10-10）' },
  'qwen3-max-preview':       { status: 'sunset',  note: '即将下线（2026-10-10），用 qwen-max' },
  'qwen3.6-max-preview':     { status: 'sunset',  note: '即将下线（2026-10-10），用 qwen-max' },
  'qwen3-max-2026-01-23':    { status: 'sunset',  note: '即将下线（2026-10-10）' },
  'qwen3-max-2025-09-23':    { status: 'sunset',  note: '即将下线（2026-10-10）' },
  'qwen-coder-turbo':        { status: 'sunset',  note: '即将下线（2026-10-10）' },
  'qwen-coder-plus':         { status: 'sunset',  note: '即将下线（2026-10-10）' },
  'qwq-plus':                { status: 'sunset',  note: '即将下线（2026-10-10）' },
  'qwen-vl-max':             { status: 'sunset',  note: '即将下线（2026-10-10），用 qwen3-vl-plus' },
  'qwen-vl-plus':            { status: 'sunset',  note: '即将下线（2026-10-10），用 qwen3-vl-plus' },
  'paraformer-v2':           { status: 'sunset',  note: '即将下线（2026-10-10），建议迁 paraformer-realtime-v2' },
  'qwen3.6-max':             { status: 'sunset',  note: '即将下线（2026-10-10）' },
  'qwen3.6-max-preview':     { status: 'sunset',  note: '即将下线（2026-10-10）' },
  'qwen3.7-plus':            { status: 'stable',  note: '主力·最新中端·100万免费' },
  'qwen3.7-flash':           { status: 'stable',  note: '速度型·100万免费' },
  'qwen3.6-plus':            { status: 'stable',  note: '中端·100万免费' },
  'qwen-plus':               { status: 'stable',  note: '自动指向最新快照(2025-12-01)' },
  'qwen-plus-2025-12-01':    { status: 'stable',  note: '快照·当前版本' },
  'qwen-plus-0112':          { status: 'sunset',  note: '即将下线（2026-10-10）→ qwen-plus(自动新版)' },
  'qwen-plus-2025-07-28':    { status: 'sunset',  note: '即将下线（2026-10-10）→ qwen-plus(自动新版)' },
  'qwen-plus-2025-09-11':    { status: 'sunset',  note: '即将下线（2026-10-10）' },
  'qwen-plus-1220':          { status: 'sunset',  note: '即将下线（2026-10-10）' },
  'qwen-plus-2025-12-01-us': { status: 'sunset',  note: '即将下线（2026-10-10）' },
  'qwen-flash':              { status: 'stable',  note: '速度型（备用）' },
  'qwen-flash-2025-07-28':   { status: 'stable',  note: '' },
  'qwen-flash-2026-01-25':   { status: 'stable',  note: '' },
  'qwen-long':               { status: 'stable',  note: '长上下文 256K' },
  'qwen-max':                { status: 'stable',  note: '旗舰·额度小慎用' },
  'qwen3.6-flash':           { status: 'stable',  note: '保底' },
  'paraformer-realtime-v2':  { status: 'stable',  note: '实时识别主力' },
  'qwen3-vl-plus':           { status: 'stable',  note: '视觉识别' },
  'qwen-audio-3.0-tts-flash':{ status: 'stable',  note: '备用音源' }
};
// 实际调用池：过滤掉已下线/即将下线的模型（保底：若全被过滤则退回原池，避免空池）
const CHAT_POOL = (CHAT_MODELS.filter(m => {
  const lc = MODEL_LIFECYCLE[m] || {};
  return lc.status !== 'retired' && lc.status !== 'sunset';
})).concat([]);
const ACTIVE_CHAT_MODELS = CHAT_POOL.length ? CHAT_POOL : CHAT_MODELS;

// 判断某模型错误是否属于「该模型不可用，应跳过尝试下一个」
function isModelFatal(code, msg){
  const m = String(msg||'').toLowerCase();
  const c = String(code||'').toLowerCase();
  // 模型下线 / 不存在 / 额度耗尽 / 权限不足 → 跳下一个
  return /model.*(not exist|decommission|retir|deprecat|not found)|invalid.*model|model_not_found| Model.*not.*support/i.test(m)
      || /quota|rate.?limit|exceed|throttl|account.*not.*open|unsupported|not.*allow/i.test(m)
      || c.includes('model') || c.includes('quota') || c.includes('throttl');
}

async function chatQwen(messages, tried) {
  tried = tried || [];
  // 从池里挑第一个还没试过的模型（池已过滤掉已下线/即将下线模型）
  const model = ACTIVE_CHAT_MODELS.find(m => !tried.includes(m));
  if (!model) throw new Error('所有对话模型均不可用（可能全部下线或额度耗尽），请检查 CHAT_MODELS 配置或 DashScope 账户。');
  const url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions';
  const body = {
    model: model,
    messages: messages,
    temperature: 0.7,
    max_tokens: 2000, // 需覆盖短文/对话/例句生成（高级短文 11-15 句+逐句中文翻译约 1.5k tokens），私教聊天一般远小于此
    stream: true, // 流式输出：首 token 极快，前端边收边朗读，消除"整段等"
    // 关键：qwen3.6-flash 默认走 thinking，首 token 慢；必须用【顶层】 enable_thinking:false 关闭。
    // 注意不能用 extra_body（那是 OpenAI Python SDK 的写法；原生 fetch 直接发 JSON 时 extra_body 会被服务端忽略，
    // 导致 thinking 从未关闭、每次回复都慢好几秒——2026-08-25 实测修正）。
    enable_thinking: false
  };
  let r;
  try {
    r = await fetch(url, {
      method: 'POST',
      signal: AbortSignal.timeout(30000),
      headers: { 'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
  } catch (netErr) {
    // 网络层失败也尝试下一个模型
    tried.push(model);
    return chatQwen(messages, tried);
  }
  const ct = r.headers.get('content-type') || '';
  // 流式：server-sent-events；非流式（个别模型不支持 stream）：普通 JSON
  if (ct.includes('text/event-stream') || ct.includes('stream')) {
    const out = await consumeStream(r, model);
    if (!out.sawChunk) {
      // 流式但一点内容都没收到，视为该模型异常 → 试下一个
      tried.push(model);
      return chatQwen(messages, tried);
    }
    return out;
  }
  const j = await r.json().catch(() => ({}));
  if (j.error) {
    const code = j.error.code || (j.error.type) || '';
    const msg = j.error.message || JSON.stringify(j.error);
    if (isModelFatal(code, msg)) {
      // 该模型不可用 → 试下一个
      tried.push(model);
      return chatQwen(messages, tried);
    }
    // 非模型问题（如请求格式错误）→ 直接抛，不再重试
    throw new Error(`云端对话失败[${code}]：${msg}`);
  }
  const content = (j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content) || '';
  return { text: content.trim(), model: model, usage: j.usage || null };
}

// 解析流式 SSE 响应，逐块累积文本；出错或缺块时返回已累积内容（保证可用性优于彻底失败）
async function consumeStream(r, model){
  const reader = r.body.getReader();
  const dec = new TextDecoder('utf-8');
  let buf = '', text = '', sawChunk = false;
  let usage = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    // SSE 以 "\n\n" 分段，每段 "data: {...}"
    let idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
      const lines = raw.split('\n').map(s => s.trim()).filter(Boolean);
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (data === '[DONE]') continue;
        try {
          const p = JSON.parse(data);
          const d = p.choices && p.choices[ 0 ] && p.choices[ 0 ].delta && p.choices[ 0 ].delta.content;
          if (d) { text += d; sawChunk = true; }
          // 计费用量：部分模型在末尾 data 行带 usage（也有放在 choices[0].usage）
          if (p.usage) usage = p.usage;
          else if (p.choices && p.choices[0] && p.choices[0].usage) usage = p.choices[0].usage;
        } catch (e) { /* 跳过无法解析的探测行 */ }
      }
    }
  }
  return { text: text.trim(), model, sawChunk, usage };
}

/* ---------- 整句 TTS：复用 DashScope key，合成整句英文（有道 dictvoice 不支持整句）---------- */
app.get('/tts', async (req, res) => {
  if (!API_KEY) return res.status(400).json({ error: 'no-key', message: '未配置 API_KEY，无法使用云端整句发音（请在 .env 配置 DashScope key，并已开通“语音合成”）' });
  const text = String(req.query.text || '').slice(0, 500).trim();
  if (!text) return res.status(400).json({ error: 'empty', message: '文本为空' });
  const model = process.env.TTS_MODEL || 'qwen-audio-3.0-tts-flash';
  const voice = process.env.TTS_VOICE || 'longanhuan_v3.6';
  const host = process.env.TTS_HOST || 'https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer';
  try {
    const r = await fetch(host, {
      method: 'POST',
      signal: AbortSignal.timeout(30000),
      headers: { 'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, input: { text }, parameters: { voice, format: 'wav', sample_rate: 24000 } })
    });
    const ct = r.headers.get('content-type') || '';
    if (ct.includes('audio')) {
      const buf = Buffer.from(await r.arrayBuffer());
      res.set('Content-Type', ct.split(';')[0]);
      try { recordUsage(model, null, false); } catch (_) {}                 // 模型池用量统计
      return res.send(buf);
    }
    // 非流式可能返回 JSON（含 audio_url）或错误
    const j = await r.json().catch(() => ({}));
    if (j.code || j.error) {
      const m = (j.error && (j.error.message || JSON.stringify(j.error))) || j.message || JSON.stringify(j);
      throw new Error('云端 TTS 失败：' + m);
    }
    const audioObj = (j.output && j.output.audio) || {};
    const audioUrl = audioObj.url || audioObj.data || j.audio_url || (j.output && j.output.audio_url);
    if (audioUrl) {
      if (/^data:/.test(audioUrl)) {
        const b64 = audioUrl.split(',')[1] || '';
        const buf = Buffer.from(b64, 'base64');
        res.set('Content-Type', 'audio/wav');
        try { recordUsage(model, null, false); } catch (_) {}               // 模型池用量统计
        return res.send(buf);
      }
      const ar = await fetch(audioUrl);
      const buf = Buffer.from(await ar.arrayBuffer());
      res.set('Content-Type', ar.headers.get('content-type') || 'audio/wav');
      try { recordUsage(model, null, false); } catch (_) {}                 // 模型池用量统计
      return res.send(buf);
    }
    throw new Error('云端 TTS 未返回音频：' + JSON.stringify(j).slice(0, 200));
  } catch (e) {
    res.status(502).json({ error: 'tts-failed', message: e.message || String(e) });
  }
});

/* ---------- 免费 Edge TTS（零 key · 纯正美式 neural 音色 · 各设备一致） ----------
 * 复用 Microsoft Edge 在线 TTS（与 Edge 浏览器“大声朗读”同源），无需任何 API Key、零成本。
 * 通过 WebSocket 直连 speech.platform.bing.com，带 Sec-MS-GEC 时间戳签名（版本跟随当前 Edge）。
 * 前端“整句朗读/听”优先走此端点；华为/安卓本机 speechSynthesis 常静音，此方案用 <audio> 播放 MP3，全设备可用。
 * 注意：TRUSTED_CLIENT_TOKEN 与 GEC 算法均来自公开 edge-tts 项目，仅作免费语音合成用途。
 */
const EDGE_TRUSTED = '6A5AA1D4EAFF4E9FB37E23D68491D6F4';
const EDGE_GEC_VERSION = '1-143.0.3650.75';
const EDGE_CHROME = '143';

function edgeSecMsGec() {
  let ticks = Math.floor(Date.now() / 1000);
  ticks += 11644473600;            // Windows 纪元(1601)偏移
  ticks -= ticks % 300;            // 向下取整到 5 分钟窗口（与服务端对齐）
  ticks *= 10000000;               // 100 纳秒间隔
  return crypto.createHash('sha256').update(String(ticks) + EDGE_TRUSTED).digest('hex').toUpperCase();
}
function edgeDateStr() {
  const d = new Date();
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const mos = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const p = n => String(n).padStart(2, '0');
  return `${days[d.getUTCDay()]} ${mos[d.getUTCMonth()]} ${p(d.getUTCDate())} ${d.getUTCFullYear()} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())} GMT+0000 (Coordinated Universal Time)`;
}
const _edgeVoiceRe = /^[a-z]{2}-[A-Z]{2}-[A-Za-z]+(Neural|Class)$/;

function edgeTTS(text, voice, rate) {
  rate = rate || '-15%';   // 默认放慢 15%：孩子听英语需要更慢的语速（可通过 ?rate= 覆盖）
  return new Promise((resolve, reject) => {
    const uid = () => crypto.randomUUID().replace(/-/g, '');
    const safe = String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const url = `wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1?TrustedClientToken=${EDGE_TRUSTED}&ConnectionId=${uid()}&Sec-MS-GEC=${edgeSecMsGec()}&Sec-MS-GEC-Version=${EDGE_GEC_VERSION}`;
    const ws = new WebSocket(url, {
      headers: {
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache',
        'Origin': 'chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold',
        'User-Agent': `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/${EDGE_CHROME}.0.0.0 Safari/537.36 Edg/${EDGE_CHROME}.0.0.0`,
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cookie': `muid=${crypto.randomBytes(16).toString('hex').toUpperCase()};`
      }
    });
    const audio = [];
    let settled = false; let _msgN = 0, _binN = 0;
    const done = (err, buf) => {
      if (settled) return; settled = true;
      console.error('[EDGE] done err=' + (err && err.message) + ' audioBytes=' + (buf ? buf.length : 0) + ' msgs=' + _msgN + ' bins=' + _binN);
      clearTimeout(timer);
      try { ws.close(); } catch (_) {}
      if (err) reject(err); else resolve(buf);
    };
    const timer = setTimeout(() => done(new Error('Edge TTS 超时')), 30000);
    ws.on('error', e => done(e));
    ws.on('message', (raw, isBinary) => {
      if (!isBinary) {
        _msgN++;
        const s = raw.toString('utf8');
        if (s.includes('turn.end')) done(null, Buffer.concat(audio));
        return;
      }
      _binN++;
      const sep = 'Path:audio\r\n';
      const i = raw.indexOf(sep);
      if (i >= 0) audio.push(raw.subarray(i + sep.length));
      else console.error('[EDGE] bin frame without Path:audio, len=' + raw.length);
    });
    ws.on('open', () => {
      const cfg = `X-Timestamp:${edgeDateStr()}\r\nContent-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n{"context":{"synthesis":{"audio":{"metadataoptions":{"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"false"},"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}\r\n`;
      ws.send(cfg, { compress: true }, (e) => {
        if (e) return done(e);
        const ssml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><voice name='${voice}'><prosody pitch='+0Hz' rate='${rate}' volume='+0%'>${safe}</prosody></voice></speak>`;
        const msg = `X-RequestId:${uid()}\r\nContent-Type:application/ssml+xml\r\nX-Timestamp:${edgeDateStr()}Z\r\nPath:ssml\r\n\r\n${ssml}`;
        ws.send(msg, { compress: true }, (e2) => { if (e2) done(e2); });
      });
    });
  });
}

/* 免费 Edge TTS 端点（零 key）。前端整句朗读/听优先调用此端点。 */
const _edgeRate = new Map(); // ip -> [timestamps]
function edgeRateOk(ip) {
  const now = Date.now();
  const arr = (_edgeRate.get(ip) || []).filter(t => now - t < 60000);
  if (arr.length >= 60) { _edgeRate.set(ip, arr); return false; }
  arr.push(now); _edgeRate.set(ip, arr); return true;
}
app.get('/tts-edge', async (req, res) => {
  if (!WS_AVAILABLE) return res.status(503).json({ error: 'edge-tts-unavailable', message: '后端未加载 ws 模块，无法使用 Edge TTS（请执行 npm install ws 后重新部署）' });
  const ip = String(req.headers['x-forwarded-for'] || req.socket.remoteAddress || '');
  if (!edgeRateOk(ip)) return res.status(429).json({ error: 'rate-limited', message: '请求过于频繁，请稍后再试' });
  const text = String(req.query.text || '').slice(0, 500).trim();
  if (!text) return res.status(400).json({ error: 'empty', message: '文本为空' });
  let voice = String(req.query.voice || 'en-US-AriaNeural').slice(0, 40).trim();
  if (!_edgeVoiceRe.test(voice)) voice = 'en-US-AriaNeural';
  let rate = String(req.query.rate || '-15%').slice(0, 8).trim();
  if (!/^-?\d{1,2}%$/.test(rate)) rate = '-15%';   // 儿童学习默认放慢 15%
  try {
    const buf = await edgeTTS(text, voice, rate);
    if (!buf || !buf.length) return res.status(502).json({ error: 'edge-empty', message: 'Edge TTS 未返回音频' });
    res.set('Content-Type', 'audio/mpeg');
    res.set('Cache-Control', 'public, max-age=86400');
    return res.send(buf);
  } catch (e) {
    res.status(502).json({ error: 'edge-failed', message: e.message || String(e) });
  }
});

/* ---------- 模型池用量统计（叠加免费额度看板） ----------
 * 百炼免费额度按模型各自发放，后端薄代理按 CHAT_MODELS 顺序自动降级。
 * 这里累计每个模型当月已用 tokens，并落盘 usage.json（Render 重启会重置到部署快照，但至少展示实时累计）。
 * 同时记录每轮练习计数，供「预计还能练几天」估算。
 */
const USAGE_FILE = path.join(__dirname, 'usage.json');
let usageCache = null; // { monthKey, perModel:{ [model]:{prompt, completion, total, calls} }, rounds }
function loadUsage(){
  if (usageCache) return usageCache;
  try { usageCache = JSON.parse(fs.readFileSync(USAGE_FILE, 'utf-8')); }
  catch (_) { usageCache = null; }
  const mk = new Date().toISOString().slice(0, 7); // YYYY-MM
  if (!usageCache || usageCache.monthKey !== mk) {
    usageCache = { monthKey: mk, perModel: {}, rounds: 0 };
  }
  return usageCache;
}
function saveUsage(){
  try { fs.writeFileSync(USAGE_FILE, JSON.stringify(usageCache), 'utf-8'); } catch (_) {}
}
function recordUsage(model, usage, rounded){
  const u = loadUsage();
  const m = u.perModel[model] || { prompt: 0, completion: 0, total: 0, calls: 0 };
  const p = (usage && (usage.prompt_tokens || usage.input_tokens)) || 0;
  const c = (usage && (usage.completion_tokens || usage.output_tokens)) || 0;
  const t = (usage && usage.total_tokens) || (p + c);
  m.prompt += Number(p) || 0;
  m.completion += Number(c) || 0;
  m.total += Number(t) || 0;
  m.calls += 1;
  u.perModel[model] = m;
  if (rounded) u.rounds += 1; // 仅在有真实对话轮次时计数
  saveUsage();
}

/* ---------- AI 口语陪练：对话端点 ---------- */
app.post('/chat', express.json({ limit: '60kb' }), async (req, res) => {
  // 无 key 时直接告知，避免前端静默失败
  if (!API_KEY) return res.status(400).json({ error: 'no-key', message: '未配置 API_KEY，无法使用 AI 聊天（请在 .env 配置 DashScope key）' });
  // messages：前端传入完整多轮对话（含 system prompt 作 messages[0]），后端只透传给模型。
  // 这样 prompt 的迭代/调参完全在前端控制，后端保持薄代理。
  const messages = Array.isArray(req.body && req.body.messages) ? req.body.messages : [];
  if (!messages.length) return res.status(400).json({ error: 'empty', message: 'messages 为空' });
  // 轻量把关：限制角色与条数，避免异常输入
  const clean = messages.slice(-40).map(m => ({
    role: (m.role === 'system' || m.role === 'user' || m.role === 'assistant') ? m.role : 'user',
    content: String(m.content || '').slice(0, 2000)
  })).filter(m => m.content);
  if (!clean.length) return res.status(400).json({ error: 'empty', message: '有效消息为空' });
  try {
    const out = await chatQwen(clean);
    if (!out || !out.text) return res.status(502).json({ error: 'empty-reply', message: 'AI 未返回内容，请重试' });
    try { recordUsage(out.model, out.usage, true); } catch (_) {}
    res.json({ ok: true,  reply: out.text, model: out.model });
  } catch (e) {
    res.status(502).json({ error: 'chat-failed', message: e.message || String(e) });
  }
});

/* 用量看板：各模型本月已用 tokens、状态、预计还能练几天 */
app.get('/chat-usage', (req, res) => {
  const u = loadUsage();
  // 单模型免费额度保守估算（百炼 flash 系列通常每月数百万 tokens，这里取保守 1,000,000；
  // 注意：免费额度的总量/剩余只能登录百炼控制台查看，无公开 API，此处为本地估算）
  const FREE_PER_MODEL = Number(process.env.CHAT_FREE_QUOTA) || 1000000;
  const dayOfMonth = new Date().getDate();
  const daysLeft = Math.max(1, new Date(new Date().getFullYear(), new Date().getMonth() + 1, 0).getDate() - dayOfMonth + 1);
  const models = ACTIVE_CHAT_MODELS.map(m => {
    const s = u.perModel[m] || { prompt: 0, completion: 0, total: 0, calls: 0 };
    const used = s.total || 0;
    const remain = Math.max(0, FREE_PER_MODEL - used);
    // 按近况估算每天消耗：若有数据，用（已用/已过天数）做保守线性外推；否则给兜底预估
    let perDay = 0;
    if (s.total > 0 && dayOfMonth > 0) perDay = s.total / dayOfMonth;
    const canDays = perDay > 0 ? Math.floor(remain / perDay) : 9999;
    return {
      model: m,
      used: used,
      remain: remain,
      calls: s.calls || 0,
      status: used >= FREE_PER_MODEL ? 'exhausted' : 'ok',
      canDays: canDays > 365 ? '>1年' : (canDays + ' 天')
    };
  });
  // 全模型清单：工作台用到的每个模型（用途/模型/提供商/免费说明/本月调用/生命周期），供「设置→模型池」展示
  const per = u.perModel || {};
  const lcOf = m => {
    const lc = MODEL_LIFECYCLE[m] || {};
    return { status: lc.status || 'stable', note: lc.note || '' };
  };
  const mk = (purpose, model, provider, note, callsKey, extra) => ({
    purpose, model, provider, free: true, note, lifecycle: lcOf(model),
    calls: (callsKey ? ((per[callsKey] || {}).calls || 0) : 0),
    used: (callsKey ? ((per[callsKey] || {}).total || 0) : 0),
    ...(extra || {})
  });
  const all = [
    mk('拍照识别（OCR 视觉）', VISION_PROVIDER === 'doubao' ? (ARK_ENDPOINT || 'doubao') : VISION_MODEL,
       VISION_PROVIDER === 'doubao' ? '火山方舟' : '阿里云百炼', '百炼免费额度内（以控制台为准）', VISION_MODEL),
    mk('语音识别（批量录音）', ASR_MODEL, '阿里云百炼', '批量兜底路径；即将下线则建议改用实时识别', ASR_MODEL),
    mk('语音识别（实时边说边出字）', 'paraformer-realtime-v2', '阿里云百炼', '实时模型免费额度内（以控制台为准）', null),
    ...ACTIVE_CHAT_MODELS.map(m => {
      const s = u.perModel[m] || {};
      const used = s.total || 0;
      const remain = Math.max(0, FREE_PER_MODEL - used);
      let perDay = 0;
      if (s.total > 0 && dayOfMonth > 0) perDay = s.total / dayOfMonth;
      const canDays = perDay > 0 ? Math.floor(remain / perDay) : 9999;
      return mk('AI 对话陪练（自动降级池）', m, '阿里云百炼', 'flash 系列免费额度，自动切换（以控制台为准）', m, {
        remain, status: used >= FREE_PER_MODEL ? 'exhausted' : 'ok',
        canDays: canDays > 365 ? '>1年' : (canDays + ' 天')
      });
    }),
    mk('语音合成（备用音源）', process.env.TTS_MODEL || 'qwen-audio-3.0-tts-flash', '阿里云百炼', '备用；主用免费 Edge，几乎不消耗', process.env.TTS_MODEL || 'qwen-audio-3.0-tts-flash'),
    mk('语音合成（主用·纯正美音）', 'Edge TTS', '微软免费接口', '完全免费，无额度限制', null)
  ];
  res.json({
    month: u.monthKey,
    rounds: u.rounds || 0,
    freePerModel: FREE_PER_MODEL,
    models: models,
    all: all,
    consoleUrl: 'https://bailian.console.aliyun.com/',
    note: '免费额度的真实总量/剩余只能在百炼控制台查看（无公开 API），表中「剩余/还能练」为本地累计+保守估算；已下线或即将下线的模型已从调用池剔除。'
  });
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
    <p>接口：<code>POST /ocr</code>（照片→4字段）　<code>POST /asr</code>（录音→文本）　<code>GET /tts-edge</code>（免费 Edge 整句发音）</p>
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

/* ---------- 实时语音识别 WebSocket 代理（paraformer-realtime-v2） ----------
 * 浏览器无法直接带 Authorization 头连 DashScope，且 API Key 不能进前端，
 * 故本服务在 /asr-realtime 做透明转发：前端连本服务，本服务带 Bearer 头连 DashScope。
 * 前端负责组装协议（run-task / 二进制 PCM / finish-task），本服务只透传字节。
 */
const server = http.createServer(app);
if (WS_AVAILABLE) {
  // 每条客户端连接对应一条“浏览器→本服务→DashScope”的透传隧道。
  // 不用“启动预热连接池”：空闲连接易被 DashScope 静默关闭而 Node 未察觉（readyState 仍 OPEN），
  // 导致 run-task 发到死连接、永远收不到 task-started。每次新建上游连接虽多一次握手，
  // 但与“浏览器→后端”握手并行，实际延迟可接受且稳定可靠。
  const wss = new WebSocketServer({ server, path: '/asr-realtime' });
  wss.on('connection', (client) => {
    console.log('[RT] 客户端已连接（workspace=' + RT_WORKSPACE + '）');
    let upstream = null, upOpen = false, pending = [];

    function openUpstream(){
      console.log('[RT] 正在连接上游: ' + RT_WS_URL + '  workspace=' + RT_WORKSPACE);
      upstream = new WebSocket(RT_WS_URL, {
        headers: { Authorization: 'Bearer ' + API_KEY, 'user-agent': 'jiabei-backend/1.0', 'X-DashScope-WorkSpace': RT_WORKSPACE }
      });
      let gotEvent = false, watchdog = null;
      upstream.on('open', () => {
        upOpen = true;
        console.log('[RT] 上游(DashScope)已打开（握手成功）');
        while (pending.length) { try { upstream.send(pending.shift()); } catch (_) {} }
      });
      upstream.on('message', (d) => {
        gotEvent = true;
        if (watchdog) { clearTimeout(watchdog); watchdog = null; }
        if (client.readyState !== WebSocket.OPEN) return;
        // DashScope 控制消息可能是 binary frame（内容仍是 JSON），必须转成字符串再转发，
        // 否则浏览器按 Blob 收到后前端 JSON.parse 失败，task-started 被丢弃 → 实时识别永远连不上。
        const isBuf = Buffer.isBuffer(d);
        const text = isBuf ? d.toString('utf8') : (typeof d === 'string' ? d : (d && d.toString ? d.toString() : ''));
        let evName = null; try { const j = JSON.parse(text); evName = j && j.header && j.header.event; } catch (_) {}
        if (evName) console.log('[RT] 收到上游事件: ' + evName);
        else console.log('[RT] 上游原始消息[' + (isBuf ? 'binary' : 'text') + ']: ' + text.slice(0, 300));
        try { client.send(text); } catch (_) {}
      });
      upstream.on('close', (code, reason) => {
        if (watchdog) { clearTimeout(watchdog); watchdog = null; }
        // code 是关键诊断：1000+空 reason=正常关闭；1008/1011=策略/服务错误；
        // 若发 run-task 后一直无事件、最后被关，多半是模型/workspace 未开通实时识别。
        console.log('[RT] 上游已关闭 code=' + code + ' reason=' + (reason && reason.toString ? reason.toString() : '(空)'));
      });
      upstream.on('error', (e) => { console.error('[RT] 上游(DashScope)错误:', e && e.message || e); });
      upstream.on('unexpected-response', (req, res) => {
        console.error('[RT] 上游非预期响应 status=' + (res && res.statusCode) + ' —— 握手未成功，可能不是有效的实时识别地址（检查 RT_WS_URL）');
      });
      // 看门狗：run-task 转发后若 N 秒无任一上游事件，判定 DashScope 未处理该任务（握手成功但不干活）
      upstream._watchdogArm = (secs) => {
        if (watchdog) clearTimeout(watchdog);
        watchdog = setTimeout(() => {
          if (!gotEvent) console.warn('[RT][WARN] 已转发 run-task 但 ' + secs + ' 秒内未收到任何上游事件 —— DashScope 很可能未处理该任务（检查：①workspace 是否开通 paraformer-realtime-v2 实时识别；②run-task 字段；③RT_WS_URL 地址是否正确）');
        }, secs * 1000);
      };
    }
    openUpstream();

    client.on('message', (data) => {
      // 判定是否为 JSON 控制消息（run-task / finish-task）。
      // 浏览器/反向代理可能把 JSON 控制消息以 binary 帧发来（ws.send(string) 经代理后常被改写为 binary 帧），
      // 而 DashScope 要求 run-task/finish-task 必须是“文本(JSON)帧”，收到 binary 帧会静默忽略
      // （表现为：握手成功、run-task 已送达，但 0 事件、最后优雅关闭）。故此处把控制消息归一化为文本帧转发。
      let str = null;
      if (typeof data === 'string') str = data;
      else if (Buffer.isBuffer(data)) str = data.toString('utf8');
      else if (typeof ArrayBuffer !== 'undefined' && data instanceof ArrayBuffer) str = new TextDecoder('utf-8').decode(data);
      let isControl = false, action = null;
      if (str) { try { const j = JSON.parse(str); if (j && j.header) { isControl = true; action = j.header.action; } } catch (_) {} }
      if (action === 'run-task') {
        console.log('[RT] 收到客户端 run-task，转发到上游（帧: ' + (typeof data === 'string' ? 'text' : 'binary→已转文本') + '）');
        console.log('[RT] run-task 内容: ' + (str ? str.slice(0, 500) : '[空]'));
        if (upstream && upstream._watchdogArm) upstream._watchdogArm(6);
      }
      if (isControl) {
        // 关键修复：控制消息一律以文本帧转发给 DashScope
        const out = (str != null) ? str : (typeof data === 'string' ? data : (Buffer.isBuffer(data) ? data.toString('utf8') : ''));
        if (upOpen && upstream && upstream.readyState === WebSocket.OPEN) try { upstream.send(out); } catch (_) {}
        else pending.push(out);
      } else {
        // 二进制音频帧（Int16 PCM）：原样转发，保持字节不变
        if (upOpen && upstream && upstream.readyState === WebSocket.OPEN) try { upstream.send(data); } catch (_) {}
        else pending.push(data);
      }
    });
    client.on('close', () => { try { if (upstream) upstream.close(); } catch (_) {} });
    client.on('error', () => {});
  });
} else {
  console.warn('[RT] /asr-realtime 未启用（ws 模块缺失）。');
}

if (require.main === module) {
  server.listen(PORT, () => {
    console.log('[加贝后端] 已启动 http://localhost:' + PORT + '  key=' + (API_KEY ? '已配置' : '未配置') + ' vision=' + VISION_PROVIDER + ' asr=' + ASR_PROVIDER + ' realtime-ws=/asr-realtime');
  });
}

module.exports = { extractWords, parseOcrText, app };
