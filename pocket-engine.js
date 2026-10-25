/* PDF-Vorleser – Neue Stimmen (Kyutai Pocket TTS, deutsch), komplett auf dem Gerät.
   Ablauf pro Satz: Text -> Tokens -> Sprachmodell (Schritt für Schritt, mit Stimm-Gedächtnis)
   -> Flow-Kopf erzeugt Klang-Latents -> Mimi-Decoder macht daraus Audio (24 kHz).
   Modell: kyutai/pocket-tts (CC BY 4.0), für den Browser umgewandelt. */
'use strict';
const PK = { ready: null, meta: null, step: null, head: null, mimi: null, lut: null, vocab: null, maxLen: 0, unkScore: -20, voices: new Map() };
const PK_BASE = new URL('pocket/', self.location.href).href;

function pkEnsureOrt(){
  if(!self.ort){
    importScripts(new URL('engine/ort.wasm.min.js', self.location.href).href);
    self.ort.env.wasm.numThreads = 1;
    self.ort.env.wasm.proxy = false;
    self.ort.env.wasm.wasmPaths = new URL('engine/', self.location.href).href;
  }
}

let PK_F16 = null;
function pkF16(){
  if(PK_F16) return PK_F16;
  const t = new Float32Array(65536);
  for(let h = 0; h < 65536; h++){
    const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    t[h] = e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return (PK_F16 = t);
}
function pkHalfToFloat(u16, out, outOff){
  const T = pkF16();
  for(let i = 0; i < u16.length; i++) out[outOff + i] = T[u16[i]];
}
/* rebuild the original ONNX file: skeleton bytes with float16 weight spans expanded to float32 */
function pkRebuild(all, m){
  const skel = all.subarray(0, m.graphSize), packed = all.subarray(m.graphSize);
  const out = new Uint8Array(m.fileSize), T = pkF16();
  let sOff = 0, pOff = 0, prev = 0;
  for(const [pos, len] of m.spans){
    const gap = pos - prev;
    out.set(skel.subarray(sOff, sOff + gap), prev); sOff += gap;
    const n = len / 4, tmp = new Float32Array(n);
    const src = new Uint16Array(packed.buffer.slice(packed.byteOffset + pOff, packed.byteOffset + pOff + n * 2));
    for(let i = 0; i < n; i++) tmp[i] = T[src[i]];
    out.set(new Uint8Array(tmp.buffer), pos);
    pOff += n * 2; prev = pos + len;
  }
  out.set(skel.subarray(sOff), prev);
  return out;
}

/* download several files with one combined progress bar; every file is stored for offline use */
async function pkFetchAll(items, label){
  const total = items.reduce((a, it) => a + it.size, 0);
  let done = 0, last = 0;
  const report = (force) => { const now = Date.now(); if(force || now - last > 150){ last = now; post({type:'progress', label, loaded: done, total}); } };
  const results = [];
  for(const it of items){
    const hit = await cacheGet(it.key);
    if(hit && hit.length === it.size){ done += it.size; results.push(hit); report(true); continue; }
    const out = new Uint8Array(it.size); let off = 0;
    for(const p of it.parts){
      const res = await fetch(PK_BASE + p);
      if(!res.ok) throw new Error('Download fehlgeschlagen (' + res.status + ')');
      if(res.body && res.body.getReader){
        const rd = res.body.getReader();
        for(;;){ const {done: d, value} = await rd.read(); if(d) break; out.set(value, off); off += value.length; done += value.length; report(false); }
      } else { const b = new Uint8Array(await res.arrayBuffer()); out.set(b, off); off += b.length; done += b.length; report(false); }
    }
    if(off !== it.size) throw new Error('Download unvollständig');
    await cachePut(it.key, out);
    results.push(out);
  }
  report(true);
  return results;
}

function pkLoad(withDecoder){
  if(PK.ready) return PK.ready.then(() => (withDecoder ? pkLoadDecoder() : null));
  PK.ready = (async () => {
    pkEnsureOrt();
    const m = await (await fetch(PK_BASE + 'meta.json')).json();
    const items = [
      { key: 'pk-step-' + m.version, size: m.step.size, parts: m.step.parts },
      { key: 'pk-head-' + m.version, size: m.flow_head.graphSize + m.flow_head.packedSize, parts: m.flow_head.parts },
      { key: 'pk-mimi-' + m.version, size: m.mimi_decoder.graphSize + m.mimi_decoder.packedSize, parts: m.mimi_decoder.parts },
      { key: 'pk-lut-' + m.version, size: m.lut.size, parts: m.lut.parts }
    ];
    const [stepB, headB, mimiB, lutB] = await pkFetchAll(items, 'Neue Stimmen');
    post({type:'status', text:'Stimmen werden vorbereitet …'});
    const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    PK.step = await self.ort.InferenceSession.create(stepB, opt);
    PK.head = await self.ort.InferenceSession.create(pkRebuild(headB, m.flow_head), opt);
    if(withDecoder) PK.mimi = await self.ort.InferenceSession.create(pkRebuild(mimiB, m.mimi_decoder), opt);
    PK.lut = new Uint16Array(lutB.buffer.slice(lutB.byteOffset, lutB.byteOffset + lutB.byteLength));
    const tok = await (await fetch(PK_BASE + 'tokenizer.json')).json();
    PK.vocab = new Map(); let minScore = 0;
    tok.vocab.forEach(([piece, score], id) => {
      if(id < 4) return;                       // <unk>, <s>, </s>, <pad>
      PK.vocab.set(piece, [id, score]);
      PK.maxLen = Math.max(PK.maxLen, Array.from(piece).length);
      if(score < minScore) minScore = score;
    });
    PK.unkScore = minScore - 10;
    PK.meta = m;
  })();
  PK.ready.catch(() => { PK.ready = null; });
  return PK.ready;
}
/* decoder only (used by the second worker that turns latents into audio in parallel) */
let PK_DEC = null;
function pkLoadDecoder(){
  if(PK.mimi) return Promise.resolve();
  if(PK_DEC) return PK_DEC;
  PK_DEC = (async () => {
    pkEnsureOrt();
    const m = PK.meta || await (await fetch(PK_BASE + 'meta.json')).json();
    const [mimiB] = await pkFetchAll([{ key: 'pk-mimi-' + m.version, size: m.mimi_decoder.graphSize + m.mimi_decoder.packedSize, parts: m.mimi_decoder.parts }], 'Neue Stimmen');
    PK.mimi = await self.ort.InferenceSession.create(pkRebuild(mimiB, m.mimi_decoder), { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    PK.meta = PK.meta || m;
  })();
  PK_DEC.catch(() => { PK_DEC = null; });
  return PK_DEC;
}
async function pkVoice(name){
  if(PK.voices.has(name)) return PK.voices.get(name);
  const v = PK.meta.voices[name]; if(!v) throw new Error('Unbekannte Stimme');
  const [b] = await pkFetchAll([{ key: 'pk-voice-' + name + '-' + PK.meta.version, size: v.size, parts: [v.file] }], 'Stimme');
  const u16 = new Uint16Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  const n = v.len * PK.meta.heads * PK.meta.head_dim, layers = [];
  for(let i = 0; i < PK.meta.layers * 2; i++){ const a = new Float32Array(n); pkHalfToFloat(u16.subarray(i * n, (i + 1) * n), a, 0); layers.push(a); }
  const st = { len: v.len, kv: layers };
  PK.voices.set(name, st);
  return st;
}

/* ---- text: same preparation as the reference implementation ---- */
const PK_REPLACE = { '"': '', '“': '', '”': '', '„': '', '«': '', '»': '', '’': "'", '‘': "'", '(': '', ')': '', '[': '', ']': '', '–': ',', '—': ',' };
const PK_CLOSERS = '"\'”’)]»';
function pkPrepare(text){
  let t = Array.from(text).map(c => (c in PK_REPLACE ? PK_REPLACE[c] : c)).join('');
  t = t.split(/\s+/).filter(Boolean).join(' ');
  t = t.replace(/([.!?…])\s*[,;:]/g, '$1').replace(/;/g, ',').trim();
  if(!t) return null;
  const words = t.split(' ').length;
  if(t[0] !== t[0].toUpperCase()) t = t[0].toUpperCase() + t.slice(1);
  let core = t; while(core && (PK_CLOSERS + ' ').includes(core[core.length - 1])) core = core.slice(0, -1);
  const closers = t.slice(core.length).trim();
  if(core && !'.!?…'.includes(core[core.length - 1])){
    if(',;:-–—'.includes(core[core.length - 1])){ while(core && ',;:-–— '.includes(core[core.length - 1])) core = core.slice(0, -1); t = core + '.' + closers; }
    else t = t + '.';
  }
  return { text: t, afterEos: (words <= 4 ? 3 : 1) + 2 };
}
/* SentencePiece unigram (Viterbi) with byte fallback */
function pkTokenize(text){
  const s = '▁' + text.replace(/ /g, '▁');
  const ids = [];
  for(const w of s.split(/(?=▁)/)){
    if(!w) continue;
    const ch = Array.from(w), n = ch.length;
    const best = new Float64Array(n + 1).fill(-Infinity), from = new Int32Array(n + 1), tok = new Int32Array(n + 1);
    best[0] = 0;
    for(let i = 0; i < n; i++){
      if(best[i] === -Infinity) continue;
      let piece = '', single = false;
      for(let j = i; j < n && j - i < PK.maxLen; j++){
        piece += ch[j];
        const e = PK.vocab.get(piece);
        if(e){ const sc = best[i] + e[1]; if(sc > best[j + 1]){ best[j + 1] = sc; from[j + 1] = i; tok[j + 1] = e[0]; } if(j === i) single = true; }
      }
      if(!single){ const sc = best[i] + PK.unkScore; if(sc > best[i + 1]){ best[i + 1] = sc; from[i + 1] = i; tok[i + 1] = -1; } }
    }
    const out = [];
    for(let k = n; k > 0; k = from[k]){
      if(tok[k] >= 0) out.push([tok[k]]);
      else {   // byte fallback for characters the vocabulary does not know
        const bytes = new TextEncoder().encode(ch[k - 1]), bt = [];
        for(const b of bytes){ const e = PK.vocab.get('<0x' + b.toString(16).toUpperCase().padStart(2, '0') + '>'); if(e) bt.push(e[0]); }
        out.push(bt);
      }
    }
    for(let k = out.length - 1; k >= 0; k--) ids.push(...out[k]);
  }
  return ids;
}
/* split long sentences so that every piece stays below the model's chunk size */
function pkChunks(text){
  const max = PK.meta.max_tokens_per_chunk;
  const prep = pkPrepare(text); if(!prep) return [];
  if(pkTokenize(prep.text).length <= max) return [prep];
  const parts = prep.text.split(/(?<=[,;:])\s+/), out = [];
  let cur = '';
  const push = s => { const p = pkPrepare(s); if(p) out.push(p); };
  for(const p of parts){
    const cand = cur ? cur + ' ' + p : p;
    if(pkTokenize(cand).length <= max){ cur = cand; continue; }
    if(cur) push(cur);
    if(pkTokenize(p).length <= max){ cur = p; continue; }
    let acc = '';
    for(const w of p.split(' ')){
      const c2 = acc ? acc + ' ' + w : w;
      if(pkTokenize(c2).length > max && acc){ push(acc); acc = w; } else acc = c2;
    }
    cur = acc;
  }
  if(cur) push(cur);
  return out;
}

function pkRandn(){ let u = 0, v = 0; while(u === 0) u = Math.random(); while(v === 0) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

async function pkSynthChunk(prep, voice, temp){
  const m = PK.meta, L = m.layers, HD = m.heads * m.head_dim, T = self.ort.Tensor;
  const ids = pkTokenize(prep.text);
  const maxFrames = Math.ceil((ids.length / 3 + 2) * 12.5);
  const cap = voice.len + ids.length + maxFrames + 4;
  const kv = voice.kv.map(a => { const b = new Float32Array(cap * HD); b.set(a); return b; });
  let P = voice.len;
  const run = async (seq, seqLen, text, textLen) => {
    const feeds = { sequence: new T('float32', seq, [1, seqLen, m.ldim]), text_embeddings: new T('float32', text, [1, textLen, 1024]) };
    for(let i = 0; i < L; i++){
      feeds['past_k_' + i] = new T('float32', kv[2 * i].subarray(0, P * HD), [1, P, m.heads, m.head_dim]);
      feeds['past_v_' + i] = new T('float32', kv[2 * i + 1].subarray(0, P * HD), [1, P, m.heads, m.head_dim]);
    }
    const r = await PK.step.run(feeds);
    const add = seqLen + textLen;
    for(let i = 0; i < L; i++){ kv[2 * i].set(r['new_k_' + i].data, P * HD); kv[2 * i + 1].set(r['new_v_' + i].data, P * HD); }
    P += add;
    return r;
  };
  // 1) text prompt on top of the voice memory
  const emb = new Float32Array(ids.length * 1024);
  ids.forEach((id, k) => pkHalfToFloat(PK.lut.subarray(id * 1024, (id + 1) * 1024), emb, k * 1024));
  await run(new Float32Array(0), 0, emb, ids.length);
  // 2) frame by frame (12.5 frames per second of speech)
  const lat = new Float32Array(maxFrames * m.ldim), noEmb = new Float32Array(0);
  let seq = new Float32Array(m.ldim).fill(NaN), frames = 0, eos = -1;
  const sd = Math.sqrt(temp), s0 = new T('float32', new Float32Array([0]), [1, 1]), t1 = new T('float32', new Float32Array([1]), [1, 1]);
  for(let i = 0; i < maxFrames; i++){
    const r = await run(seq, 1, noEmb, 0);
    if(r.eos_logit.data[0] > m.eos_threshold && eos < 0 && i >= m.min_frames_before_eos) eos = i;
    if(eos >= 0 && i >= eos + prep.afterEos) break;
    const x = new Float32Array(m.ldim); for(let k = 0; k < m.ldim; k++) x[k] = pkRandn() * sd;
    const h = await PK.head.run({ c: new T('float32', r.conditioning.data, [1, 1024]), s: s0, t: t1, x: new T('float32', x, [1, m.ldim]) });
    const d = h.flow_dir.data;
    seq = new Float32Array(m.ldim);
    for(let k = 0; k < m.ldim; k++) seq[k] = x[k] + d[k];
    lat.set(seq, frames * m.ldim); frames++;
  }
  return lat.slice(0, frames * m.ldim);
}
/* latents -> audio, one decoder pass per piece; the decoder starts with a tiny click, so the first 5 ms fade in */
async function pkDecode(latents){
  const T = self.ort.Tensor, ld = 32, parts = [];
  for(const lat of latents){
    const frames = lat.length / ld;
    if(!frames){ parts.push(new Float32Array(0)); continue; }
    const a = await PK.mimi.run({ latent: new T('float32', lat, [1, frames, ld]) });
    const audio = new Float32Array(a.audio.data);
    const fade = Math.min(120, audio.length);
    for(let k = 0; k < fade; k++) audio[k] *= k / fade;
    parts.push(audio);
  }
  return pkJoin(parts);
}
async function pkLatents(job){
  await pkLoad(false);
  const voice = await pkVoice(job.speaker || 'anna');
  const temp = PK.meta.temp * Math.min(1.5, Math.max(0.5, job.expr || 1));
  const out = [];
  for(const prep of pkChunks(job.text)) out.push(await pkSynthChunk(prep, voice, temp));
  return out;
}
async function pkSynth(job){
  const lat = await pkLatents(job);
  await pkLoadDecoder();
  return pkDecode(lat);
}
function pkJoin(parts){
  if(parts.length <= 1) return parts[0] || new Float32Array(0);
  const gap = Math.round(24000 * 0.12), n = parts.reduce((a, p) => a + p.length, 0) + gap * (parts.length - 1);
  const out = new Float32Array(n); let o = 0;
  parts.forEach((p, k) => { out.set(p, o); o += p.length + (k < parts.length - 1 ? gap : 0); });
  return out;
}
async function pkIsCached(){
  try{
    const m = PK.meta || await (await fetch(PK_BASE + 'meta.json')).json();
    for(const k of ['step', 'head', 'mimi', 'lut']) if(!(await cacheHas('pk-' + k + '-' + m.version))) return false;
    return true;
  }catch(e){ return false; }
}
