/* PDF-Vorleser App – Neural-Stimmen (Piper/VITS) direkt auf dem Gerät.
   Die Stimmen werden einmal von Hugging Face geladen und danach offline aus dem Speicher genutzt. */
'use strict';
const BASE = new URL('./', self.location.href).href;
const CACHE = 'vorleser-app-voices-v1';
const PPDATA = { parts: ['engine/pp-data.part0.wasm', 'engine/pp-data.part1.wasm'], size: 18077249 };
const HF = 'https://huggingface.co/';
const MODELS = {
  hoerbuch: HF + 'domoskanonos/piper-tts-models-de/resolve/main/de_DE-audio_book-medium.onnx'
};

importScripts('pocket-engine-fbf2275a.js');   // new voices (Kyutai Pocket TTS)

let engineReady = null, phonemizer = null, phonOut = '';
let session = null, meta = null, voiceId = null, loadingVoice = null;
let queue = [], busy = false;

const post = (m, t) => self.postMessage(m, t || []);

async function cacheGet(key){
  try{ const c = await caches.open(CACHE); const r = await c.match(BASE + '__cache/' + key); if(r) return new Uint8Array(await r.arrayBuffer()); }catch(e){}
  return null;
}
async function cachePut(key, data){
  try{ const c = await caches.open(CACHE); await c.put(BASE + '__cache/' + key, new Response(data)); }catch(e){}
}
async function cacheHas(key){
  try{ const c = await caches.open(CACHE); return !!(await c.match(BASE + '__cache/' + key)); }catch(e){ return false; }
}

async function fetchParts(key, parts, size, label){
  const hit = await cacheGet(key);
  if(hit && (!size || hit.length === size)){ post({type:'progress', label, loaded: hit.length, total: hit.length, cached: true}); return hit; }
  const chunks = []; let off = 0, last = 0, total = size || 0;
  for(const p of parts){
    const res = await fetch(new URL(p, BASE).href);
    if(!res.ok) throw new Error('Download fehlgeschlagen (' + res.status + ')');
    if(!size){ const cl = +res.headers.get('content-length'); if(cl) total += cl; }
    if(res.body && res.body.getReader){
      const reader = res.body.getReader();
      for(;;){
        const {done, value} = await reader.read();
        if(done) break;
        chunks.push(value); off += value.length;
        const now = Date.now(); if(now - last > 150){ last = now; post({type:'progress', label, loaded: off, total: Math.max(total, off)}); }
      }
    } else { const b = new Uint8Array(await res.arrayBuffer()); chunks.push(b); off += b.length; }
  }
  if(size && off !== size) throw new Error('Download unvollständig');
  const out = new Uint8Array(off); let o = 0;
  for(const c of chunks){ out.set(c, o); o += c.length; }
  post({type:'progress', label, loaded: off, total: off});
  await cachePut(key, out);
  return out;
}

function initEngine(){
  if(engineReady) return engineReady;
  engineReady = (async () => {
    pkEnsureOrt();
    importScripts(BASE + 'engine/piper_phonemize.js');
    const data = await fetchParts('ppdata', PPDATA.parts, PPDATA.size, 'Sprach-Engine');
    const wasm = await (await fetch(BASE + 'engine/piper_phonemize.wasm')).arrayBuffer();
    phonemizer = await self.createPiperPhonemize({
      wasmBinary: wasm,
      getPreloadedPackage: () => data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
      locateFile: f => BASE + 'engine/' + f,
      print: s => { phonOut += s + '\n'; },
      printErr: () => {},
      noInitialRun: true
    });
  })();
  engineReady.catch(() => { engineReady = null; });
  return engineReady;
}

async function loadVoice(id){
  if(id === 'pocket'){ await pkLoad(false); return; }
  if(id === 'pocket-dec'){ await pkLoadDecoder(); return; }
  if(voiceId === id && session) return;
  await initEngine();
  const url = MODELS[id]; if(!url) throw new Error('Unbekannte Stimme');
  let cfg = null;
  const cfgHit = await cacheGet('cfg-' + id);
  if(cfgHit) cfg = JSON.parse(new TextDecoder().decode(cfgHit));
  else {
    const r = await fetch(url + '.json'); if(!r.ok) throw new Error('Download fehlgeschlagen (' + r.status + ')');
    const txt = await r.text(); cfg = JSON.parse(txt); await cachePut('cfg-' + id, new TextEncoder().encode(txt));
  }
  const model = await fetchParts('voice-' + id, [url], 0, 'Stimme');
  post({type:'status', text:'Stimme wird vorbereitet …'});
  if(session){ try{ await session.release(); }catch(e){} session = null; }
  session = await self.ort.InferenceSession.create(model, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  meta = { sample_rate: cfg.audio.sample_rate, espeak: (cfg.espeak && cfg.espeak.voice) || 'de', inference: cfg.inference,
           num_speakers: cfg.num_speakers || 1, speaker_id_map: cfg.speaker_id_map || {}, phoneme_id_map: cfg.phoneme_id_map };
  voiceId = id;
}

function phonemize(text){
  phonOut = '';
  phonemizer.callMain(['-l', meta.espeak, '--input', JSON.stringify([{text}]), '--espeak_data', '/espeak-ng-data']);
  const phs = [];
  for(const l of phonOut.split('\n')){ if(!l.trim()) continue; try{ phs.push(...JSON.parse(l).phonemes); }catch(e){} }
  return phs;
}
function toIds(phs){
  const map = meta.phoneme_id_map;
  const ids = [...map['^'], ...map['_']];
  let prev = null;
  for(const p of phs){
    if(map[p]){ ids.push(...map[p], ...map['_']); prev = p; continue; }
    if(prev){
      const c = (prev + p).normalize('NFC');
      if([...c].length === 1 && map[c]){ ids.splice(ids.length - 1 - map[prev].length, map[prev].length + 1, ...map[c], ...map['_']); prev = c; }
    }
  }
  ids.push(...map['$']);
  return ids;
}

async function synth(job){
  if(job.voice === 'pocket') return pkSynth(job);
  const phs = phonemize(job.text);
  if(!phs.length) return new Float32Array(0);
  const ids = toIds(phs);
  const inf = meta.inference, ex = job.expr || 1;
  const feeds = {
    input: new self.ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    input_lengths: new self.ort.Tensor('int64', BigInt64Array.from([BigInt(ids.length)]), [1]),
    scales: new self.ort.Tensor('float32', Float32Array.from([inf.noise_scale * ex, inf.length_scale / (job.rate || 1), Math.min(1.2, inf.noise_w * ex)]), [3])
  };
  if(meta.num_speakers > 1){
    const sid = meta.speaker_id_map[job.speaker] ?? meta.speaker_id_map.neutral ?? 0;
    feeds.sid = new self.ort.Tensor('int64', BigInt64Array.from([BigInt(sid)]), [1]);
  }
  const r = await session.run(feeds);
  return new Float32Array(r.output.data);
}

/* whole-book preparation: finished audio is packed as MP3 so a long book fits on the phone */
function encodeMp3(job){
  if(!self.lamejs || !self.lamejs.Mp3Encoder) importScripts(BASE + 'lame.min.js');
  const enc = new self.lamejs.Mp3Encoder(1, job.sr, job.kbps || 48), f = job.pcm, parts = [], step = 1152 * 16;
  let total = 0;
  const i16 = new Int16Array(step);
  for(let o = 0; o < f.length; o += step){
    const n = Math.min(step, f.length - o);
    for(let k = 0; k < n; k++){ const v = f[o + k]; i16[k] = v <= -1 ? -32768 : v >= 1 ? 32767 : v * 32767; }
    const b = enc.encodeBuffer(n === step ? i16 : i16.subarray(0, n)); if(b.length){ parts.push(new Uint8Array(b.buffer, b.byteOffset, b.length).slice()); total += b.length; }
  }
  const e = enc.flush(); if(e.length){ parts.push(new Uint8Array(e.buffer, e.byteOffset, e.length).slice()); total += e.length; }
  const out = new Uint8Array(total); let o = 0; for(const p of parts){ out.set(p, o); o += p.length; }
  return out;
}

let gen = 0;   // bumped by a full reset: running book jobs stop early
async function runJob(job){
  try{
    if(job.type === 'encode'){
      const data = encodeMp3(job);
      post({type:'mp3', id: job.id, data: data.buffer}, [data.buffer]);
      return;
    }
    if(job.type === 'decode'){   // latents -> audio (mostly in the second worker, in parallel to the first one)
      await pkLoadDecoder();
      const audio = await pkDecode(job.chunks);
      post({type:'audio', id: job.id, audio, sampleRate: 24000}, [audio.buffer]);
      return;
    }
    if(job.type === 'synthMany'){   // book preparation: many sentences, computed 4 at a time
      await loadVoice('pocket');
      const my = gen;
      const jobs = job.items.map(it => ({ text: it.text, speaker: job.speaker, expr: job.expr }));
      await pkLatentsMany(jobs, job.batch || 4, async (j, chunks) => {
        const id = job.items[j].id;
        if(job.mode === 'latents'){ post({type:'latents', id, chunks}, chunks.map(c => c.buffer)); return; }
        await pkLoadDecoder(); const audio = await pkDecode(chunks);
        post({type:'audio', id, audio, sampleRate: 24000}, [audio.buffer]);
      }, async () => { if(gen !== my) throw new Error('abgebrochen'); await servePrio(); });
      return;
    }
    await loadVoice(job.voice);
    if(job.voice === 'pocket' && job.mode === 'latents'){
      const chunks = await pkLatents(job);
      post({type:'latents', id: job.id, chunks}, chunks.map(c => c.buffer));
      return;
    }
    const audio = await synth(job);
    post({type:'audio', id: job.id, audio, sampleRate: job.voice === 'pocket' ? 24000 : meta.sample_rate}, [audio.buffer]);
  }catch(e){
    if(job.type === 'synthMany') for(const it of job.items) post({type:'error', id: it.id, message: String(e && e.message || e)});
    else post({type:'error', id: job.id, message: String(e && e.message || e)});
  }
}
/* reading aloud may not wait behind a long book job: it is served between two batches */
async function servePrio(){
  for(;;){ const k = queue.findIndex(j => j.prio); if(k < 0) return; await runJob(queue.splice(k, 1)[0]); }
}
async function pump(){
  if(busy) return;
  busy = true;
  while(queue.length){
    const k = queue.findIndex(j => j.prio);   // reading aloud goes before the book preparation
    await runJob(queue.splice(k < 0 ? 0 : k, 1)[0]);
  }
  busy = false;
}

self.onmessage = async e => {
  const m = e.data;
  if(m.type === 'load'){
    try{ loadingVoice = m.voice; await loadVoice(m.voice); post({type:'ready', voice: m.voice}); }
    catch(err){ post({type:'loadError', voice: m.voice, message: String(err && err.message || err)}); }
    finally{ loadingVoice = null; }
  } else if(m.type === 'synth' || m.type === 'decode' || m.type === 'encode' || m.type === 'synthMany'){
    queue.push(m); pump();
  } else if(m.type === 'reset'){
    if(m.all) gen++;
    queue = m.all ? [] : queue.filter(j => !j.prio);
  } else if(m.type === 'isCached'){
    const ok = m.voice === 'pocket' ? await pkIsCached() : (await cacheHas('voice-' + m.voice)) && (await cacheHas('cfg-' + m.voice)) && (await cacheHas('ppdata'));
    post({type:'cached', voice: m.voice, value: ok});
  }
};
