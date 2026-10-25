/* PDF-Vorleser – Offline-Speicher für die App-Dateien */
const VERSION = '21fd7d1fdd';
const SHELL = 'vorleser-shell-' + VERSION;
const RUNTIME = 'vorleser-runtime-v1';
const FILES = ["./", "index.html", "manifest.webmanifest", "icon-180.png", "icon-192.png", "icon-512.png", "tts-worker.js", "pocket-engine.js", "lib/pdf.min.js", "lib/pdf.worker.min.js", "engine/ort.wasm.min.js", "engine/piper_phonemize.js", "engine/piper_phonemize.wasm", "engine/ort-wasm-simd.wasm"];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(SHELL).then(c => c.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k.startsWith('vorleser-shell-') && k !== SHELL).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const req = e.request;
  if(req.method !== 'GET') return;
  const url = new URL(req.url);
  if(url.hostname.endsWith('huggingface.co') || url.hostname.endsWith('hf.co')) return;   // voices are stored by the voice worker itself
  if(req.mode === 'navigate'){
    e.respondWith(fetch(req).then(r => { const c = r.clone(); caches.open(SHELL).then(x => x.put('index.html', c)); return r; })
      .catch(() => caches.match('index.html', { ignoreSearch: true })));
    return;
  }
  const sameOrigin = url.origin === self.location.origin;
  if(sameOrigin && url.pathname.endsWith('.wasm') && (url.pathname.includes('/pocket/') || url.pathname.includes('pp-data'))) return;   // large voice files: stored by the voice worker itself
  const fonts = url.hostname === 'fonts.googleapis.com' || url.hostname === 'fonts.gstatic.com';
  if(!sameOrigin && !fonts) return;
  e.respondWith(caches.match(req, { ignoreSearch: true }).then(hit => hit || fetch(req).then(r => {
    if(r.ok || r.type === 'opaque'){ const c = r.clone(); caches.open(sameOrigin ? SHELL : RUNTIME).then(x => x.put(req, c)); }
    return r;
  })));
});
