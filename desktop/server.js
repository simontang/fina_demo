// Desktop local server: serves the packaged SPA under /admin/ and reverse-proxies
// /api/* to the remote Agent Gateway. Streaming/SSE-safe (no buffering).
//
// The gateway target is NOT hardcoded here — it is passed to start() by the
// Electron main process (which resolves it from env / .env / userData config).
'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const httpProxy = require('http-proxy');

// Resolve the renderer directory. In packaged builds the SPA lives in
// <app>/Contents/Resources/renderer (extraResources); in dev it points at
// ai_web/dist via the RENDERER_DIR env var (set in package.json scripts).
function resolveRendererDir() {
  if (process.env.RENDERER_DIR) return process.env.RENDERER_DIR;
  if (process.resourcesPath) return path.join(process.resourcesPath, 'renderer');
  return path.join(__dirname, '..', 'ai_web', 'dist');
}

const RENDERER_DIR = resolveRendererDir();

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.eot': 'application/vnd.ms-fontobject',
  '.wasm': 'application/wasm',
};

// Gateway target + proxy instance, set by start().
let gatewayUrl = null;
let proxy = null;

// Restored sessionStorage snapshot from the previous launch (Electron main
// process persists it to userData). Injected into the SPA entry so login
// survives app restarts — see sendIndex().
let restoreSession = null;

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1');
  const p = url.pathname;

  // SPA entry: /, /admin, /admin/
  if (p === '/' || p === '/admin' || p === '/admin/') {
    return sendIndex(res);
  }

  // Pass-through reverse proxy for all gateway APIs (path preserved).
  if (p.startsWith('/api/') || p === '/api') {
    if (!proxy || !gatewayUrl) {
      res.writeHead(502, { 'Content-Type': 'text/plain; charset=utf-8' });
      return res.end('Agent Gateway not configured. Set GATEWAY_URL via env, desktop/.env, or userData/config.json.');
    }
    return proxy.web(req, res);
  }

  // Static assets under /admin/.
  if (p.startsWith('/admin/')) {
    const rel = p.slice('/admin/'.length);
    const filePath = path.join(RENDERER_DIR, rel);
    if (!filePath.startsWith(RENDERER_DIR + path.sep)) {
      res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
      return res.end('Forbidden');
    }
    // Direct /admin/index.html requests get the same session-restore treatment.
    if (path.basename(filePath) === 'index.html') {
      return sendIndex(res);
    }
    return sendFile(res, filePath);
  }

  res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
  res.end('Not found');
});

// SPA entry. When a stored session exists, injects an inline script that
// restores sessionStorage BEFORE the deferred module bundle runs (classic inline
// scripts execute during parse; module scripts run after the document is parsed).
// The guard (sessionStorage.length === 0) means only a fresh tab is hydrated —
// a mid-run reload never clobbers live auth state with a stale snapshot.
function sendIndex(res) {
  fs.readFile(path.join(RENDERER_DIR, 'index.html'), (err, buf) => {
    if (err || !buf) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Not found');
      return;
    }
    let html = buf.toString('utf8');
    if (restoreSession && Object.keys(restoreSession).length) {
      // Escape '<' so no stored value can break out of the inline script.
      const payload = JSON.stringify(restoreSession).replace(/</g, '\\u003c');
      const script =
        '<script>(function(){try{' +
        'if(sessionStorage.length>0)return;' +
        'var s=' + payload + ';' +
        'for(var k in s){sessionStorage.setItem(k,s[k]);}' +
        '}catch(e){}})();</script>';
      html = html.includes('</body>')
        ? html.replace('</body>', script + '</body>')
        : html + script;
    }
    res.writeHead(200, {
      'Content-Type': 'text/html; charset=utf-8',
      'Cache-Control': 'no-cache',
    });
    res.end(html);
  });
}

function sendFile(res, filePath) {
  fs.stat(filePath, (err, st) => {
    if (err || !st.isFile()) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Not found');
      return;
    }
    const type = MIME[path.extname(filePath).toLowerCase()] || 'application/octet-stream';
    // Only the hashed build assets under /admin/assets/ are immutable. Brand files
    // (config.json, logo.png, favicon.ico) and index.html must never be cached long —
    // otherwise an app update keeps serving the previous version's logo/branding.
    const isHashedAsset = filePath.includes(path.sep + 'assets' + path.sep);
    res.writeHead(200, {
      'Content-Type': type,
      'Cache-Control': isHashedAsset ? 'public, max-age=31536000, immutable' : 'no-cache',
    });
    fs.createReadStream(filePath).pipe(res);
  });
}

function start({ host = '127.0.0.1', preferredPort = 5721, gatewayUrl: g = null, restoreSession: rs = null } = {}) {
  gatewayUrl = g;
  restoreSession = rs;

  if (g) {
    proxy = httpProxy.createProxyServer({
      target: g.replace(/\/+$/, ''),
      changeOrigin: true,
      xfwd: true,
    });

    // SSE hardening: never buffer, keep alive, avoid intermediary caching.
    proxy.on('proxyRes', (proxyRes, req, res) => {
      const ct = (proxyRes.headers['content-type'] || '').toLowerCase();
      if (ct.includes('text/event-stream')) {
        res.setHeader('X-Accel-Buffering', 'no');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
      }
    });

    proxy.on('error', (err, req, res) => {
      if (res && !res.headersSent) {
        res.writeHead(502, { 'Content-Type': 'text/plain; charset=utf-8' });
      }
      if (res) res.end(`Gateway unreachable (${gatewayUrl}): ${err.message}`);
    });
  } else {
    proxy = null;
  }

  return new Promise((resolve) => {
    const onListen = () => {
      const actual = server.address().port;
      if (preferredPort && actual !== preferredPort) {
        console.warn(`[desktop] port ${preferredPort} in use; fell back to ephemeral port ${actual} (localStorage origin changes)`);
      }
      resolve({ server, port: actual, gatewayUrl, rendererDir: RENDERER_DIR });
    };
    server.once('error', (e) => {
      if (e.code === 'EADDRINUSE' && preferredPort) {
        preferredPort = 0;
        server.listen(0, host, onListen);
      } else {
        throw e;
      }
    });
    server.listen(preferredPort, host, onListen);
  });
}

module.exports = { start, server };
