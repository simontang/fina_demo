// Evario Desktop — Electron main process.
// Loads the packaged renderer from a local server (server.js) which serves the
// SPA under /admin/ and proxies /api/* to the remote Agent Gateway.
//
// The gateway URL is deliberately NOT hardcoded here. Resolution order:
//   1. process env  GATEWAY_URL
//   2. desktop/.env (dotenv, gitignored — dev convenience)
//   3. userData/config.json  { "gatewayUrl": "..." }  (user override)
//   4. bundled config.json  (extraResources default, ships with the app)
// If none is set, /api returns a clear 502 and the app logs a message.
'use strict';

const { app, BrowserWindow, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const dotenv = require('dotenv');
const { start } = require('./server');

const DEFAULT_PORT = 5721;

// Load optional .env next to this file (dev). Silently no-ops when absent and
// never overrides already-set process env vars.
dotenv.config({ path: path.join(__dirname, '.env') });

function readGatewayUrlFromFile(filePath) {
  try {
    if (filePath && fs.existsSync(filePath)) {
      const cfg = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      if (cfg.gatewayUrl && cfg.gatewayUrl.trim()) {
        return cfg.gatewayUrl.trim().replace(/\/+$/, '');
      }
    }
  } catch (e) {
    console.warn(`[desktop] config read failed (${filePath}):`, e.message);
  }
  return null;
}

function bundledConfigPath() {
  // Packaged: <app>/Contents/Resources/config.json (extraResources).
  // Dev: local template next to this file.
  return app.isPackaged
    ? path.join(process.resourcesPath, 'config.json')
    : path.join(__dirname, 'config.default.json');
}

function resolveGatewayUrl() {
  if (process.env.GATEWAY_URL && process.env.GATEWAY_URL.trim()) {
    return process.env.GATEWAY_URL.trim().replace(/\/+$/, '');
  }
  const userCfg = readGatewayUrlFromFile(path.join(app.getPath('userData'), 'config.json'));
  if (userCfg) return userCfg;
  return readGatewayUrlFromFile(bundledConfigPath());
}

let port;
let win;

// --- sessionStorage persistence -------------------------------------------------
// The SPA + SDK keep auth in sessionStorage, which dies with the tab, so a fresh
// launch always landed back on the login page. We snapshot the whole
// sessionStorage to userData and re-inject it into the SPA entry on next start
// (server.js sendIndex). Snapshot cadence: every 20s while a window is alive,
// plus a final best-effort capture right before quit.
const sessionStoreFile = () => path.join(app.getPath('userData'), 'session-storage.json');

function readStoredSession() {
  try {
    return JSON.parse(fs.readFileSync(sessionStoreFile(), 'utf8'));
  } catch (e) {
    return null;
  }
}

async function snapshotSession() {
  if (!win || win.isDestroyed() || win.webContents.isDestroyed()) return;
  try {
    // Return a JSON string so executeJavaScript avoids structured-clone of a big
    // object; write only when the payload actually changed.
    const data = await win.webContents.executeJavaScript(
      'JSON.stringify(Object.fromEntries(Object.entries(sessionStorage)))'
    );
    const existing = fs.existsSync(sessionStoreFile()) ? fs.readFileSync(sessionStoreFile(), 'utf8') : '';
    if (existing !== data) {
      fs.writeFileSync(sessionStoreFile(), data);
    }
  } catch (e) {
    // Renderer busy / navigating — skip this tick; not fatal.
  }
}

function createWindow() {
  win = new BrowserWindow({
    width: 1440,
    height: 900,
    title: 'Scale',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  // Load via 127.0.0.1 (NOT localhost): the renderer's getBaseAPIPath()
  // special-cases localhost to http://localhost:5701, which would be wrong here.
  win.loadURL(`http://127.0.0.1:${port}/admin/`);

  // Send external https links (e.g. the sandbox page) to the system browser
  // instead of navigating the shell away from the local app origin.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
  win.webContents.on('will-navigate', (e, url) => {
    if (!url.startsWith(`http://127.0.0.1:${port}`)) {
      e.preventDefault();
      if (/^https?:\/\//.test(url)) shell.openExternal(url);
    }
  });
}

app.whenReady().then(async () => {
  // Dev mode (electron .): show our icon in the dock/task switcher instead of
  // the default Electron icon. Packaged apps get it from the .app bundle.
  if (!app.isPackaged && process.platform === 'darwin') {
    app.dock.setIcon(path.join(__dirname, 'assets', 'icon.png'));
  }

  const gatewayUrl = resolveGatewayUrl();
  if (!gatewayUrl) {
    console.error('[desktop] Agent Gateway not configured. Set GATEWAY_URL via env, desktop/.env, or userData/config.json.');
  }

  const { port: p } = await start({
    host: '127.0.0.1',
    preferredPort: DEFAULT_PORT,
    gatewayUrl,
    restoreSession: readStoredSession(),
  });
  port = p;
  console.log(`[desktop] serving on http://127.0.0.1:${port}/admin/ -> gateway ${gatewayUrl || '(not configured)'}`);

  // Periodically snapshot sessionStorage so a login mid-session is captured (and
  // a logout isn't resurrected from a stale file next launch).
  setInterval(() => {
    snapshotSession();
  }, 20000);

  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// Final snapshot before quit so the last ≤20s of state (e.g. a fresh login) is
// not lost on Cmd+Q. Defer quit until the renderer has handed over its session.
let quitPending = false;
app.on('before-quit', (e) => {
  if (quitPending) return;
  e.preventDefault();
  quitPending = true;
  const done = Promise.race([
    snapshotSession(),
    new Promise((resolve) => setTimeout(resolve, 1000)),
  ]);
  done.finally(() => app.quit());
});

app.on('will-quit', () => {
  require('./server').server.close();
});
