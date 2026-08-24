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

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    title: 'Chrysalis',
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

  const { port: p } = await start({ host: '127.0.0.1', preferredPort: DEFAULT_PORT, gatewayUrl });
  port = p;
  console.log(`[desktop] serving on http://127.0.0.1:${port}/admin/ -> gateway ${gatewayUrl || '(not configured)'}`);

  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('will-quit', () => {
  require('./server').server.close();
});
