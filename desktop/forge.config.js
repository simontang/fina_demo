// Electron Forge config — replaces the former electron-builder "build" block.
// Keeps the same behavior: app named Scale (bundle id ai.scale.desktop), dmg-only
// output, renderer + default config copied into Contents/Resources, and hooks
// that (1) copy the renderer/config and (2) ad-hoc re-sign the whole bundle so
// quarantined recipients get the bypassable "unidentified developer" message
// instead of "app is damaged".
'use strict';

const fs = require('fs');
const path = require('path');
const pkg = require('./package.json');

module.exports = {
  packagerConfig: {
    asar: true,
    name: 'Scale',
    appBundleId: 'ai.scale.desktop',
    appCategoryType: 'public.app-category.business',
    icon: 'assets/icon.icns',
    ignore: [
      /(^|\/)\.git($|\/)/,
      /(^|\/)\.env(\.local)?$/,
      /(^|\/)release($|\/)/,
      /(^|\/)out($|\/)/,
    ],
    osxSign: false,
    osxNotarize: false,
  },
  makers: [
    {
      name: '@electron-forge/maker-dmg',
      config: {
        name: `Scale-${pkg.version}-${process.arch}`,
        icon: 'assets/icon.icns',
      },
    },
  ],
  hooks: {
    // After packaging: copy the built SPA + default gateway config into
    // Contents/Resources, then ad-hoc re-sign the whole bundle (copy first so
    // the new files are sealed too).
    postPackage: async (forgeConfig, options) => {
      const { execSync } = require('child_process');
      const appPaths = (options && options.outputPaths) || [];
      if (!appPaths.length) return;

      const dir = appPaths[0];
      const appPath = fs.readdirSync(dir).map((f) => path.join(dir, f)).find((f) => f.endsWith('.app')) || dir;
      const resources = path.join(appPath, 'Contents', 'Resources');

      const rendererDest = path.join(resources, 'renderer');
      fs.rmSync(rendererDest, { recursive: true, force: true });
      fs.cpSync(path.join(__dirname, '..', 'ai_web', 'dist'), rendererDest, { recursive: true });
      fs.copyFileSync(path.join(__dirname, 'config.default.json'), path.join(resources, 'config.json'));
      console.log(`[forge] copied renderer + default config into ${resources}`);

      console.log(`[forge] ad-hoc re-signing ${appPath}`);
      try {
        execSync(`codesign --force --deep --sign - "${appPath}"`, { stdio: 'inherit' });
        console.log('[forge] bundle sealed with ad-hoc identity');
      } catch (e) {
        console.warn('[forge] ad-hoc re-sign failed:', e.message);
      }
    },
  },
};
