// electron-builder afterSign hook: re-sign the whole bundle consistently with an
// ad-hoc identity (codesign --force --deep --sign -).
//
// Why: Electron ships arm64 binaries that are only partially signed (the main
// executable is linker-signed but the bundle is not sealed), which quarantined
// recipients hit as the hard-to-bypass "app is damaged". A full ad-hoc re-sign
// produces a valid, sealed signature — Gatekeeper then treats it as an
// "unidentified developer" app, which recipients can bypass with right-click ->
// Open. (Developer ID + notarization would be the proper fix; this is the best
// unsigned fallback.)
'use strict';

const { execSync } = require('child_process');
const path = require('path');

exports.default = async function afterSign(context) {
  const appName = context.packager.appInfo.productFilename;
  const appPath = path.join(context.appOutDir, `${appName}.app`);
  console.log(`[afterSign] ad-hoc re-signing ${appPath}`);
  try {
    execSync(`codesign --force --deep --sign - "${appPath}"`, { stdio: 'inherit' });
    console.log('[afterSign] bundle sealed with ad-hoc identity');
  } catch (e) {
    console.warn('[afterSign] ad-hoc re-sign failed:', e.message);
  }
};
