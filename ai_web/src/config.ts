interface AppConfig {
  appName: string;
  logoFilename: string;
}

let _config: AppConfig = { appName: "evario.ai", logoFilename: "logo.png" };

export async function loadAppConfig(): Promise<AppConfig> {
  try {
    const res = await fetch("/config.json");
    if (res.ok) {
      _config = await res.json();
    }
  } catch {
    // use defaults
  }
  if (_config.appName) {
    document.title = _config.appName;
  }
  return _config;
}

export function getAppConfig(): AppConfig {
  return _config;
}
