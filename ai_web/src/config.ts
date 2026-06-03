interface AppConfig {
  appName: string;
  logoFilename: string;
  faviconFilename: string;
}

let _config: AppConfig = {
  appName: "evario.ai",
  logoFilename: "logo.png",
  faviconFilename: "favicon.ico",
};

export async function loadAppConfig(): Promise<AppConfig> {
  try {
    const res = await fetch("/admin/config.json");
    if (res.ok) {
      _config = await res.json();
    }
  } catch {
    // use defaults
  }
  if (_config.appName) {
    document.title = _config.appName;
  }
  if (_config.faviconFilename) {
    const faviconLink = document.getElementById("favicon-link") as HTMLLinkElement;
    if (faviconLink) {
      faviconLink.href = `./${_config.faviconFilename}`;
    }
  }
  return _config;
}

export function getAppConfig(): AppConfig {
  return _config;
}
