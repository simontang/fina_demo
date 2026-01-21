export function getBaseAPIPath(): string {
    // 本地开发环境
    if (window.location.hostname === "localhost") {
        return "http://localhost:5702";
    }
    // 生产环境：使用当前页面的协议和主机名（自动使用域名）
    // 这样当通过 https://demo.alphafina.cn 访问时，API 也会使用 https://demo.alphafina.cn
    // 当通过 http://14.103.152.204 访问时，API 也会使用 http://14.103.152.204
    return `${window.location.protocol}//${window.location.hostname}`;
}

