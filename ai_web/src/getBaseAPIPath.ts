/**
 * 返回 API 根路径（不包含 /api），供 SDK 自动拼接 /api 前缀。
 * 使用 window.location.origin：既兼容标准端口（80/443），也保留非标准端口
 * （如 Electron 内嵌服务 127.0.0.1:5721、本地开发 5701）。
 */
export function getBaseAPIPath(): string {
    return window.location.origin;
}

