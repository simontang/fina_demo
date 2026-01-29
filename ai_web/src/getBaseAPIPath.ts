/**
 * 返回 API 根路径（不包含 /api），供 SDK 自动拼接 /api 前缀。
 * 本地：http://localhost:5702；生产：当前页面的 origin（如 https://demo.alphafina.cn）。
 */
export function getBaseAPIPath(): string {
    if (window.location.hostname === "localhost") {
        return "http://localhost:5702";
    }
    return `${window.location.protocol}//${window.location.hostname}`;
}

