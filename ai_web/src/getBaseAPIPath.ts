export function getBaseAPIPath(): string {
    return window.location.hostname === "localhost" ? "http://localhost:5702" : "http://14.103.152.204/";
}

