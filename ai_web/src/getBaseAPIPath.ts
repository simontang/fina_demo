export function getBaseAPIPath(): string {
    // Return base API URL with /api prefix for LatticeGateway routes
    // LatticeGateway routes are registered at root path but accessed via /api/assistants/* etc.
    if (window.location.hostname === "localhost") {
        return "http://localhost:5702/api";
    }
    // For production, use relative path /api which will be proxied by nginx
    return "/api";
}

