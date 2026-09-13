package com.fina.platform.tenant;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.web.servlet.HandlerInterceptor;

/**
 * Extracts X-Tenant-Id (case-insensitive, bjy_crm_ai style) into
 * TenantContextHolder before any controller or SQL runs.
 *
 * Optional X-Api-Key check: enforced only when file.api-key is configured.
 * Optional default tenant (file.tenant.default-tenant): fuli_survey style
 * dev convenience — keep blank in production so a missing header fails fast.
 */
public class TenantContextInterceptor implements HandlerInterceptor {

    private final String apiKey;
    private final String defaultTenant;

    public TenantContextInterceptor(String apiKey, String defaultTenant) {
        this.apiKey = apiKey;
        this.defaultTenant = defaultTenant;
    }

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler)
            throws Exception {
        if (apiKey != null && !apiKey.isBlank()) {
            String presented = headerIgnoreCase(request, "X-Api-Key");
            if (!apiKey.equals(presented)) {
                reject(response, HttpStatus.UNAUTHORIZED.value(),
                        "API_KEY_INVALID", "X-Api-Key header is missing or invalid");
                return false;
            }
        }

        String tenant = headerIgnoreCase(request, "X-Tenant-Id");
        if (tenant == null || tenant.isBlank()) {
            if (defaultTenant != null && !defaultTenant.isBlank()) {
                tenant = defaultTenant;
            } else {
                reject(response, HttpStatus.BAD_REQUEST.value(),
                        "TENANT_REQUIRED", "X-Tenant-Id header is required");
                return false;
            }
        }
        TenantContextHolder.setTenant(tenant.trim());

        String user = headerIgnoreCase(request, "X-User-Id");
        if (user != null && !user.isBlank()) {
            TenantContextHolder.setUser(user.trim());
        }
        return true;
    }

    @Override
    public void afterCompletion(HttpServletRequest request, HttpServletResponse response,
                                Object handler, Exception ex) {
        TenantContextHolder.clear();
    }

    private String headerIgnoreCase(HttpServletRequest request, String name) {
        String exact = request.getHeader(name);
        if (exact != null) {
            return exact;
        }
        java.util.Enumeration<String> names = request.getHeaderNames();
        while (names.hasMoreElements()) {
            String candidate = names.nextElement();
            if (name.equalsIgnoreCase(candidate)) {
                return request.getHeader(candidate);
            }
        }
        return null;
    }

    private void reject(HttpServletResponse response, int status, String code, String message)
            throws java.io.IOException {
        response.setStatus(status);
        response.setContentType("application/json;charset=UTF-8");
        response.getWriter().write("{\"code\":\"" + code + "\",\"message\":\"" + message + "\"}");
    }
}
