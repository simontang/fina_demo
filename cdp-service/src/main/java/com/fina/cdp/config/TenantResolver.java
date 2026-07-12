package com.fina.cdp.config;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.stereotype.Component;

@Component
public class TenantResolver {

    public String resolve(HttpServletRequest request) {
        if (request == null) {
            return "default";
        }
        return normalize(firstHeader(request, "X-Tenant-Id", "x-tenant-id", "Tenant-Id"));
    }

    public static String normalize(String tenantId) {
        if (tenantId == null || tenantId.isBlank()) {
            return "default";
        }
        return tenantId.trim();
    }

    private static String firstHeader(HttpServletRequest request, String... names) {
        for (String name : names) {
            String value = request.getHeader(name);
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return null;
    }
}
