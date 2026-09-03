package com.fina.metrics.util;

import org.springframework.util.StringUtils;

public final class TenantHeaderResolver {

    public static final String DEFAULT_TENANT_ID = "default";

    private TenantHeaderResolver() {
    }

    public static String resolve(String tenantId) {
        return StringUtils.hasText(tenantId) ? tenantId.trim() : DEFAULT_TENANT_ID;
    }
}
