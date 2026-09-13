package com.fina.platform.tenant;

/**
 * ThreadLocal holder for the current request's tenant (and optional user).
 * Mirrors bjy_crm_ai's ContextHolder: set by TenantContextInterceptor,
 * cleared in afterCompletion to avoid leaks. Business code reads tenant
 * from here; controllers never touch the header.
 */
public final class TenantContextHolder {

    private static final ThreadLocal<String> TENANT = new ThreadLocal<>();
    private static final ThreadLocal<String> USER = new ThreadLocal<>();

    private TenantContextHolder() {
    }

    public static void setTenant(String tenantId) {
        TENANT.set(tenantId);
    }

    public static String getTenant() {
        return TENANT.get();
    }

    public static void setUser(String userId) {
        USER.set(userId);
    }

    public static String getUser() {
        return USER.get();
    }

    public static void clear() {
        TENANT.remove();
        USER.remove();
    }
}
