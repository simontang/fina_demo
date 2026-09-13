package com.fina.platform.config;

import com.baomidou.mybatisplus.extension.plugins.MybatisPlusInterceptor;
import com.baomidou.mybatisplus.extension.plugins.handler.TenantLineHandler;
import com.baomidou.mybatisplus.extension.plugins.inner.TenantLineInnerInterceptor;
import com.fina.platform.tenant.TenantContextHolder;
import net.sf.jsqlparser.expression.Expression;
import net.sf.jsqlparser.expression.StringValue;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Transparent query-side tenant isolation: every SELECT/UPDATE/DELETE gets
 * `tenant_id = '<current>'` appended and every INSERT gets the column filled,
 * driven by TenantContextHolder. Mappers and services stay tenant-free.
 */
@Configuration
public class MybatisPlusConfig {

    @Bean
    public MybatisPlusInterceptor mybatisPlusInterceptor(TenantLineInnerInterceptor tenantLine) {
        MybatisPlusInterceptor interceptor = new MybatisPlusInterceptor();
        // Tenant filter must run before pagination or any other inner interceptor.
        interceptor.addInnerInterceptor(tenantLine);
        return interceptor;
    }

    @Bean
    public TenantLineInnerInterceptor tenantLineInnerInterceptor() {
        TenantLineHandler handler = new TenantLineHandler() {
            @Override
            public Expression getTenantId() {
                String tenant = TenantContextHolder.getTenant();
                if (tenant == null || tenant.isBlank()) {
                    // Should be unreachable behind TenantContextInterceptor;
                    // fail loudly rather than leak across tenants.
                    throw new IllegalStateException("tenant context is missing");
                }
                return new StringValue(tenant);
            }

            @Override
            public String getTenantIdColumn() {
                return "tenant_id";
            }

            @Override
            public boolean ignoreTable(String tableName) {
                // Every table in this service is tenant-scoped.
                return false;
            }
        };
        return new TenantLineInnerInterceptor(handler);
    }
}
