package com.fina.cdp.config;

import org.springframework.util.StringUtils;

import java.util.Locale;

public enum DataSourceType {
    CDP_POSTGRES("cdp_postgres", "org.postgresql.Driver", "cdp-postgres-pool", "cdp-postgres-test");

    private final String code;
    private final String driverClassName;
    private final String poolNamePrefix;
    private final String testPoolNamePrefix;

    DataSourceType(String code, String driverClassName, String poolNamePrefix, String testPoolNamePrefix) {
        this.code = code;
        this.driverClassName = driverClassName;
        this.poolNamePrefix = poolNamePrefix;
        this.testPoolNamePrefix = testPoolNamePrefix;
    }

    public String getCode() {
        return code;
    }

    public String getDriverClassName() {
        return driverClassName;
    }

    public String getPoolNamePrefix() {
        return poolNamePrefix;
    }

    public String buildConnectionInitSql(String schemaName) {
        if (!StringUtils.hasText(schemaName)) {
            return null;
        }
        return "SET search_path TO \"" + schemaName + "\"";
    }

    public static DataSourceType resolve(String sourceType, String jdbcUrl) {
        if (StringUtils.hasText(sourceType)) {
            String normalized = sourceType.trim().toLowerCase(Locale.ROOT);
            for (DataSourceType type : values()) {
                if (type.code.equals(normalized)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("Unsupported CDP datasource sourceType: " + sourceType);
        }

        String normalizedUrl = jdbcUrl != null ? jdbcUrl.trim().toLowerCase(Locale.ROOT) : "";
        if (normalizedUrl.startsWith("jdbc:postgresql:")) {
            return CDP_POSTGRES;
        }
        throw new IllegalArgumentException("CDP service only supports PostgreSQL datasource URLs");
    }
}
