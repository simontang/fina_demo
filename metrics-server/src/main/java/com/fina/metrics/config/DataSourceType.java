package com.fina.metrics.config;

import org.springframework.util.StringUtils;

import java.util.Locale;

public enum DataSourceType {
    SAP_B1_HANA("sap_b1_hana", "com.sap.db.jdbc.Driver", "b1-hana-pool", "b1-hana-test"),
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

    public String getTestPoolNamePrefix() {
        return testPoolNamePrefix;
    }

    public String buildConnectionInitSql(String schemaName) {
        if (!StringUtils.hasText(schemaName)) {
            return null;
        }
        return switch (this) {
            case SAP_B1_HANA -> "SET SCHEMA \"" + schemaName + "\"";
            case CDP_POSTGRES -> "SET search_path TO \"" + schemaName + "\"";
        };
    }

    public String getTransactionIsolationName() {
        return switch (this) {
            case SAP_B1_HANA -> null;
            case CDP_POSTGRES -> "TRANSACTION_READ_COMMITTED";
        };
    }

    public static DataSourceType resolve(String sourceType, String jdbcUrl) {
        if (StringUtils.hasText(sourceType)) {
            String normalized = sourceType.trim().toLowerCase(Locale.ROOT);
            for (DataSourceType type : values()) {
                if (type.code.equals(normalized)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("Unsupported datasource sourceType: " + sourceType);
        }

        String normalizedUrl = jdbcUrl != null ? jdbcUrl.trim().toLowerCase(Locale.ROOT) : "";
        if (normalizedUrl.startsWith("jdbc:postgresql:")) {
            return CDP_POSTGRES;
        }
        return SAP_B1_HANA;
    }

    public boolean isCdp() {
        return this == CDP_POSTGRES;
    }
}
