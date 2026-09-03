package com.fina.metrics.config;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class DataSourceTypeTest {

    @Test
    void resolvesExplicitCdpPostgresType() {
        DataSourceType type = DataSourceType.resolve(
                "cdp_postgres",
                "jdbc:sap://hana.example.com:30015?currentSchema=SBODEMOUS");

        assertThat(type).isEqualTo(DataSourceType.CDP_POSTGRES);
        assertThat(type.getDriverClassName()).isEqualTo("org.postgresql.Driver");
        assertThat(type.buildConnectionInitSql("public")).isEqualTo("SET search_path TO \"public\"");
        assertThat(type.getTransactionIsolationName()).isEqualTo("TRANSACTION_READ_COMMITTED");
    }

    @Test
    void infersPostgresAndHanaFromJdbcUrlWhenSourceTypeIsMissing() {
        assertThat(DataSourceType.resolve(null, "jdbc:postgresql://db.example.com:5432/postgres"))
                .isEqualTo(DataSourceType.CDP_POSTGRES);

        assertThat(DataSourceType.resolve("", "jdbc:sap://hana.example.com:30015"))
                .isEqualTo(DataSourceType.SAP_B1_HANA);
        assertThat(DataSourceType.SAP_B1_HANA.getTransactionIsolationName()).isNull();
    }

    @Test
    void postgresDatasourceWithoutSchemaDoesNotRunConnectionInitSql() {
        DataSourceType type = DataSourceType.resolve("cdp_postgres", "jdbc:postgresql://db/postgres");

        assertThat(type.buildConnectionInitSql(null)).isNull();
        assertThat(type.buildConnectionInitSql("")).isNull();
    }
}
