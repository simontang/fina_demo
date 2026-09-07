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
    void infersPostgresHanaAndSqlServerFromJdbcUrlWhenSourceTypeIsMissing() {
        assertThat(DataSourceType.resolve(null, "jdbc:postgresql://db.example.com:5432/postgres"))
                .isEqualTo(DataSourceType.CDP_POSTGRES);

        assertThat(DataSourceType.resolve("", "jdbc:sap://hana.example.com:30015"))
                .isEqualTo(DataSourceType.SAP_B1_HANA);
        assertThat(DataSourceType.SAP_B1_HANA.getTransactionIsolationName()).isNull();

        assertThat(DataSourceType.resolve(null, "jdbc:sqlserver://db.example.com:1433;databaseName=SBODemoUS"))
                .isEqualTo(DataSourceType.SAP_B1_SQLSERVER);
        assertThat(DataSourceType.SAP_B1_SQLSERVER.getDriverClassName())
                .isEqualTo("com.microsoft.sqlserver.jdbc.SQLServerDriver");
        assertThat(DataSourceType.SAP_B1_SQLSERVER.buildConnectionInitSql("dbo")).isNull();
    }

    @Test
    void postgresDatasourceWithoutSchemaDoesNotRunConnectionInitSql() {
        DataSourceType type = DataSourceType.resolve("cdp_postgres", "jdbc:postgresql://db/postgres");

        assertThat(type.buildConnectionInitSql(null)).isNull();
        assertThat(type.buildConnectionInitSql("")).isNull();
    }

    @Test
    void sqlServerJdbcUrlCorrectsLegacyB1HanaSourceType() {
        assertThat(DataSourceType.resolve(
                "sap_b1_hana",
                "jdbc:sqlserver://db.example.com:1433;databaseName=SBODemoUS"))
                .isEqualTo(DataSourceType.SAP_B1_SQLSERVER);

        assertThat(DataSourceType.resolve("mssql", null))
                .isEqualTo(DataSourceType.SAP_B1_SQLSERVER);
    }
}
