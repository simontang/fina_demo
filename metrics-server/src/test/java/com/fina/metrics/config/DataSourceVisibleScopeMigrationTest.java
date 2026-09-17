package com.fina.metrics.config;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.core.io.ClassPathResource;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.springframework.jdbc.datasource.init.ResourceDatabasePopulator;

import java.util.List;
import java.util.Properties;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** Opt-in real PostgreSQL test; never uses application datasource credentials. */
@EnabledIfEnvironmentVariable(named = "METRICS_TEST_PG_URL",
        matches = "jdbc:postgresql://(localhost|127\\.0\\.0\\.1|\\[::1\\])(:[0-9]+)?/.*")
class DataSourceVisibleScopeMigrationTest {

    private JdbcTemplate admin;
    private JdbcTemplate jdbc;
    private DriverManagerDataSource dataSource;
    private String schema;

    @BeforeEach
    void createIsolatedSchema() {
        String url = System.getenv("METRICS_TEST_PG_URL");
        String user = System.getenv().getOrDefault("METRICS_TEST_PG_USER", System.getProperty("user.name"));
        String password = System.getenv().getOrDefault("METRICS_TEST_PG_PASSWORD", "");
        admin = new JdbcTemplate(new DriverManagerDataSource(url, user, password));
        schema = "visibility_test_" + UUID.randomUUID().toString().replace("-", "");
        admin.execute("CREATE SCHEMA " + schema);
        dataSource = new DriverManagerDataSource(url, user, password);
        Properties properties = new Properties();
        properties.setProperty("currentSchema", schema);
        dataSource.setConnectionProperties(properties);
        jdbc = new JdbcTemplate(dataSource);
    }

    @AfterEach
    void removeIsolatedSchema() {
        if (admin != null && schema != null) {
            admin.execute("DROP SCHEMA IF EXISTS " + schema + " CASCADE");
        }
    }

    @Test
    void historicalRowsMigrateConservativelyAndStartupRerunsNeverRestoreGrants() {
        createLegacySchema();
        List<String> originalGrants = grantRows();
        List<String> originalDatasources = datasourceRowsWithoutMode();

        new MasterSchemaInitializer(dataSource).init();

        assertThat(modes()).containsExactly("RESTRICTED", "RESTRICTED", "RESTRICTED", "RESTRICTED",
                "ALL", "RESTRICTED", "ALL", "RESTRICTED");
        assertThat(grantRows()).isEqualTo(originalGrants);
        assertThat(datasourceRowsWithoutMode()).isEqualTo(originalDatasources);
        assertThat(jdbc.queryForObject("SELECT indexdef FROM pg_indexes WHERE schemaname = ? AND indexname = ?",
                String.class, schema, "idx_dstg_ds_status_deleted"))
                .contains("(datasource_id, status, deleted)");

        jdbc.update("UPDATE t_datasource_config SET visible_scope_mode = 'ALL' WHERE id = 1");
        jdbc.update("UPDATE t_datasource_config SET visible_scope_mode = 'RESTRICTED' WHERE id = 5");
        jdbc.update("UPDATE t_datasource_table_grant SET deleted = 1 WHERE datasource_id = 1");
        List<String> modifiedGrants = grantRows();
        List<String> explicitModes = modes();

        new MasterSchemaInitializer(dataSource).init();
        migrate();

        assertThat(modes()).isEqualTo(explicitModes);
        assertThat(grantRows()).isEqualTo(modifiedGrants);
        assertNewDefaultsAndValidation();
    }

    @Test
    void partiallyMigratedDatabaseOnlyInitializesNullModes() {
        createLegacySchema();
        jdbc.execute("ALTER TABLE t_datasource_config ADD COLUMN visible_scope_mode VARCHAR(16)");
        jdbc.update("UPDATE t_datasource_config SET visible_scope_mode = 'ALL' WHERE id = 1");
        jdbc.update("UPDATE t_datasource_config SET visible_scope_mode = 'RESTRICTED' WHERE id = 5");

        migrate();
        migrate();

        assertThat(modes()).containsExactly("ALL", "RESTRICTED", "RESTRICTED", "RESTRICTED",
                "RESTRICTED", "RESTRICTED", "ALL", "RESTRICTED");
        assertNewDefaultsAndValidation();
    }

    @Test
    void freshStartupHasRestrictedDefaultsWithoutHardcodedSeedRows() {
        new MasterSchemaInitializer(dataSource).init();
        new MasterSchemaInitializer(dataSource).init();

        assertThat(grantRows()).isEmpty();
        assertThat(modes()).isEmpty();
        assertNewDefaultsAndValidation();
    }

    private void migrate() {
        new ResourceDatabasePopulator(new ClassPathResource("sql/datasource_visible_scope_migration.sql"))
                .execute(dataSource);
    }

    private void assertNewDefaultsAndValidation() {
        jdbc.update("""
                INSERT INTO t_datasource_config (id, name, url, username, password)
                VALUES (1000, 'new', 'jdbc:postgresql://localhost/example', 'example', 'encrypted')
                """);
        assertThat(jdbc.queryForObject("SELECT visible_scope_mode FROM t_datasource_config WHERE id = 1000",
                String.class)).isEqualTo("RESTRICTED");
        jdbc.update("UPDATE t_datasource_config SET visible_scope_mode = 'ALL' WHERE id = 1000");
        assertThat(jdbc.queryForObject("SELECT visible_scope_mode FROM t_datasource_config WHERE id = 1000",
                String.class)).isEqualTo("ALL");
        for (String invalid : new String[]{"all", "", "NONE", "ALL "}) {
            assertThatThrownBy(() -> jdbc.update(
                    "UPDATE t_datasource_config SET visible_scope_mode = ? WHERE id = 1000", invalid))
                    .isInstanceOf(DataIntegrityViolationException.class);
        }
        assertThatThrownBy(() -> jdbc.update(
                "UPDATE t_datasource_config SET visible_scope_mode = NULL WHERE id = 1000"))
                .isInstanceOf(DataIntegrityViolationException.class);
        jdbc.update("INSERT INTO t_datasource_table_grant (id, datasource_id, table_pattern) VALUES (1000, 1000, 'new_')");
        assertThat(jdbc.queryForObject("SELECT tenant_id FROM t_datasource_table_grant WHERE id = 1000",
                String.class)).isEqualTo("__datasource__");
    }

    private List<String> modes() {
        return jdbc.queryForList("SELECT visible_scope_mode FROM t_datasource_config ORDER BY id", String.class);
    }

    private List<String> grantRows() {
        return jdbc.queryForList("SELECT row_to_json(g)::text FROM t_datasource_table_grant g ORDER BY id", String.class);
    }

    private List<String> datasourceRowsWithoutMode() {
        return jdbc.queryForList("SELECT (to_jsonb(ds) - 'visible_scope_mode')::text FROM t_datasource_config ds ORDER BY id",
                String.class);
    }

    private void createLegacySchema() {
        jdbc.execute("""
                CREATE TABLE t_datasource_config (
                    id BIGSERIAL PRIMARY KEY, name VARCHAR(200) NOT NULL, url VARCHAR(500) NOT NULL,
                    username VARCHAR(200) NOT NULL, password VARCHAR(500) NOT NULL, schema_name VARCHAR(128),
                    source_type VARCHAR(64) NOT NULL DEFAULT 'sap_b1_hana', description VARCHAR(1000),
                    status SMALLINT NOT NULL DEFAULT 1, created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, deleted SMALLINT NOT NULL DEFAULT 0
                )
                """);
        jdbc.execute("""
                CREATE TABLE t_datasource_table_grant (
                    id BIGSERIAL PRIMARY KEY, tenant_id VARCHAR(100) NOT NULL, datasource_id BIGINT NOT NULL,
                    schema_name VARCHAR(128), table_pattern VARCHAR(200) NOT NULL,
                    pattern_type VARCHAR(32) NOT NULL DEFAULT 'PREFIX', case_sensitive BOOLEAN NOT NULL DEFAULT FALSE,
                    status SMALLINT NOT NULL DEFAULT 1, created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, deleted SMALLINT NOT NULL DEFAULT 0
                )
                """);
        jdbc.update("""
                INSERT INTO t_datasource_config (id, name, url, username, password, status, deleted)
                SELECT id, 'legacy-' || id, 'jdbc:postgresql://localhost/example', 'example', 'encrypted',
                    CASE WHEN id = 7 THEN 0 ELSE 1 END, CASE WHEN id = 6 THEN 1 ELSE 0 END
                FROM unnest(ARRAY[1, 2, 3, 4, 5, 6, 7, 15]) AS id
                """);
        jdbc.update("""
                INSERT INTO t_datasource_table_grant
                    (id, tenant_id, datasource_id, schema_name, table_pattern, status, deleted)
                VALUES (101, 'legacy-a', 1, 'public', 'active_', 1, 0),
                       (102, 'legacy-b', 2, 'public', 'disabled_', 0, 0),
                       (103, 'legacy-c', 3, 'public', 'deleted_', 1, 1),
                       (104, 'legacy-d', 4, 'public', 'both_', 0, 1),
                       (105, 'legacy-e', 6, 'public', 'deleted_datasource_', 1, 0),
                       (115, 'hankel', 15, 'public', 'hankel_', 1, 1)
                """);
    }
}
