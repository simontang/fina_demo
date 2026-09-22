package com.fina.platform.bo;

import com.fina.platform.bo.BusinessObjectDtos.FieldDefinition;
import com.fina.platform.bo.BusinessObjectDtos.IndexDefinition;
import com.fina.platform.exception.ApiException;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class BusinessObjectSqlSupportTest {

    @Test
    void createTableSqlUsesSafePostgresIdentifiers() {
        List<FieldDefinition> fields = BusinessObjectSqlSupport.normalizeFields(List.of(
                new FieldDefinition("customer_name", "string", true, 120, null, null, null),
                new FieldDefinition("amount", "decimal", false, null, 18, 2, null),
                new FieldDefinition("profile", "json", false, null, null, null, null)
        ));

        String sql = BusinessObjectSqlSupport.createTableSql("bo_customer", fields);

        assertThat(sql)
                .contains("CREATE TABLE IF NOT EXISTS \"bo_customer\"")
                .contains("\"customer_name\" VARCHAR(120)")
                .contains("\"amount\" NUMERIC(18,2)")
                .contains("\"profile\" JSONB")
                .doesNotContain("NOT NULL, \"customer_name\"");
    }

    @Test
    void invalidIdentifiersAreRejectedBeforeSqlGeneration() {
        assertThatThrownBy(() -> BusinessObjectSqlSupport.tableNameFor("Customer;drop table x"))
                .isInstanceOf(ApiException.class)
                .hasMessageContaining("objectKey must match");

        assertThatThrownBy(() -> BusinessObjectSqlSupport.normalizeFields(List.of(
                new FieldDefinition("created_at", "string", false, null, null, null, null))))
                .isInstanceOf(ApiException.class)
                .hasMessageContaining("reserved");
    }

    @Test
    void alterOnlyAllowsAdditiveSchemaEvolutionInV1() {
        List<FieldDefinition> oldFields = BusinessObjectSqlSupport.normalizeFields(List.of(
                new FieldDefinition("name", "string", true, 255, null, null, null)
        ));
        List<FieldDefinition> newFields = BusinessObjectSqlSupport.normalizeFields(List.of(
                new FieldDefinition("name", "string", true, 255, null, null, null),
                new FieldDefinition("email", "string", false, 255, null, null, null)
        ));

        assertThat(BusinessObjectSqlSupport.alterTableSql("bo_customer", oldFields, newFields))
                .containsExactly("ALTER TABLE \"bo_customer\" ADD COLUMN IF NOT EXISTS \"email\" VARCHAR(255)");

        List<FieldDefinition> changedType = BusinessObjectSqlSupport.normalizeFields(List.of(
                new FieldDefinition("name", "text", true, null, null, null, null)
        ));
        assertThatThrownBy(() -> BusinessObjectSqlSupport.alterTableSql("bo_customer", oldFields, changedType))
                .isInstanceOf(ApiException.class)
                .hasMessageContaining("type changes");

        assertThatThrownBy(() -> BusinessObjectSqlSupport.alterTableSql("bo_customer", newFields, oldFields))
                .isInstanceOf(ApiException.class)
                .hasMessageContaining("field removal");
    }

    @Test
    void indexDefinitionsMustReferenceKnownFields() {
        List<FieldDefinition> fields = BusinessObjectSqlSupport.normalizeFields(List.of(
                new FieldDefinition("name", "string", false, null, null, null, null)
        ));

        assertThat(BusinessObjectSqlSupport.normalizeIndexes(List.of(
                new IndexDefinition("name_idx", List.of("name"), true)), fields))
                .hasSize(1)
                .first()
                .extracting(IndexDefinition::unique)
                .isEqualTo(true);

        assertThatThrownBy(() -> BusinessObjectSqlSupport.normalizeIndexes(List.of(
                new IndexDefinition("bad_idx", List.of("missing"), false)), fields))
                .isInstanceOf(ApiException.class)
                .hasMessageContaining("not defined");
    }

    @Test
    void uniqueIndexesArePartialOverLiveRows() {
        List<FieldDefinition> fields = BusinessObjectSqlSupport.normalizeFields(List.of(
                new FieldDefinition("customer_no", "string", true, 64, null, null, null),
                new FieldDefinition("tag_key", "string", true, 64, null, null, null)
        ));
        List<IndexDefinition> indexes = BusinessObjectSqlSupport.normalizeIndexes(List.of(
                new IndexDefinition("uk_tag_customer_key", List.of("customer_no", "tag_key"), true),
                new IndexDefinition("idx_tag_key", List.of("tag_key"), false)
        ), fields);

        List<String> sql = BusinessObjectSqlSupport.createIndexSql("bo_customer_tag", "customer_tag", indexes);

        assertThat(sql).containsSubsequence(
                "DROP INDEX IF EXISTS \"idx_customer_tag_uk_tag_customer_key\"",
                "CREATE UNIQUE INDEX IF NOT EXISTS \"idx_customer_tag_uk_tag_customer_key\" "
                        + "ON \"bo_customer_tag\" (\"customer_no\", \"tag_key\") WHERE \"deleted\" = 0");
        assertThat(sql).contains(
                "CREATE INDEX IF NOT EXISTS \"idx_customer_tag_idx_tag_key\" "
                        + "ON \"bo_customer_tag\" (\"tag_key\")");
        assertThat(sql).noneMatch(s -> s.contains("idx_tag_key") && s.contains("WHERE"));
    }

    @Test
    void deleteModeDefaultsToHardAndRejectsOtherValues() {
        assertThat(BusinessObjectSqlSupport.normalizeDeleteMode(null)).isEqualTo("hard");
        assertThat(BusinessObjectSqlSupport.normalizeDeleteMode("  ")).isEqualTo("hard");
        assertThat(BusinessObjectSqlSupport.normalizeDeleteMode("SOFT")).isEqualTo("soft");
        assertThat(BusinessObjectSqlSupport.normalizeDeleteMode("hard")).isEqualTo("hard");

        assertThatThrownBy(() -> BusinessObjectSqlSupport.normalizeDeleteMode("purge"))
                .isInstanceOf(ApiException.class)
                .hasMessageContaining("deleteMode must be");
    }
}
