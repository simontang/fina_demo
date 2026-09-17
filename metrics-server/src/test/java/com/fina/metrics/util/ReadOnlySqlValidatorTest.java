package com.fina.metrics.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.assertj.core.api.Assertions.assertThatNoException;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ReadOnlySqlValidatorTest {

    @Test
    void acceptsSelectAndWithQueries() {
        assertThatNoException().isThrownBy(() ->
                ReadOnlySqlValidator.validate("SELECT \"CardCode\", SUM(\"DocTotal\") FROM \"ORDR\""));
        assertThatNoException().isThrownBy(() ->
                ReadOnlySqlValidator.validate("WITH base AS (SELECT 1 AS value) SELECT value FROM base"));
        assertThatNoException().isThrownBy(() ->
                ReadOnlySqlValidator.validate("SELECT TOP 10 [delete], [Order Date] FROM [Sales Order]"));
    }

    @Test
    void ignoresKeywordsInsideStringsAndComments() {
        assertThatNoException().isThrownBy(() ->
                ReadOnlySqlValidator.validate("SELECT 'delete from table' AS note -- drop table\nFROM \"ORDR\""));
    }

    @Test
    void rejectsWriteAndDdlSql() {
        assertThatThrownBy(() -> ReadOnlySqlValidator.validate("UPDATE OCRD SET CardName = 'x'"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Only SELECT or WITH");
        assertThatThrownBy(() -> ReadOnlySqlValidator.validate("WITH deleted AS (DELETE FROM OCRD RETURNING *) SELECT * FROM deleted"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Write or DDL");
    }

    @Test
    void rejectsMultiStatementSql() {
        assertThatThrownBy(() -> ReadOnlySqlValidator.validate("SELECT 1; SELECT 2"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("single read-only SQL statement");
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "SELECT * INTO new_table FROM source",
            "SELECT * INTO #tmp FROM source",
            "WITH x AS (SELECT * INTO copied FROM source) SELECT * FROM x",
            "WITH x AS (UPDATE source SET id = 1 RETURNING *) SELECT * FROM x",
            "WITH x AS (INSERT INTO source VALUES (1) RETURNING *) SELECT * FROM x",
            "SELECT * FROM source FOR UPDATE",
            "SELECT * FROM source FOR SHARE",
            "SELECT NEXT VALUE FOR sequence",
            "SELECT @value = id FROM source",
            "SELECT 1; SELECT 2",
            "SELECT 1 /* outer /* inner */ still outer */; DELETE FROM source",
            "selective value",
            "withhold value",
            "SELECT 'unterminated",
            "SELECT * FROM \"unterminated",
            "SELECT * FROM [unterminated",
            "SELECT 1 /* unterminated",
            "SELECT 1)"
    })
    void rejectsWritableOrMalformedLexicalForms(String sql) {
        assertThatThrownBy(() -> ReadOnlySqlValidator.validate(sql))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "SELECT [into], \"delete\", 'create; update' FROM [table]",
            "SELECT 'it''s a ; delete' FROM orders",
            "SELECT 1 /* outer /* nested */ end */ FROM orders",
            "SELECT dbo.custom_metric(id) FROM orders",
            "SELECT custom_metric(id) FROM orders",
            "SELECT TOP (10) [id] FROM [dbo].[orders]"
    })
    void preservesBroadValidatorDialectAndCustomFunctionCompatibility(String sql) {
        assertThatNoException().isThrownBy(() -> ReadOnlySqlValidator.validate(sql));
    }
}
