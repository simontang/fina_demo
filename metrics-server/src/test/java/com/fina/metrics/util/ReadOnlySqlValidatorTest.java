package com.fina.metrics.util;

import org.junit.jupiter.api.Test;

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
}
