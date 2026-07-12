package com.fina.cdp.util;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class SqlSafetyValidatorTest {

    private final SqlSafetyValidator validator = new SqlSafetyValidator();

    @Test
    void acceptsReadOnlySelectAndWithSql() {
        assertThatCode(() -> validator.validateReadOnly("select customer_id from retailcdp_customers"))
                .doesNotThrowAnyException();
        assertThatCode(() -> validator.validateReadOnly("""
                WITH active_customers AS (
                  SELECT customer_id FROM retailcdp_customers
                )
                SELECT * FROM active_customers
                """)).doesNotThrowAnyException();
    }

    @Test
    void rejectsWriteAndMultiStatementSql() {
        assertThatThrownBy(() -> validator.validateReadOnly("update retailcdp_customers set vip_score = 1"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("read-only");

        assertThatThrownBy(() -> validator.validateReadOnly("select * from retailcdp_customers; delete from retailcdp_customers"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("single statement");
    }
}
