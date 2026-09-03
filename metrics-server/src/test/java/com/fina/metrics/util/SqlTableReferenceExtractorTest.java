package com.fina.metrics.util;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class SqlTableReferenceExtractorTest {

    @Test
    void extractsPlainSchemaQualifiedAndSqlServerBracketTables() {
        assertThat(SqlTableReferenceExtractor.extract("""
                SELECT TOP 10 *
                FROM [public].[hankel_orders] o
                JOIN "sales"."hankel_items" i ON i.id = o.item_id
                JOIN hankel_customers c ON c.id = o.customer_id
                """))
                .extracting(SqlTableReferenceExtractor.TableReference::original)
                .containsExactly("public.hankel_orders", "sales.hankel_items", "hankel_customers");
    }

    @Test
    void ignoresCteNamesButKeepsTablesInsideCte() {
        assertThat(SqlTableReferenceExtractor.extract("""
                WITH base AS (
                  SELECT * FROM public.hankel_orders
                )
                SELECT * FROM base
                """))
                .extracting(SqlTableReferenceExtractor.TableReference::original)
                .containsExactly("public.hankel_orders");
    }

    @Test
    void extractsCommaJoinedTables() {
        assertThat(SqlTableReferenceExtractor.extract("""
                SELECT *
                FROM hankel_orders o, hankel_customers c
                WHERE o.customer_id = c.id
                """))
                .extracting(SqlTableReferenceExtractor.TableReference::tableName)
                .containsExactly("hankel_orders", "hankel_customers");
    }
}
