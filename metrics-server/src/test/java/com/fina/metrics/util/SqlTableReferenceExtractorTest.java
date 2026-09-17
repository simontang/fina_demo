package com.fina.metrics.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

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

    @Test
    void preservesOuterCommaContextAcrossNestedQueriesAndJoinConditions() {
        assertThat(SqlTableReferenceExtractor.extract("""
                SELECT * FROM allowed a
                JOIN (SELECT * FROM nested n, nested_extra e) b ON b.id = a.id
                AND EXISTS (SELECT 1 FROM condition_table), hidden h
                WHERE h.id = a.id
                """))
                .extracting(SqlTableReferenceExtractor.TableReference::tableName)
                .containsExactlyInAnyOrder("allowed", "nested", "nested_extra", "condition_table", "hidden");
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "WITH hidden AS (SELECT * FROM hidden) SELECT * FROM hidden",
            "WITH a AS (SELECT * FROM hidden), hidden AS (SELECT 1) SELECT * FROM a",
            "SELECT * FROM (WITH hidden AS (SELECT 1) SELECT * FROM hidden) x, hidden",
            "SELECT (WITH hidden AS (SELECT 1) SELECT * FROM hidden) FROM hidden",
            "WITH \"HIDDEN\" AS (SELECT 1) SELECT * FROM hidden",
            "WITH hidden AS (SELECT 1) SELECT * FROM public.hidden"
    })
    void doesNotHideRealTablesUsingOutOfScopeOrNonMatchingCteNames(String sql) {
        assertThat(SqlTableReferenceExtractor.extract(sql))
                .extracting(SqlTableReferenceExtractor.TableReference::tableName)
                .contains("hidden");
    }

    @Test
    void scopesCtesSequentiallyAndKeepsAliasesOutOfReferences() {
        assertThat(SqlTableReferenceExtractor.extract("""
                WITH base(id) AS (SELECT id FROM public.orders),
                     totals AS (SELECT COUNT(*) AS total FROM base)
                SELECT TOP (10) t.total, x.id FROM totals AS t
                CROSS JOIN (SELECT id FROM base) AS x
                """))
                .extracting(SqlTableReferenceExtractor.TableReference::original)
                .containsExactly("public.orders");
    }

    @Test
    void supportsRecursiveCtesWithoutHidingQualifiedBaseTables() {
        assertThat(SqlTableReferenceExtractor.extract("""
                WITH RECURSIVE tree(id) AS (
                  SELECT id FROM public.nodes
                  UNION ALL
                  SELECT n.id FROM public.nodes n JOIN tree t ON n.parent_id = t.id
                ) SELECT * FROM tree
                """))
                .extracting(SqlTableReferenceExtractor.TableReference::original)
                .containsExactly("public.nodes");
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "SELECT (SELECT id FROM hidden) FROM allowed",
            "SELECT * FROM allowed WHERE (SELECT id FROM hidden) IS NULL",
            "SELECT * FROM allowed WHERE (SELECT flag FROM hidden) IS TRUE",
            "SELECT * FROM allowed ORDER BY (SELECT id FROM hidden)",
            "SELECT COUNT(*) FROM allowed GROUP BY (SELECT id FROM hidden)",
            "SELECT COUNT(*) FROM allowed HAVING MAX(id) > (SELECT id FROM hidden)",
            "SELECT SUM((SELECT id FROM hidden)) OVER () FROM allowed",
            "SELECT ROW_NUMBER() OVER (ORDER BY (SELECT id FROM hidden)) FROM allowed",
            "SELECT SUM(id) FILTER (WHERE EXISTS (SELECT 1 FROM hidden)) FROM allowed",
            "SELECT CAST((SELECT id FROM hidden) AS INTEGER) FROM allowed",
            "SELECT * FROM allowed WHERE id = ANY (SELECT id FROM hidden)",
            "SELECT * FROM allowed UNION ALL SELECT * FROM hidden",
            "SELECT * FROM allowed CROSS JOIN LATERAL (SELECT * FROM hidden) x"
    })
    void visitsEverySupportedExpressionAndSubqueryContext(String sql) {
        assertThat(SqlTableReferenceExtractor.extract(sql))
                .extracting(SqlTableReferenceExtractor.TableReference::tableName)
                .contains("allowed", "hidden");
    }

    @Test
    void preservesQuotedIdentifierContentsWithoutTreatingThemAsKeywords() {
        assertThat(SqlTableReferenceExtractor.extract("""
                SELECT TOP 10 [delete] FROM [sales].[order]]detail] AS [from]
                JOIN "Sales"."order detail" AS "join" ON 1 = 1
                """))
                .extracting(SqlTableReferenceExtractor.TableReference::original)
                .containsExactly("sales.order]detail", "Sales.order detail");
        assertThat(SqlTableReferenceExtractor.extract("SELECT * FROM \"sales.orders\""))
                .containsExactly(new SqlTableReferenceExtractor.TableReference(null, "sales.orders", "sales.orders"));
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "SELECT * FROM otherdb.public.allowed",
            "SELECT * FROM [otherdb].[dbo].[allowed]",
            "SELECT * FROM server.otherdb.dbo.allowed",
            "SELECT * FROM otherdb..allowed",
            "SELECT * FROM allowed@remote",
            "SELECT * FROM ONLY hidden",
            "SELECT * FROM ONLY (hidden)",
            "SELECT * FROM LATERAL read_hidden() x",
            "SELECT * FROM read_hidden() x",
            "SELECT * FROM allowed CROSS APPLY dbo.read_hidden() x",
            "SELECT * FROM OPENQUERY(remote, 'SELECT * FROM hidden')",
            "SELECT * FROM OPENROWSET('provider', 'connection', 'SELECT * FROM hidden')",
            "SELECT * FROM allowed TABLESAMPLE SYSTEM (10)",
            "SELECT * FROM \"a\"\"b\"",
            "SELECT values[(SELECT id FROM hidden)] FROM allowed",
            "SELECT values[read_hidden()] FROM allowed",
            "SELECT values[id::custom_type] FROM allowed",
            "SELECT * FROM",
            "SELECT * FROM allowed,",
            "SELECT * FROM (SELECT * FROM hidden",
            "SELECT * FROM allowed; SELECT * FROM hidden",
            "SELECT * INTO copied FROM allowed",
            "WITH x AS (DELETE FROM hidden RETURNING *) SELECT * FROM x"
    })
    void rejectsUnsupportedAmbiguousOrWritableSql(String sql) {
        assertThatThrownBy(() -> SqlTableReferenceExtractor.extract(sql))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "SELECT query_to_xml('SELECT * FROM hidden', true, false, '') FROM allowed",
            "SELECT pg_read_file('/etc/passwd') FROM allowed",
            "SELECT pg_catalog.pg_read_binary_file('/etc/passwd') FROM allowed",
            "SELECT dblink_exec('connection', 'DELETE FROM hidden') FROM allowed",
            "SELECT nextval('sequence') FROM allowed",
            "SELECT set_config('search_path', 'hidden', false) FROM allowed",
            "SELECT dbo.read_hidden() FROM allowed",
            "SELECT read_hidden() FROM allowed",
            "SELECT \"sum\"(id) FROM allowed",
            "SELECT SUM(id) FROM allowed ORDER BY read_hidden()",
            "SELECT * FROM allowed WHERE read_hidden() IS NULL",
            "SELECT SUM(id) OVER (ORDER BY read_hidden()) FROM allowed",
            "SELECT CAST('payload' AS custom_type) FROM allowed"
    })
    void rejectsFunctionsAndCastsWhoseIndirectAccessCannotBeScoped(String sql) {
        assertThatThrownBy(() -> SqlTableReferenceExtractor.extract(sql))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "SELECT SUM(COALESCE(amount, 0)), COUNT(DISTINCT id) FROM public.orders",
            "SELECT TOP (10) [id], ISNULL([amount], 0) FROM [dbo].[orders] o ORDER BY [id]",
            "SELECT YEAR(\"DocDate\"), MONTH(\"DocDate\"), SUM(\"DocTotal\") FROM \"ORDR\" GROUP BY YEAR(\"DocDate\"), MONTH(\"DocDate\")",
            "SELECT DATE_TRUNC('month', created_at), EXTRACT(YEAR FROM created_at) FROM public.orders",
            "SELECT CASE WHEN amount > 0 THEN ROUND(amount, 2) ELSE 0 END FROM orders",
            "SELECT CAST(amount AS DECIMAL(18, 2)), amount::numeric FROM orders",
            "SELECT ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY id), SUM(amount) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM orders",
            "SELECT * FROM (orders o JOIN items i ON o.id = i.order_id) JOIN customers c ON c.id = o.customer_id",
            "WITH \"base\" AS (SELECT * FROM orders) SELECT * FROM \"base\""
    })
    void keepsCommonPostgresHanaAndSqlServerQueriesValid(String sql) {
        assertThat(SqlTableReferenceExtractor.extract(sql)).isNotEmpty();
    }

    @Test
    void permitsCatalogReferencesForCallerPolicyAndPostgresArrays() {
        assertThat(SqlTableReferenceExtractor.extract("SELECT ARRAY[1,2,3] FROM information_schema.tables"))
                .containsExactly(new SqlTableReferenceExtractor.TableReference(
                        "information_schema", "tables", "information_schema.tables"));
        assertThat(SqlTableReferenceExtractor.extract("SELECT ARRAY[1,2,3]")).isEmpty();
        assertThat(SqlTableReferenceExtractor.extract("SELECT ARRAY[(SELECT id FROM hidden)]"))
                .extracting(SqlTableReferenceExtractor.TableReference::tableName).containsExactly("hidden");
        assertThatThrownBy(() -> SqlTableReferenceExtractor.extract("SELECT ARRAY[read_hidden()]"))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
