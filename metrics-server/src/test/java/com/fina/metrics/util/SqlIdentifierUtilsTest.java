package com.fina.metrics.util;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class SqlIdentifierUtilsTest {

    @Test
    void parsesPlainAndDelimitedQualifiedNames() {
        assertThat(SqlIdentifierUtils.parseTableIdentifier(null, "public.hankel_sales"))
                .extracting(
                        SqlIdentifierUtils.TableIdentifier::schemaName,
                        SqlIdentifierUtils.TableIdentifier::tableName)
                .containsExactly("public", "hankel_sales");

        assertThat(SqlIdentifierUtils.parseTableIdentifier(null, "\"sales\".\"hankel.items\""))
                .extracting(
                        SqlIdentifierUtils.TableIdentifier::schemaName,
                        SqlIdentifierUtils.TableIdentifier::tableName)
                .containsExactly("sales", "hankel.items");

        assertThat(SqlIdentifierUtils.parseTableIdentifier(null, "[dbo].[OITM]"))
                .extracting(
                        SqlIdentifierUtils.TableIdentifier::schemaName,
                        SqlIdentifierUtils.TableIdentifier::tableName)
                .containsExactly("dbo", "OITM");
    }

    @Test
    void quotesQualifiedNamesPartByPart() {
        assertThat(SqlIdentifierUtils.quoteQualified("public.hankel_sales"))
                .isEqualTo("\"public\".\"hankel_sales\"");
        assertThat(SqlIdentifierUtils.quoteQualified("[dbo].[OITM]"))
                .isEqualTo("\"dbo\".\"OITM\"");
    }

    @Test
    void comparesQualifiedAndUnqualifiedTableNames() {
        assertThat(SqlIdentifierUtils.sameTableName("public.hankel_sales", "hankel_sales", false))
                .isTrue();
        assertThat(SqlIdentifierUtils.sameTableName("public.hankel_sales", "sales.hankel_sales", false))
                .isFalse();
    }

    @Test
    void rejectsCrossDatabaseAndConflictingSchemaIdentifiers() {
        assertThatThrownBy(() -> SqlIdentifierUtils.parseTableIdentifier(null, "other.public.hankel_sales"))
                .isInstanceOf(IllegalArgumentException.class).hasMessageContaining("Cross-database");
        assertThatThrownBy(() -> SqlIdentifierUtils.parseTableIdentifier("public", "private.hankel_sales"))
                .isInstanceOf(IllegalArgumentException.class).hasMessageContaining("Conflicting");
    }
}
