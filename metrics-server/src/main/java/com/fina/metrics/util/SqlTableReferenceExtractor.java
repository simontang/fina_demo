package com.fina.metrics.util;

import net.sf.jsqlparser.JSQLParserException;
import net.sf.jsqlparser.expression.*;
import net.sf.jsqlparser.expression.operators.relational.*;
import net.sf.jsqlparser.parser.CCJSqlParserUtil;
import net.sf.jsqlparser.schema.Column;
import net.sf.jsqlparser.schema.Table;
import net.sf.jsqlparser.statement.Statement;
import net.sf.jsqlparser.statement.select.*;
import org.springframework.util.StringUtils;

import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public final class SqlTableReferenceExtractor {

    private static final Set<Class<?>> LITERAL_TYPES = Set.of(
            NullValue.class, LongValue.class, DoubleValue.class, StringValue.class,
            DateValue.class, TimeValue.class, TimestampValue.class, DateTimeLiteralExpression.class,
            TimeKeyExpression.class, HexValue.class, JdbcParameter.class, JdbcNamedParameter.class);
    private static final Set<String> BINARY_TYPES = Set.of(
            "Addition", "Subtraction", "Multiplication", "Division", "IntegerDivision", "Modulo",
            "Concat", "AndExpression", "OrExpression", "XorExpression", "EqualsTo", "NotEqualsTo",
            "GreaterThan", "GreaterThanEquals", "MinorThan", "MinorThanEquals", "LikeExpression",
            "IsDistinctExpression", "SimilarToExpression", "RegExpMatchOperator", "RegExpMySQLOperator",
            "BitwiseAnd", "BitwiseOr", "BitwiseXor", "BitwiseLeftShift", "BitwiseRightShift");
    private static final Set<String> CAST_TYPES = Set.of(
            "boolean", "bool", "bit", "tinyint", "smallint", "int", "integer", "bigint",
            "decimal", "numeric", "real", "float", "double", "double precision", "smallmoney", "money",
            "char", "character", "varchar", "character varying", "nchar", "nvarchar", "text", "ntext",
            "date", "time", "timestamp", "datetime", "datetime2", "smalldatetime", "datetimeoffset",
            "binary", "varbinary", "bytea");

    private SqlTableReferenceExtractor() {
    }

    /**
     * Extracts direct references from a deliberately restricted SELECT grammar.
     * Unknown syntax/functions are rejected, never treated as having no references.
     * Views, synonyms, implicit casts/operators and function resolution still require trusted
     * database configuration and a least-privilege, read-only database account.
     */
    public static Set<TableReference> extract(String sql) {
        String parserSql = ReadOnlySqlValidator.validateAndPrepare(sql);
        try {
            // JSqlParser 4.6 is already supplied by MyBatis Plus. Do not use TablesNamesFinder:
            // its CTE names are global and its expression traversal omits several SQL clauses.
            Statement statement = CCJSqlParserUtil.parse(parserSql, parser -> parser.withSquareBracketQuotation(false));
            if (!(statement instanceof Select select)) {
                throw unsupported("non-SELECT statement");
            }
            Walker walker = new Walker();
            walker.query(select.getWithItemsList(), select.getSelectBody(), Set.of());
            return walker.references;
        } catch (JSQLParserException e) {
            throw new IllegalArgumentException("SQL scope cannot validate unsupported or malformed SQL", e);
        }
    }

    private static final class Walker {
        private final Set<TableReference> references = new LinkedHashSet<>();

        private void query(List<WithItem> withItems, SelectBody body, Set<String> inherited) {
            Set<String> visible = new LinkedHashSet<>(inherited);
            Set<String> declared = new LinkedHashSet<>();
            boolean recursive = withItems != null && withItems.stream().anyMatch(WithItem::isRecursive);
            if (withItems != null) {
                for (WithItem item : withItems) {
                    String key = cteKey(item.getName());
                    if (!declared.add(key) || item.isUseValues() || item.getSubSelect() == null) {
                        throw unsupported("CTE definition");
                    }
                    // Non-recursive CTEs see earlier CTEs, not themselves or later siblings.
                    if (recursive) {
                        visible.add(key);
                    }
                    subquery(item.getSubSelect(), visible);
                    visible.add(key);
                }
            }
            body(body, visible);
        }

        private void subquery(SubSelect select, Set<String> ctes) {
            query(select.getWithItemsList(), select.getSelectBody(), ctes);
        }

        private void body(SelectBody body, Set<String> ctes) {
            if (body instanceof PlainSelect select) {
                if (select.getIntoTables() != null || select.isForUpdate()
                        || select.getForUpdateTable() != null || select.getOracleHierarchical() != null
                        || select.getOracleHint() != null || select.getKsqlWindow() != null
                        || select.isEmitChanges() || select.getForXmlPath() != null
                        || select.getWithIsolation() != null || select.getOptimizeFor() != null
                        || select.getSkip() != null || select.getFirst() != null) {
                    throw unsupported("SELECT modifier");
                }
                selectItems(select.getSelectItems(), ctes);
                from(select.getFromItem(), ctes);
                joins(select.getJoins(), ctes);
                expression(select.getWhere(), ctes);
                expression(select.getHaving(), ctes);
                if (select.getGroupBy() != null) {
                    items(select.getGroupBy().getGroupByExpressionList(), ctes);
                    List<?> groupingSets = select.getGroupBy().getGroupingSets();
                    if (groupingSets != null && !groupingSets.isEmpty()) {
                        throw unsupported("GROUPING SETS");
                    }
                }
                if (select.getDistinct() != null) {
                    selectItems(select.getDistinct().getOnSelectItems(), ctes);
                }
                orderBy(select.getOrderByElements(), ctes);
                if (select.getTop() != null) {
                    expression(select.getTop().getExpression(), ctes);
                }
                limit(select.getLimit(), select.getOffset(), ctes);
                if (select.getWindowDefinitions() != null) {
                    for (WindowDefinition window : select.getWindowDefinitions()) {
                        window(window, ctes);
                    }
                }
            } else if (body instanceof SetOperationList set) {
                if (set.getWithIsolation() != null) {
                    throw unsupported("set operation isolation");
                }
                for (SelectBody branch : set.getSelects()) {
                    body(branch, ctes);
                }
                orderBy(set.getOrderByElements(), ctes);
                limit(set.getLimit(), set.getOffset(), ctes);
            } else {
                throw unsupported(body);
            }
        }

        private void selectItems(List<SelectItem> selectItems, Set<String> ctes) {
            if (selectItems == null) {
                return;
            }
            for (SelectItem item : selectItems) {
                if (item instanceof SelectExpressionItem value) {
                    expression(value.getExpression(), ctes);
                } else if (item instanceof AllTableColumns all) {
                    localName(all.getTable());
                } else if (!(item instanceof AllColumns)) {
                    throw unsupported(item);
                }
            }
        }

        private void from(FromItem item, Set<String> ctes) {
            if (item == null) {
                return;
            }
            if (item.getPivot() != null || item.getUnPivot() != null) {
                throw unsupported("PIVOT");
            }
            if (item instanceof Table table) {
                localName(table);
                if (table.getIndexHint() != null || table.getSqlServerHints() != null) {
                    throw unsupported("table hint");
                }
                String schema = identifier(table.getSchemaName());
                String rawName = table.getNameParts().get(0);
                String name = identifier(rawName);
                if (schema == null && ctes.contains(cteKey(rawName))) {
                    return;
                }
                references.add(new TableReference(schema, name, schema == null ? name : schema + "." + name));
            } else if (item instanceof SubSelect select) {
                subquery(select, ctes);
            } else if (item instanceof LateralSubSelect lateral) {
                subquery(lateral.getSubSelect(), ctes);
            } else if (item instanceof ParenthesisFromItem parenthesis) {
                from(parenthesis.getFromItem(), ctes);
            } else if (item instanceof SubJoin join) {
                from(join.getLeft(), ctes);
                joins(join.getJoinList(), ctes);
            } else {
                // Includes table-valued functions, VALUES sources, remote providers and extensions.
                throw unsupported(item);
            }
        }

        private void joins(List<Join> joins, Set<String> ctes) {
            if (joins == null) {
                return;
            }
            for (Join join : joins) {
                if (join.getJoinWindow() != null || join.isGlobal()) {
                    throw unsupported("JOIN modifier");
                }
                from(join.getRightItem(), ctes);
                expressions(join.getOnExpressions(), ctes);
            }
        }

        private void expression(Expression expression, Set<String> ctes) {
            if (expression == null || LITERAL_TYPES.contains(expression.getClass()) || expression instanceof AllColumns) {
                return;
            }
            if (expression instanceof Column column) {
                if (column.getTable() != null && column.getTable().getName() != null) {
                    localName(column.getTable());
                }
                if ("nextval".equalsIgnoreCase(identifier(column.getColumnName()))
                        || "currval".equalsIgnoreCase(identifier(column.getColumnName()))) {
                    throw unsupported("sequence access");
                }
            } else if (expression instanceof AllTableColumns all) {
                localName(all.getTable());
            } else if (expression instanceof SubSelect select) {
                subquery(select, ctes);
            } else if (expression instanceof BinaryExpression binary && BINARY_TYPES.contains(binary.getClass().getSimpleName())) {
                expression(binary.getLeftExpression(), ctes);
                expression(binary.getRightExpression(), ctes);
                if (binary instanceof LikeExpression like) {
                    expression(like.getEscape(), ctes);
                }
            } else if (expression instanceof Function function) {
                ReadOnlySqlValidator.validateRestrictedFunction(function.getName());
                if (function.getAttribute() != null || function.getAttributeName() != null || function.getKeep() != null) {
                    throw unsupported("function modifier");
                }
                items(function.getParameters(), ctes);
                if (function.getNamedParameters() != null) {
                    expressions(function.getNamedParameters().getExpressions(), ctes);
                }
                orderBy(function.getOrderByElements(), ctes);
            } else if (expression instanceof AnalyticExpression analytic) {
                ReadOnlySqlValidator.validateRestrictedFunction(analytic.getName());
                if (analytic.getKeep() != null) {
                    throw unsupported("analytic KEEP");
                }
                expression(analytic.getExpression(), ctes);
                expression(analytic.getOffset(), ctes);
                expression(analytic.getDefaultValue(), ctes);
                expression(analytic.getFilterExpression(), ctes);
                orderBy(analytic.getFuncOrderBy(), ctes);
                window(analytic.getWindowDefinition(), ctes);
            } else if (expression instanceof Parenthesis parenthesis) {
                expression(parenthesis.getExpression(), ctes);
            } else if (expression instanceof SignedExpression signed) {
                expression(signed.getExpression(), ctes);
            } else if (expression instanceof NotExpression not) {
                expression(not.getExpression(), ctes);
            } else if (expression instanceof IsNullExpression isNull) {
                expression(isNull.getLeftExpression(), ctes);
            } else if (expression instanceof IsBooleanExpression isBoolean) {
                expression(isBoolean.getLeftExpression(), ctes);
            } else if (expression instanceof ExistsExpression exists) {
                expression(exists.getRightExpression(), ctes);
            } else if (expression instanceof InExpression in) {
                expression(in.getLeftExpression(), ctes);
                expression(in.getRightExpression(), ctes);
                items(in.getRightItemsList(), ctes);
            } else if (expression instanceof Between between) {
                expression(between.getLeftExpression(), ctes);
                expression(between.getBetweenExpressionStart(), ctes);
                expression(between.getBetweenExpressionEnd(), ctes);
            } else if (expression instanceof CaseExpression choice) {
                expression(choice.getSwitchExpression(), ctes);
                expressions(choice.getWhenClauses(), ctes);
                expression(choice.getElseExpression(), ctes);
            } else if (expression instanceof WhenClause when) {
                expression(when.getWhenExpression(), ctes);
                expression(when.getThenExpression(), ctes);
            } else if (expression instanceof AnyComparisonExpression any) {
                expression(any.getSubSelect(), ctes);
                items(any.getItemsList(), ctes);
            } else if (expression instanceof CastExpression cast) {
                if (cast.getType() == null
                        || !CAST_TYPES.contains(cast.getType().getDataType().toLowerCase(Locale.ROOT))) {
                    throw unsupported("cast type");
                }
                expression(cast.getLeftExpression(), ctes);
                expression(cast.getRowConstructor(), ctes);
            } else if (expression instanceof ExtractExpression extract) {
                expression(extract.getExpression(), ctes);
            } else if (expression instanceof IntervalExpression interval) {
                expression(interval.getExpression(), ctes);
            } else if (expression instanceof RowConstructor row) {
                items(row.getExprList(), ctes);
            } else if (expression instanceof ArrayConstructor array) {
                expressions(array.getExpressions(), ctes);
            } else if (expression instanceof TimezoneExpression timezone) {
                expression(timezone.getLeftExpression(), ctes);
                expressions(timezone.getTimezoneExpressions(), ctes);
            } else {
                throw unsupported(expression);
            }
        }

        private void items(ItemsList items, Set<String> ctes) {
            if (items == null) {
                return;
            }
            if (items instanceof ExpressionList list) {
                expressions(list.getExpressions(), ctes);
            } else if (items instanceof SubSelect select) {
                subquery(select, ctes);
            } else {
                throw unsupported(items);
            }
        }

        private void expressions(Collection<? extends Expression> expressions, Set<String> ctes) {
            if (expressions != null) {
                for (Expression expression : expressions) {
                    expression(expression, ctes);
                }
            }
        }

        private void orderBy(List<OrderByElement> order, Set<String> ctes) {
            if (order != null) {
                for (OrderByElement element : order) {
                    expression(element.getExpression(), ctes);
                }
            }
        }

        private void limit(Limit limit, Offset offset, Set<String> ctes) {
            if (limit != null) {
                expression(limit.getOffset(), ctes);
                expression(limit.getRowCount(), ctes);
            }
            if (offset != null) {
                expression(offset.getOffset(), ctes);
            }
        }

        private void window(WindowDefinition window, Set<String> ctes) {
            if (window == null) {
                return;
            }
            items(window.getPartitionExpressionList(), ctes);
            orderBy(window.getOrderByElements(), ctes);
            WindowElement element = window.getWindowElement();
            if (element != null) {
                windowOffset(element.getOffset(), ctes);
                if (element.getRange() != null) {
                    windowOffset(element.getRange().getStart(), ctes);
                    windowOffset(element.getRange().getEnd(), ctes);
                }
            }
        }

        private void windowOffset(WindowOffset offset, Set<String> ctes) {
            if (offset != null) {
                expression(offset.getExpression(), ctes);
            }
        }
    }

    private static void localName(Table table) {
        List<String> parts = table.getNameParts();
        if (parts.isEmpty() || parts.size() > 2 || parts.stream().anyMatch(part -> !StringUtils.hasText(part)
                || (!isQuoted(part) && (part.contains("@") || part.startsWith("#"))))) {
            throw unsupported("cross-database, remote, variable or incomplete table reference");
        }
    }

    private static String cteKey(String name) {
        // Without a dialect, do not equate quoted names with folded names: PG and HANA fold
        // in opposite directions. A conservative extra reference is safer than hiding a table.
        return isQuoted(name) ? "quoted:" + identifier(name) : "plain:" + name.toLowerCase(Locale.ROOT);
    }

    private static boolean isQuoted(String name) {
        return name != null && (name.startsWith("\"") || name.startsWith("[") || name.startsWith("`"));
    }

    private static String identifier(String name) {
        if (!isQuoted(name)) {
            return name;
        }
        char closing = name.charAt(0) == '[' ? ']' : name.charAt(0);
        if (name.length() < 2 || name.charAt(name.length() - 1) != closing) {
            throw unsupported("identifier");
        }
        return name.substring(1, name.length() - 1).replace("" + closing + closing, "" + closing);
    }

    private static IllegalArgumentException unsupported(Object node) {
        String detail = node instanceof String text ? text : node == null ? "missing node" : node.getClass().getSimpleName();
        return new IllegalArgumentException("SQL scope cannot validate unsupported " + detail);
    }

    public record TableReference(String schemaName, String tableName, String original) {}
}
