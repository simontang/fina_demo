package com.fina.metrics.util;

import org.springframework.util.StringUtils;

import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;

public final class ReadOnlySqlValidator {

    private static final Pattern READ_START = Pattern.compile("^(select|with)\\b", Pattern.CASE_INSENSITIVE);
    private static final Pattern WRITE_KEYWORDS = Pattern.compile(
            "\\b(insert|update|delete|drop|alter|create|truncate|merge|call|execute|exec|grant|revoke|copy|"
                    + "vacuum|analyze|into|refresh|reindex|attach|detach|load|do|set|reset|lock)\\b",
            Pattern.CASE_INSENSITIVE);
    private static final Pattern SIDE_EFFECT_SYNTAX = Pattern.compile(
            "\\bfor\\s+(share|key)\\b|\\bnext\\s+value\\s+for\\b|@[\\w@$#]+\\s*=|:=",
            Pattern.CASE_INSENSITIVE);
    private static final Pattern DOLLAR_QUOTE = Pattern.compile("\\$(?:[A-Za-z_][A-Za-z_0-9]*)?\\$");
    private static final Pattern ARRAY_PREFIX = Pattern.compile("(?is).*\\barray\\s*$");

    // Restricted scopes cannot inspect UDF bodies. Keep this separate from broad SQL validation.
    private static final Set<String> RESTRICTED_FUNCTIONS = Set.of(
            "abs", "ceil", "ceiling", "floor", "round", "trunc", "mod", "power", "sqrt",
            "sign", "exp", "ln", "log", "log10", "sum", "avg", "min", "max", "count", "count_big",
            "stddev", "stddev_pop", "stddev_samp", "variance", "var_pop", "var_samp",
            "coalesce", "nullif", "ifnull", "isnull", "greatest", "least",
            "lower", "upper", "length", "char_length", "character_length", "len", "octet_length",
            "concat", "concat_ws", "substring", "substr", "left", "right", "trim", "ltrim", "rtrim",
            "replace", "reverse", "lpad", "rpad", "locate", "instr", "position",
            "to_char", "to_date", "to_timestamp", "to_varchar", "to_nvarchar", "to_decimal",
            "to_integer", "to_bigint", "to_double", "to_real",
            "year", "quarter", "month", "day", "dayofmonth", "dayofyear", "hour", "minute", "second",
            "date_trunc", "date_part", "dateadd", "datediff", "datepart", "datename",
            "add_days", "add_months", "add_years", "add_seconds", "days_between", "seconds_between",
            "months_between", "last_day", "now", "getdate", "getutcdate", "current_date",
            "current_time", "current_timestamp", "row_number", "rank", "dense_rank", "ntile",
            "lag", "lead", "first_value", "last_value", "nth_value", "percent_rank", "cume_dist",
            "string_agg", "listagg", "bool_and", "bool_or", "every");

    private ReadOnlySqlValidator() {
    }

    /**
     * Syntactic read-only guard, not a sandbox for functions, views, or database permissions.
     * Restricted datasource callers must also use {@link SqlTableReferenceExtractor#extract(String)}.
     */
    public static void validate(String sql) {
        validateAndPrepare(sql);
    }

    static String validateAndPrepare(String sql) {
        if (!StringUtils.hasText(sql)) {
            throw new IllegalArgumentException("customSql is required");
        }
        SqlText text = scan(sql);
        String stripped = text.stripped().trim();
        if (stripped.contains(";")) {
            throw new IllegalArgumentException("Only a single read-only SQL statement is allowed");
        }
        if (!READ_START.matcher(stripped).find()) {
            throw new IllegalArgumentException("Only SELECT or WITH read-only SQL is allowed");
        }
        if (WRITE_KEYWORDS.matcher(stripped).find() || SIDE_EFFECT_SYNTAX.matcher(stripped).find()) {
            throw new IllegalArgumentException("Write or DDL SQL is not allowed");
        }
        int depth = 0;
        for (int i = 0; i < stripped.length(); i++) {
            if (stripped.charAt(i) == '(') {
                depth++;
            } else if (stripped.charAt(i) == ')' && --depth < 0) {
                throw malformedSql();
            }
        }
        if (depth != 0) {
            throw malformedSql();
        }
        return text.parserSql();
    }

    static void validateRestrictedFunction(String name) {
        // Quoted/qualified spellings could resolve to a user-defined function with the same name.
        if (name == null || !name.matches("[A-Za-z_][A-Za-z_0-9]*")
                || !RESTRICTED_FUNCTIONS.contains(name.toLowerCase(Locale.ROOT))) {
            throw new IllegalArgumentException("SQL scope cannot validate executable function: " + name);
        }
    }

    private static SqlText scan(String sql) {
        StringBuilder out = new StringBuilder(sql.length());
        StringBuilder parserSql = new StringBuilder(sql.length());
        for (int i = 0; i < sql.length(); i++) {
            char c = sql.charAt(i);
            char next = i + 1 < sql.length() ? sql.charAt(i + 1) : '\0';
            if (c == '-' && next == '-') {
                while (i + 1 < sql.length() && sql.charAt(i + 1) != '\n' && sql.charAt(i + 1) != '\r') {
                    i++;
                }
                out.append(' ');
                parserSql.append(' ');
            } else if (c == '/' && next == '*') {
                if (i + 2 < sql.length() && sql.charAt(i + 2) == '!') {
                    throw malformedSql();
                }
                int depth = 1;
                i += 2;
                while (i < sql.length() && depth > 0) {
                    if (i + 1 < sql.length() && sql.charAt(i) == '/' && sql.charAt(i + 1) == '*') {
                        depth++;
                        i += 2;
                    } else if (i + 1 < sql.length() && sql.charAt(i) == '*' && sql.charAt(i + 1) == '/') {
                        depth--;
                        i += 2;
                    } else {
                        i++;
                    }
                }
                if (depth != 0) {
                    throw malformedSql();
                }
                i--;
                out.append(' ');
                parserSql.append(' ');
            } else if (c == '[' && ARRAY_PREFIX.matcher(out).matches()) {
                out.append(c);
                parserSql.append(c);
            } else if (c == '\'' || c == '"' || c == '[' || c == '`') {
                int start = i;
                char closing = c == '[' ? ']' : c;
                StringBuilder identifier = new StringBuilder();
                boolean closed = false;
                for (i++; i < sql.length(); i++) {
                    char value = sql.charAt(i);
                    // Backslash escaping depends on server settings and must not hide SQL tokens.
                    if (value == '\\') {
                        throw malformedSql();
                    }
                    if (value == closing) {
                        if (i + 1 < sql.length() && sql.charAt(i + 1) == closing) {
                            if (c == '"' || c == '`') {
                                // JSqlParser 4.6 can split escaped quotes into a table plus alias.
                                throw malformedSql();
                            }
                            identifier.append(closing);
                            i++;
                        } else {
                            closed = true;
                            break;
                        }
                    } else {
                        identifier.append(value);
                    }
                }
                if (!closed) {
                    throw malformedSql();
                }
                out.append(" ? ");
                if (c == '[' || c == '`') {
                    // SQL Server brackets conflict with PostgreSQL subscripts. Do not allow an
                    // expression/subquery/cast to disappear inside what the parser sees as a name.
                    if (c == '[' && !identifier.toString().matches("[\\p{L}\\p{N}_@$# .\\]\\-]+")) {
                        throw malformedSql();
                    }
                    parserSql.append('"').append(identifier.toString().replace("\"", "\"\"")).append('"');
                } else {
                    parserSql.append(sql, start, i + 1);
                }
            } else {
                if (c == '$' && DOLLAR_QUOTE.matcher(sql).region(i, sql.length()).lookingAt()) {
                    throw malformedSql();
                }
                out.append(c);
                parserSql.append(c);
            }
        }
        return new SqlText(out.toString(), parserSql.toString());
    }

    private record SqlText(String stripped, String parserSql) {}

    private static IllegalArgumentException malformedSql() {
        return new IllegalArgumentException("Malformed or unsupported SQL quoting, comment, or parentheses");
    }
}
