package com.fina.metrics.util;

import org.springframework.util.StringUtils;

import java.util.Locale;
import java.util.regex.Pattern;

public final class ReadOnlySqlValidator {

    private static final Pattern WRITE_KEYWORDS = Pattern.compile(
            "\\b(insert|update|delete|drop|alter|create|truncate|merge|call|execute|grant|revoke|copy|vacuum|analyze)\\b",
            Pattern.CASE_INSENSITIVE);

    private ReadOnlySqlValidator() {
    }

    public static void validate(String sql) {
        if (!StringUtils.hasText(sql)) {
            throw new IllegalArgumentException("customSql is required");
        }
        String stripped = stripCommentsAndLiterals(sql).trim();
        if (stripped.contains(";")) {
            throw new IllegalArgumentException("Only a single read-only SQL statement is allowed");
        }
        String lower = stripped.toLowerCase(Locale.ROOT);
        if (!(lower.startsWith("select") || lower.startsWith("with"))) {
            throw new IllegalArgumentException("Only SELECT or WITH read-only SQL is allowed");
        }
        if (WRITE_KEYWORDS.matcher(stripped).find()) {
            throw new IllegalArgumentException("Write or DDL SQL is not allowed");
        }
    }

    private static String stripCommentsAndLiterals(String sql) {
        StringBuilder out = new StringBuilder(sql.length());
        boolean inSingle = false;
        boolean inDouble = false;
        boolean inBracketIdentifier = false;
        boolean inLineComment = false;
        boolean inBlockComment = false;

        for (int i = 0; i < sql.length(); i++) {
            char c = sql.charAt(i);
            char next = i + 1 < sql.length() ? sql.charAt(i + 1) : '\0';

            if (inLineComment) {
                if (c == '\n' || c == '\r') {
                    inLineComment = false;
                    out.append(c);
                } else {
                    out.append(' ');
                }
                continue;
            }
            if (inBlockComment) {
                if (c == '*' && next == '/') {
                    inBlockComment = false;
                    out.append("  ");
                    i++;
                } else {
                    out.append(' ');
                }
                continue;
            }
            if (inSingle) {
                if (c == '\'' && next == '\'') {
                    out.append("  ");
                    i++;
                } else if (c == '\'') {
                    inSingle = false;
                    out.append(' ');
                } else {
                    out.append(' ');
                }
                continue;
            }
            if (inDouble) {
                if (c == '"' && next == '"') {
                    out.append("  ");
                    i++;
                } else if (c == '"') {
                    inDouble = false;
                    out.append(' ');
                } else {
                    out.append(' ');
                }
                continue;
            }
            if (inBracketIdentifier) {
                if (c == ']' && next == ']') {
                    out.append("  ");
                    i++;
                } else if (c == ']') {
                    inBracketIdentifier = false;
                    out.append(' ');
                } else {
                    out.append(' ');
                }
                continue;
            }

            if (c == '-' && next == '-') {
                inLineComment = true;
                out.append("  ");
                i++;
            } else if (c == '/' && next == '*') {
                inBlockComment = true;
                out.append("  ");
                i++;
            } else if (c == '\'') {
                inSingle = true;
                out.append(' ');
            } else if (c == '"') {
                inDouble = true;
                out.append(' ');
            } else if (c == '[') {
                inBracketIdentifier = true;
                out.append(' ');
            } else {
                out.append(c);
            }
        }
        return out.toString();
    }
}
