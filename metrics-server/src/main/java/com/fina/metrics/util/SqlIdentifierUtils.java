package com.fina.metrics.util;

import org.springframework.util.StringUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.stream.Collectors;

public final class SqlIdentifierUtils {

    private SqlIdentifierUtils() {
    }

    public static TableIdentifier parseTableIdentifier(String schemaName, String tableName) {
        if (!StringUtils.hasText(tableName)) {
            return new TableIdentifier(normalize(schemaName), null);
        }
        List<String> parts = splitQualifiedIdentifier(tableName);
        String parsedTable = parts.isEmpty() ? normalize(tableName) : parts.get(parts.size() - 1);
        String parsedSchema = StringUtils.hasText(schemaName)
                ? normalize(schemaName)
                : parts.size() >= 2 ? parts.get(parts.size() - 2) : null;
        return new TableIdentifier(parsedSchema, parsedTable);
    }

    public static String quoteQualified(String identifier) {
        List<String> parts = splitQualifiedIdentifier(identifier);
        if (parts.isEmpty()) {
            return quotePart(identifier);
        }
        return parts.stream()
                .map(SqlIdentifierUtils::quotePart)
                .collect(Collectors.joining("."));
    }

    public static boolean sameTableName(String left, String right, boolean caseSensitive) {
        TableIdentifier leftId = parseTableIdentifier(null, left);
        TableIdentifier rightId = parseTableIdentifier(null, right);
        if (!StringUtils.hasText(leftId.tableName()) || !StringUtils.hasText(rightId.tableName())) {
            return false;
        }
        boolean tableMatches = caseSensitive
                ? leftId.tableName().equals(rightId.tableName())
                : leftId.tableName().equalsIgnoreCase(rightId.tableName());
        if (!tableMatches) {
            return false;
        }
        if (!StringUtils.hasText(leftId.schemaName()) || !StringUtils.hasText(rightId.schemaName())) {
            return true;
        }
        return caseSensitive
                ? leftId.schemaName().equals(rightId.schemaName())
                : leftId.schemaName().equalsIgnoreCase(rightId.schemaName());
    }

    private static List<String> splitQualifiedIdentifier(String identifier) {
        List<String> parts = new ArrayList<>();
        if (!StringUtils.hasText(identifier)) {
            return parts;
        }
        StringBuilder current = new StringBuilder();
        char quote = 0;
        for (int i = 0; i < identifier.length(); i++) {
            char c = identifier.charAt(i);
            if (quote != 0) {
                if (c == quote) {
                    if (i + 1 < identifier.length() && identifier.charAt(i + 1) == quote) {
                        current.append(c);
                        i++;
                    } else {
                        quote = 0;
                    }
                } else {
                    current.append(c);
                }
                continue;
            }
            if (c == '"' || c == '`') {
                quote = c;
                continue;
            }
            if (c == '[') {
                quote = ']';
                continue;
            }
            if (c == '.') {
                addPart(parts, current);
                current.setLength(0);
                continue;
            }
            current.append(c);
        }
        addPart(parts, current);
        return parts;
    }

    private static void addPart(List<String> parts, StringBuilder current) {
        String part = normalize(current.toString());
        if (StringUtils.hasText(part)) {
            parts.add(part);
        }
    }

    private static String normalize(String value) {
        return StringUtils.hasText(value) ? value.trim() : null;
    }

    private static String quotePart(String part) {
        String value = Objects.toString(part, "").trim();
        return "\"" + value.replace("\"", "\"\"") + "\"";
    }

    public static String normalizeForComparison(String value, boolean caseSensitive) {
        String normalized = normalize(value);
        if (normalized == null || caseSensitive) {
            return normalized;
        }
        return normalized.toLowerCase(Locale.ROOT);
    }

    public record TableIdentifier(String schemaName, String tableName) {}
}
