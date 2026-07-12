package com.fina.cdp.util;

import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import java.util.Locale;
import java.util.regex.Pattern;

@Component
public class SqlSafetyValidator {

    private static final Pattern FORBIDDEN_KEYWORDS = Pattern.compile(
            "\\b(insert|update|delete|drop|alter|truncate|create|merge|call|grant|revoke|vacuum|analyze)\\b",
            Pattern.CASE_INSENSITIVE);

    public void validateReadOnly(String sql) {
        if (!StringUtils.hasText(sql)) {
            throw new IllegalArgumentException("Segment SQL is required");
        }
        String trimmed = sql.trim();
        if (trimmed.contains(";")) {
            throw new IllegalArgumentException("Segment SQL must be a single statement without semicolons");
        }
        String normalized = trimmed.toLowerCase(Locale.ROOT);
        if (!(normalized.startsWith("select ") || normalized.startsWith("select\n")
                || normalized.startsWith("with ") || normalized.startsWith("with\n"))) {
            throw new IllegalArgumentException("Segment SQL must be read-only SELECT or WITH SQL");
        }
        if (FORBIDDEN_KEYWORDS.matcher(trimmed).find()) {
            throw new IllegalArgumentException("Segment SQL must be read-only and cannot contain write/DDL keywords");
        }
    }
}
