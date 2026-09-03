package com.fina.metrics.util;

import org.springframework.util.StringUtils;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public final class SqlTableReferenceExtractor {

    private static final Set<String> FROM_END_KEYWORDS = Set.of(
            "where", "group", "order", "having", "limit", "offset", "fetch",
            "union", "intersect", "except", "qualify", "window", "connect", "start");

    private SqlTableReferenceExtractor() {
    }

    public static Set<TableReference> extract(String sql) {
        if (!StringUtils.hasText(sql)) {
            return Set.of();
        }
        List<Token> tokens = tokenize(sql);
        Set<String> cteNames = extractCteNames(tokens);
        Set<TableReference> references = new LinkedHashSet<>();
        boolean expectTable = false;
        boolean inFromList = false;
        int fromDepth = -1;
        int depth = 0;

        for (int i = 0; i < tokens.size(); i++) {
            Token token = tokens.get(i);
            if (expectTable && token.isSymbol("(")) {
                expectTable = false;
                inFromList = true;
                fromDepth = depth;
                depth++;
                continue;
            }
            if (token.isSymbol("(")) {
                depth++;
                continue;
            }
            if (token.isSymbol(")")) {
                depth = Math.max(0, depth - 1);
                if (inFromList && depth < fromDepth) {
                    inFromList = false;
                    fromDepth = -1;
                }
                continue;
            }

            if (expectTable) {
                if (token.isSymbol("(")) {
                    expectTable = false;
                    inFromList = true;
                    fromDepth = depth;
                    continue;
                }
                if (token.isIdentifier()) {
                    QualifiedName name = readQualifiedName(tokens, i);
                    i = name.endIndex();
                    if (name.isFunctionCall(tokens)) {
                        throw new IllegalArgumentException("SQL probe cannot validate table-valued function: " + name.original());
                    }
                    if (!(name.schemaName() == null && cteNames.contains(name.tableName().toLowerCase(Locale.ROOT)))) {
                        references.add(new TableReference(name.schemaName(), name.tableName(), name.original()));
                    }
                    expectTable = false;
                    inFromList = true;
                    fromDepth = depth;
                }
                continue;
            }

            if (token.isKeyword("from") || token.isKeyword("join")) {
                expectTable = true;
                inFromList = false;
                fromDepth = depth;
                continue;
            }

            if (inFromList && depth == fromDepth) {
                if (token.isSymbol(",")) {
                    expectTable = true;
                } else if (token.isIdentifier()
                        && FROM_END_KEYWORDS.contains(token.lower())) {
                    inFromList = false;
                    fromDepth = -1;
                }
            }
        }
        return references;
    }

    private static Set<String> extractCteNames(List<Token> tokens) {
        Set<String> cteNames = new LinkedHashSet<>();
        if (tokens.isEmpty() || !tokens.get(0).isKeyword("with")) {
            return cteNames;
        }
        int i = 1;
        if (i < tokens.size() && tokens.get(i).isKeyword("recursive")) {
            i++;
        }
        while (i < tokens.size()) {
            Token name = tokens.get(i);
            if (!name.isIdentifier()) {
                break;
            }
            cteNames.add(name.lower());
            i++;
            if (i < tokens.size() && tokens.get(i).isSymbol("(")) {
                i = skipBalanced(tokens, i);
            }
            if (i >= tokens.size() || !tokens.get(i).isKeyword("as")) {
                break;
            }
            i++;
            if (i >= tokens.size() || !tokens.get(i).isSymbol("(")) {
                break;
            }
            i = skipBalanced(tokens, i);
            if (i < tokens.size() && tokens.get(i).isSymbol(",")) {
                i++;
                continue;
            }
            break;
        }
        return cteNames;
    }

    private static int skipBalanced(List<Token> tokens, int openIndex) {
        int depth = 0;
        for (int i = openIndex; i < tokens.size(); i++) {
            if (tokens.get(i).isSymbol("(")) {
                depth++;
            } else if (tokens.get(i).isSymbol(")")) {
                depth--;
                if (depth == 0) {
                    return i + 1;
                }
            }
        }
        return tokens.size();
    }

    private static QualifiedName readQualifiedName(List<Token> tokens, int startIndex) {
        List<String> parts = new ArrayList<>();
        int i = startIndex;
        parts.add(tokens.get(i).value());
        while (i + 2 < tokens.size()
                && tokens.get(i + 1).isSymbol(".")
                && tokens.get(i + 2).isIdentifier()) {
            parts.add(tokens.get(i + 2).value());
            i += 2;
        }
        String tableName = parts.get(parts.size() - 1);
        String schemaName = parts.size() >= 2 ? parts.get(parts.size() - 2) : null;
        return new QualifiedName(schemaName, tableName, String.join(".", parts), i);
    }

    private static List<Token> tokenize(String sql) {
        List<Token> tokens = new ArrayList<>();
        for (int i = 0; i < sql.length(); i++) {
            char c = sql.charAt(i);
            char next = i + 1 < sql.length() ? sql.charAt(i + 1) : '\0';

            if (Character.isWhitespace(c)) {
                continue;
            }
            if (c == '-' && next == '-') {
                i++;
                while (i + 1 < sql.length() && sql.charAt(i + 1) != '\n' && sql.charAt(i + 1) != '\r') {
                    i++;
                }
                continue;
            }
            if (c == '/' && next == '*') {
                i += 2;
                while (i < sql.length() - 1 && !(sql.charAt(i) == '*' && sql.charAt(i + 1) == '/')) {
                    i++;
                }
                i++;
                continue;
            }
            if (c == '\'') {
                i = skipSingleQuoted(sql, i);
                continue;
            }
            if (c == '"') {
                ParsedIdentifier parsed = readDelimitedIdentifier(sql, i, '"', "\"\"");
                tokens.add(new Token(parsed.value(), true));
                i = parsed.endIndex();
                continue;
            }
            if (c == '[') {
                ParsedIdentifier parsed = readDelimitedIdentifier(sql, i, ']', "]]");
                tokens.add(new Token(parsed.value(), true));
                i = parsed.endIndex();
                continue;
            }
            if (c == '`') {
                ParsedIdentifier parsed = readDelimitedIdentifier(sql, i, '`', "``");
                tokens.add(new Token(parsed.value(), true));
                i = parsed.endIndex();
                continue;
            }
            if (c == '.' || c == ',' || c == '(' || c == ')') {
                tokens.add(new Token(String.valueOf(c), false));
                continue;
            }
            if (isIdentifierChar(c)) {
                int start = i;
                while (i + 1 < sql.length() && isIdentifierChar(sql.charAt(i + 1))) {
                    i++;
                }
                tokens.add(new Token(sql.substring(start, i + 1), true));
            }
        }
        return tokens;
    }

    private static boolean isIdentifierChar(char c) {
        return Character.isLetterOrDigit(c) || c == '_' || c == '$' || c == '#' || c == '@';
    }

    private static int skipSingleQuoted(String sql, int start) {
        for (int i = start + 1; i < sql.length(); i++) {
            char c = sql.charAt(i);
            char next = i + 1 < sql.length() ? sql.charAt(i + 1) : '\0';
            if (c == '\'' && next == '\'') {
                i++;
                continue;
            }
            if (c == '\'') {
                return i;
            }
        }
        return sql.length() - 1;
    }

    private static ParsedIdentifier readDelimitedIdentifier(
            String sql,
            int start,
            char closing,
            String escapedClosing) {
        StringBuilder value = new StringBuilder();
        for (int i = start + 1; i < sql.length(); i++) {
            char c = sql.charAt(i);
            if (i + escapedClosing.length() <= sql.length()
                    && sql.substring(i, i + escapedClosing.length()).equals(escapedClosing)) {
                value.append(closing);
                i += escapedClosing.length() - 1;
                continue;
            }
            if (c == closing) {
                return new ParsedIdentifier(value.toString(), i);
            }
            value.append(c);
        }
        return new ParsedIdentifier(value.toString(), sql.length() - 1);
    }

    private record Token(String value, boolean identifier) {
        boolean isIdentifier() {
            return identifier;
        }

        boolean isKeyword(String expected) {
            return identifier && lower().equals(expected);
        }

        boolean isSymbol(String symbol) {
            return !identifier && value.equals(symbol);
        }

        String lower() {
            return value.toLowerCase(Locale.ROOT);
        }
    }

    private record QualifiedName(String schemaName, String tableName, String original, int endIndex) {
        boolean isFunctionCall(List<Token> tokens) {
            return endIndex + 1 < tokens.size() && tokens.get(endIndex + 1).isSymbol("(");
        }
    }

    private record ParsedIdentifier(String value, int endIndex) {}

    public record TableReference(String schemaName, String tableName, String original) {}
}
