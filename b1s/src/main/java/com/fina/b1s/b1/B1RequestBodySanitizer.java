package com.fina.b1s.b1;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.Map;

@Component
public class B1RequestBodySanitizer {

    private static final String REDACTED = "***";
    private static final String TRUNCATED_SUFFIX = "... [truncated]";
    private static final String[] SENSITIVE_FIELD_PARTS = {
            "password",
            "token",
            "secret",
            "session",
            "cookie",
            "authorization",
            "accesskey",
            "secretkey"
    };

    private final ObjectMapper objectMapper;

    public B1RequestBodySanitizer(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public String sanitize(byte[] body, int maxChars) {
        if (body == null || body.length == 0) {
            return "";
        }
        String text = new String(body, StandardCharsets.UTF_8);
        String sanitized = sanitizeJsonIfPossible(text);
        return truncate(sanitized, maxChars);
    }

    private String sanitizeJsonIfPossible(String text) {
        String trimmed = text.trim();
        if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) {
            return text;
        }
        try {
            JsonNode node = objectMapper.readTree(text);
            redact(node);
            return objectMapper.writeValueAsString(node);
        } catch (Exception ignored) {
            return text;
        }
    }

    private void redact(JsonNode node) {
        if (node instanceof ObjectNode objectNode) {
            Iterator<Map.Entry<String, JsonNode>> fields = objectNode.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> field = fields.next();
                if (isSensitive(field.getKey())) {
                    objectNode.put(field.getKey(), REDACTED);
                } else {
                    redact(field.getValue());
                }
            }
            return;
        }
        if (node instanceof ArrayNode arrayNode) {
            for (JsonNode item : arrayNode) {
                redact(item);
            }
        }
    }

    private static boolean isSensitive(String fieldName) {
        String normalized = fieldName == null ? "" : fieldName.replaceAll("[^A-Za-z0-9]", "").toLowerCase();
        for (String part : SENSITIVE_FIELD_PARTS) {
            if (normalized.contains(part)) {
                return true;
            }
        }
        return false;
    }

    private static String truncate(String value, int maxChars) {
        if (value.length() <= maxChars) {
            return value;
        }
        if (maxChars <= TRUNCATED_SUFFIX.length()) {
            return value.substring(0, Math.max(0, maxChars));
        }
        int end = Math.max(0, maxChars - TRUNCATED_SUFFIX.length());
        return value.substring(0, end) + TRUNCATED_SUFFIX;
    }
}
