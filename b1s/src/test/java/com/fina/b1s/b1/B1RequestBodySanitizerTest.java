package com.fina.b1s.b1;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class B1RequestBodySanitizerTest {

    private final B1RequestBodySanitizer sanitizer = new B1RequestBodySanitizer(new ObjectMapper());

    @Test
    void redactsSensitiveJsonFieldsRecursively() {
        String body = """
                {"CardCode":"C_TROPICAL","Password":"secret","nested":{"accessToken":"abc","cookieValue":"cookie"}}
                """;

        String sanitized = sanitizer.sanitize(body.getBytes(StandardCharsets.UTF_8), 1000);

        assertTrue(sanitized.contains("\"CardCode\":\"C_TROPICAL\""));
        assertTrue(sanitized.contains("\"Password\":\"***\""));
        assertTrue(sanitized.contains("\"accessToken\":\"***\""));
        assertTrue(sanitized.contains("\"cookieValue\":\"***\""));
        assertFalse(sanitized.contains("secret"));
        assertFalse(sanitized.contains("abc"));
    }

    @Test
    void truncatesLongBody() {
        String body = "0123456789".repeat(20);

        String sanitized = sanitizer.sanitize(body.getBytes(StandardCharsets.UTF_8), 40);

        assertTrue(sanitized.length() <= 40);
        assertTrue(sanitized.endsWith("... [truncated]"));
    }
}
