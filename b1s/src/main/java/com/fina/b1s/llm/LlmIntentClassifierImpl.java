package com.fina.b1s.llm;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class LlmIntentClassifierImpl implements LlmIntentClassifier {

    private final RestTemplate llmIntentRestTemplate;
    private final LlmIntentProperties properties;
    private final ObjectMapper objectMapper;

    @Override
    public Classification classify(String subject, String bodyText) {
        if (!isConfigured()) {
            return new Classification(false, false, null, "LLM intent classifier is not configured");
        }

        String prompt = """
                判断以下邮件是否表达了明确的下单/采购/订购意向。
                只返回 JSON，不要解释。格式：
                {"order_intent":true|false}

                判定为 true 的例子：
                - 清华大学要订购单单联万向节总成2件，按照之前的价格。
                - 请按上次价格给我们下单 3 件。

                邮件主题：
                %s

                邮件正文：
                %s
                """.formatted(nullToEmpty(subject), nullToEmpty(bodyText));

        Map<String, Object> payload = Map.of(
                "model", properties.model(),
                "max_tokens", properties.maxTokens() > 0 ? properties.maxTokens() : 128,
                "metadata", Map.of("user_id", "b1s-mail-intent"),
                "messages", List.of(Map.of("role", "user", "content", prompt))
        );

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.add("anthropic-version", StringUtils.hasText(properties.anthropicVersion())
                ? properties.anthropicVersion() : "2023-06-01");
        String token = properties.bearerToken();
        if (StringUtils.hasText(token)) {
            headers.setBearerAuth(token.startsWith("Bearer ") ? token.substring(7) : token);
        }

        try {
            String raw = llmIntentRestTemplate.postForObject(
                    properties.endpoint(),
                    new HttpEntity<>(payload, headers),
                    String.class);
            return parse(raw);
        } catch (Exception e) {
            log.warn("LLM intent classification failed: {}", e.getMessage(), e);
            return new Classification(false, false, null, e.getMessage());
        }
    }

    private Classification parse(String raw) throws Exception {
        if (!StringUtils.hasText(raw)) {
            return new Classification(false, false, raw, "empty LLM response");
        }
        JsonNode root = objectMapper.readTree(raw);
        String text = root.path("content").isArray() && root.path("content").size() > 0
                ? root.path("content").get(0).path("text").asText("")
                : raw;
        JsonNode parsed = tryReadJson(text);
        if (parsed == null) {
            parsed = tryReadJson(extractJsonObject(text));
        }
        if (parsed != null && parsed.has("order_intent")) {
            return new Classification(true, parsed.path("order_intent").asBoolean(false), raw, null);
        }
        return new Classification(false, false, raw, "LLM response did not include order_intent");
    }

    private JsonNode tryReadJson(String text) {
        if (!StringUtils.hasText(text)) {
            return null;
        }
        try {
            return objectMapper.readTree(text);
        } catch (Exception e) {
            return null;
        }
    }

    private String extractJsonObject(String text) {
        if (text == null) {
            return null;
        }
        int start = text.indexOf('{');
        int end = text.lastIndexOf('}');
        if (start >= 0 && end > start) {
            return text.substring(start, end + 1);
        }
        return null;
    }

    private boolean isConfigured() {
        return properties.enabled()
                && StringUtils.hasText(properties.endpoint())
                && StringUtils.hasText(properties.model())
                && StringUtils.hasText(properties.bearerToken());
    }

    private String nullToEmpty(String value) {
        return value == null ? "" : value;
    }
}
