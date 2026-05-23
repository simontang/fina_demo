package com.fina.b1s.mail;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.b1s.llm.LlmIntentProperties;
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
public class PurchaseOrderSummaryServiceImpl implements PurchaseOrderSummaryService {

    private static final int BODY_LIMIT = 6000;
    private static final int ATTACHMENT_LIMIT = 12000;

    private final RestTemplate llmIntentRestTemplate;
    private final LlmIntentProperties properties;
    private final ObjectMapper objectMapper;

    @Override
    public PurchaseOrderSummary summarize(String subject, String bodyText, String attachmentText) {
        if (isConfigured()) {
            PurchaseOrderSummary llmSummary = summarizeWithLlm(subject, bodyText, attachmentText);
            if (llmSummary != null) {
                return llmSummary;
            }
        }
        return summarizeHeuristically(subject, bodyText, attachmentText);
    }

    private PurchaseOrderSummary summarizeWithLlm(String subject, String bodyText, String attachmentText) {
        String prompt = """
                You are extracting purchase order information from an email and its attachment text.
                Return JSON only. Do not explain.
                Format:
                {
                  "summary_text": "compact plain text summary",
                  "agent_message": "plain text order request for downstream ERP agent"
                }

                Requirements:
                - Focus on customer, document type, order/request intent, item names or codes, quantities, price hints, delivery dates, and notes.
                - If the attachment contains the actual purchase order details, prefer the attachment over the email body.
                - Keep agent_message concise but complete enough for order processing.
                - Use English if the source is English, Chinese if the source is Chinese.

                Email subject:
                %s

                Email body:
                %s

                Attachment text:
                %s
                """.formatted(
                clip(subject, 1000),
                clip(bodyText, BODY_LIMIT),
                clip(attachmentText, ATTACHMENT_LIMIT));

        Map<String, Object> payload = Map.of(
                "model", properties.model(),
                "max_tokens", Math.max(properties.maxTokens(), 256),
                "metadata", Map.of("user_id", "b1s-mail-po-summary"),
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
            JsonNode parsed = parsePayload(raw);
            if (parsed == null) {
                return null;
            }
            String summary = normalize(parsed.path("summary_text").asText(null));
            String agentMessage = normalize(parsed.path("agent_message").asText(null));
            if (!StringUtils.hasText(summary) && !StringUtils.hasText(agentMessage)) {
                return null;
            }
            if (!StringUtils.hasText(agentMessage)) {
                agentMessage = summary;
            }
            if (!StringUtils.hasText(summary)) {
                summary = agentMessage;
            }
            return new PurchaseOrderSummary(summary, agentMessage);
        } catch (Exception e) {
            log.warn("PO summary extraction failed: {}", e.getMessage());
            return null;
        }
    }

    private PurchaseOrderSummary summarizeHeuristically(String subject, String bodyText, String attachmentText) {
        String primary = firstNonBlank(attachmentText, bodyText, subject);
        if (!StringUtils.hasText(primary)) {
            return new PurchaseOrderSummary(null, null);
        }
        String summary = normalize(primary);
        if (summary.length() > 2000) {
            summary = summary.substring(0, 2000);
        }
        return new PurchaseOrderSummary(summary, summary);
    }

    private JsonNode parsePayload(String raw) {
        if (!StringUtils.hasText(raw)) {
            return null;
        }
        try {
            JsonNode root = objectMapper.readTree(raw);
            String text = root.path("content").isArray() && root.path("content").size() > 0
                    ? root.path("content").get(0).path("text").asText("")
                    : raw;
            return parseJson(text);
        } catch (Exception e) {
            return parseJson(raw);
        }
    }

    private JsonNode parseJson(String text) {
        if (!StringUtils.hasText(text)) {
            return null;
        }
        try {
            return objectMapper.readTree(text);
        } catch (Exception e) {
            int start = text.indexOf('{');
            int end = text.lastIndexOf('}');
            if (start >= 0 && end > start) {
                try {
                    return objectMapper.readTree(text.substring(start, end + 1));
                } catch (Exception ignore) {
                    return null;
                }
            }
            return null;
        }
    }

    private boolean isConfigured() {
        return properties.enabled()
                && StringUtils.hasText(properties.endpoint())
                && StringUtils.hasText(properties.model())
                && StringUtils.hasText(properties.bearerToken());
    }

    private String clip(String value, int limit) {
        if (!StringUtils.hasText(value)) {
            return "";
        }
        String normalized = normalize(value);
        return normalized.length() <= limit ? normalized : normalized.substring(0, limit);
    }

    private String firstNonBlank(String... values) {
        for (String value : values) {
            if (StringUtils.hasText(value)) {
                return value;
            }
        }
        return null;
    }

    private String normalize(String value) {
        if (!StringUtils.hasText(value)) {
            return null;
        }
        return value.replace('\u00a0', ' ')
                .replaceAll("[ \\t\\x0B\\f\\r]+", " ")
                .replaceAll("\\n{3,}", "\n\n")
                .trim();
    }
}
