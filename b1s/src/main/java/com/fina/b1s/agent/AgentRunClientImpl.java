package com.fina.b1s.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class AgentRunClientImpl implements AgentRunClient {

    private final HttpClient agentRunHttpClient;
    private final AgentRunProperties properties;
    private final ObjectMapper objectMapper;

    @Override
    public AgentRunResult run(AgentRunRequest request) {
        if (!isConfigured()) {
            return new AgentRunResult(false, "DISABLED", null, null, "Agent run is not configured");
        }
        try {
            URI uri = URI.create(normalizeBaseUrl(properties.baseUrl()) + "/api/runs");
            HttpRequest.Builder requestBuilder = HttpRequest.newBuilder(uri)
                    .timeout(java.time.Duration.ofMillis(timeoutOrDefault(properties.readTimeoutMs(), 600_000)))
                    .header("Content-Type", "application/json")
                    .header("Accept", Boolean.TRUE.equals(request.streaming())
                            ? "text/event-stream, application/json"
                            : "application/json");
            applyAuthHeaders(requestBuilder);
            HttpRequest httpRequest = requestBuilder
                    .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(request), StandardCharsets.UTF_8))
                    .build();

            if (Boolean.TRUE.equals(request.streaming())) {
                return runStreaming(httpRequest);
            }

            HttpResponse<String> response = agentRunHttpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            String raw = response.body();
            String runId = extractRunId(raw);
            boolean success = isSuccess(response.statusCode());
            return new AgentRunResult(success, String.valueOf(response.statusCode()), runId, raw, success ? null : raw);
        } catch (IOException e) {
            log.warn("Agent run request failed: {}", e.getMessage(), e);
            return new AgentRunResult(false, "ERROR", null, null, e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.warn("Agent run request interrupted: {}", e.getMessage(), e);
            return new AgentRunResult(false, "INTERRUPTED", null, null, e.getMessage());
        } catch (Exception e) {
            log.warn("Agent run request failed: {}", e.getMessage(), e);
            return new AgentRunResult(false, "ERROR", null, null, e.getMessage());
        }
    }

    private boolean isConfigured() {
        return properties.enabled()
                && StringUtils.hasText(properties.baseUrl())
                && StringUtils.hasText(properties.tenantId())
                && StringUtils.hasText(properties.assistantId());
    }

    private String normalizeBaseUrl(String baseUrl) {
        String normalized = baseUrl.trim();
        while (normalized.endsWith("/")) {
            normalized = normalized.substring(0, normalized.length() - 1);
        }
        return normalized;
    }

    private AgentRunResult runStreaming(HttpRequest httpRequest) throws IOException, InterruptedException {
        HttpResponse<InputStream> response = agentRunHttpClient.send(httpRequest, HttpResponse.BodyHandlers.ofInputStream());
        String raw = summarizeResponse(response);
        boolean success = isSuccess(response.statusCode());
        try (InputStream body = response.body()) {
            // The service layer can keep the SSE stream open; headers are enough to treat the dispatch as accepted.
        }
        return new AgentRunResult(success, String.valueOf(response.statusCode()), null, raw, success ? null : raw);
    }

    private String summarizeResponse(HttpResponse<?> response) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("status", response.statusCode());
        summary.put("contentType", response.headers().firstValue("content-type").orElse(null));
        summary.put("contentLength", response.headers().firstValue("content-length").orElse(null));
        summary.put("transferEncoding", response.headers().firstValue("transfer-encoding").orElse(null));
        summary.put("location", response.headers().firstValue("location").orElse(null));
        try {
            return objectMapper.writeValueAsString(summary);
        } catch (Exception e) {
            return summary.toString();
        }
    }

    private void applyAuthHeaders(HttpRequest.Builder builder) {
        String token = properties.bearerToken();
        if (StringUtils.hasText(token)) {
            String normalizedToken = token.trim();
            if (normalizedToken.startsWith("Bearer ")) {
                normalizedToken = normalizedToken.substring(7);
            }
            builder.header("Authorization", "Bearer " + normalizedToken);
        }
        builder.header("x-tenant-id", properties.tenantId());
        if (StringUtils.hasText(properties.workspaceId())) {
            builder.header("x-workspace-id", properties.workspaceId());
        }
        if (StringUtils.hasText(properties.projectId())) {
            builder.header("x-project-id", properties.projectId());
        }
    }

    private boolean isSuccess(int statusCode) {
        return statusCode >= 200 && statusCode < 300;
    }

    private long timeoutOrDefault(long value, long defaultValue) {
        return value > 0 ? value : defaultValue;
    }

    private String extractRunId(String raw) {
        if (!StringUtils.hasText(raw)) {
            return null;
        }
        try {
            JsonNode node = objectMapper.readTree(raw);
            for (String key : new String[]{"run_id", "id", "runId"}) {
                JsonNode value = node.get(key);
                if (value != null && !value.isNull()) {
                    return value.asText();
                }
            }
        } catch (Exception ignore) {
        }
        return null;
    }
}
