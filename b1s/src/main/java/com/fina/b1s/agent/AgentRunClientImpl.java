package com.fina.b1s.agent;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;

@Slf4j
@Service
public class AgentRunClientImpl implements AgentRunClient {

    private final HttpClient agentRunHttpClient;
    private final AgentRunProperties properties;
    private final ObjectMapper objectMapper;

    public AgentRunClientImpl(@Qualifier("agentRunHttpClient") HttpClient agentRunHttpClient,
                              AgentRunProperties properties,
                              ObjectMapper objectMapper) {
        this.agentRunHttpClient = agentRunHttpClient;
        this.properties = properties;
        this.objectMapper = objectMapper;
    }

    @Override
    public AgentRunResult run(AgentInboundRequest request) {
        if (!isConfigured()) {
            return new AgentRunResult(false, "DISABLED", null, null, "Agent inbound is not configured");
        }
        try {
            URI uri = URI.create(normalizeBaseUrl(properties.baseUrl()) + "/api/channels/inbound");
            HttpRequest.Builder requestBuilder = HttpRequest.newBuilder(uri)
                    .timeout(java.time.Duration.ofMillis(timeoutOrDefault(properties.readTimeoutMs(), 600_000)))
                    .header("Content-Type", "application/json")
                    .header("Accept", "application/json");
            applyAuthHeaders(requestBuilder);
            HttpRequest httpRequest = requestBuilder
                    .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(request), StandardCharsets.UTF_8))
                    .build();

            HttpResponse<String> response = agentRunHttpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            String raw = response.body();
            boolean success = isSuccess(response.statusCode());
            return new AgentRunResult(success, String.valueOf(response.statusCode()), null, raw, success ? null : raw);
        } catch (IOException e) {
            log.warn("Agent inbound request failed: {}", e.getMessage(), e);
            return new AgentRunResult(false, "ERROR", null, null, e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.warn("Agent inbound request interrupted: {}", e.getMessage(), e);
            return new AgentRunResult(false, "INTERRUPTED", null, null, e.getMessage());
        } catch (Exception e) {
            log.warn("Agent inbound request failed: {}", e.getMessage(), e);
            return new AgentRunResult(false, "ERROR", null, null, e.getMessage());
        }
    }

    private boolean isConfigured() {
        return properties.isConfigured();
    }

    private String normalizeBaseUrl(String baseUrl) {
        String normalized = baseUrl.trim();
        while (normalized.endsWith("/")) {
            normalized = normalized.substring(0, normalized.length() - 1);
        }
        return normalized;
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
    }

    private boolean isSuccess(int statusCode) {
        return statusCode >= 200 && statusCode < 300;
    }

    private long timeoutOrDefault(long value, long defaultValue) {
        return value > 0 ? value : defaultValue;
    }
}
