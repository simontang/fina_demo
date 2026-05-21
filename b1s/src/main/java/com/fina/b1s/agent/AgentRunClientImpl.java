package com.fina.b1s.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

import java.net.URI;
import java.util.LinkedHashMap;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class AgentRunClientImpl implements AgentRunClient {

    private final RestTemplate agentRunRestTemplate;
    private final AgentRunProperties properties;
    private final ObjectMapper objectMapper;

    @Override
    public AgentRunResult run(AgentRunRequest request) {
        if (!isConfigured()) {
            return new AgentRunResult(false, "DISABLED", null, null, "Agent run is not configured");
        }
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            if (StringUtils.hasText(properties.bearerToken())) {
                String token = properties.bearerToken().trim();
                headers.setBearerAuth(token.startsWith("Bearer ") ? token.substring(7) : token);
            }
            headers.add("x-tenant-id", properties.tenantId());
            if (StringUtils.hasText(properties.workspaceId())) {
                headers.add("x-workspace-id", properties.workspaceId());
            }
            if (StringUtils.hasText(properties.projectId())) {
                headers.add("x-project-id", properties.projectId());
            }

            URI uri = URI.create(normalizeBaseUrl(properties.baseUrl()) + "/api/runs");
            HttpEntity<AgentRunRequest> entity = new HttpEntity<>(request, headers);
            ResponseEntity<String> response = agentRunRestTemplate.postForEntity(uri, entity, String.class);
            String raw = response.getBody();
            String runId = extractRunId(raw);
            return new AgentRunResult(true, String.valueOf(response.getStatusCode().value()), runId, raw, null);
        } catch (RestClientException e) {
            log.warn("Agent run request failed: {}", e.getMessage(), e);
            return new AgentRunResult(false, "ERROR", null, null, e.getMessage());
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
