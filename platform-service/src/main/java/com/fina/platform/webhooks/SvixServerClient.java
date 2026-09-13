package com.fina.platform.webhooks;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * HTTP adapter to svix-server. Translates our tenant model (X-Tenant-Id) and
 * topic vocabulary onto Svix's application / endpoint / event-type / message
 * concepts. This is the ONLY class that knows Svix exists — swapping webhook
 * providers means rewriting this file plus the compose service.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class SvixServerClient {

    private final SvixProps props;
    private final SvixTokenService tokenService;

    private RestClient restClient() {
        return RestClient.builder()
                .baseUrl(props.getServerUrl())
                .defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer " + tokenService.bearerToken())
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .build();
    }

    // ── applications (our tenants) ──────────────────────────────────────

    /** Returns the Svix application id for a tenant, creating it on first use. */
    public String ensureApplication(String tenantId) {
        try {
            JsonNode existing = restClient().get()
                    .uri("/api/v1/app/uid/{uid}/", tenantId)
                    .retrieve().body(JsonNode.class);
            if (existing != null && existing.hasNonNull("id")) {
                return existing.get("id").asText();
            }
        } catch (RestClientException notFoundOrError) {
            // fall through to creation; a real server failure resurfaces there
        }
        ObjectNode body = com.fasterxml.jackson.databind.json.JsonMapper.builder().build().createObjectNode();
        body.put("name", tenantId);
        body.put("uid", tenantId);
        JsonNode created = restClient().post()
                .uri("/api/v1/app/")
                .body(body)
                .retrieve().body(JsonNode.class);
        String id = created.get("id").asText();
        log.info("created svix application for tenant {} -> {}", tenantId, id);
        return id;
    }

    // ── event types (our topics) ────────────────────────────────────────

    public void ensureEventType(String topic) {
        try {
            restClient().get()
                    .uri("/api/v1/event-type/{name}/", topic)
                    .retrieve().toBodilessEntity();
        } catch (RestClientException e) {
            ObjectNode body = com.fasterxml.jackson.databind.json.JsonMapper.builder().build().createObjectNode();
            body.put("name", topic);
            body.put("description", "factory topic: " + topic);
            restClient().post()
                    .uri("/api/v1/event-type/")
                    .body(body)
                    .retrieve().toBodilessEntity();
            log.info("registered svix event type {}", topic);
        }
    }

    // ── endpoints (our destinations) ────────────────────────────────────

    public Map<String, Object> createEndpoint(String appId, String url, List<String> topics, String description) {
        topics.forEach(this::ensureEventType);
        ObjectNode body = com.fasterxml.jackson.databind.json.JsonMapper.builder().build().createObjectNode();
        body.put("url", url);
        body.put("description", description == null ? "tenant destination" : description);
        ArrayNode filter = body.putArray("filterTypes");
        topics.forEach(filter::add);
        JsonNode created = restClient().post()
                .uri("/api/v1/app/{appId}/endpoint/", appId)
                .body(body)
                .retrieve().body(JsonNode.class);
        return Map.of(
                "endpointId", created.get("id").asText(),
                // Svix generates the whsec_... signing secret; receiver verifies
                // Standard Webhooks signatures with it.
                "secret", created.hasNonNull("key") ? created.get("key").asText() : "",
                "url", created.get("url").asText(),
                "topics", topics);
    }

    public List<Map<String, Object>> listEndpoints(String appId) {
        JsonNode node = restClient().get()
                .uri("/api/v1/app/{appId}/endpoint/{limit}", appId, 50)
                .retrieve().body(JsonNode.class);
        List<Map<String, Object>> out = new ArrayList<>();
        if (node != null && node.has("data")) {
            for (JsonNode ep : node.get("data")) {
                out.add(Map.of(
                        "endpointId", ep.get("id").asText(),
                        "url", ep.get("url").asText(),
                        "disabled", ep.hasNonNull("disabled") && ep.get("disabled").asBoolean()));
            }
        }
        return out;
    }

    public void deleteEndpoint(String appId, String endpointId) {
        restClient().delete()
                .uri("/api/v1/app/{appId}/endpoint/{endpointId}/", appId, endpointId)
                .retrieve().toBodilessEntity();
    }

    // ── messages (our publish) ──────────────────────────────────────────

    public String publish(String appId, String topic, Map<String, Object> data) {
        ensureEventType(topic);
        ObjectNode payload = com.fasterxml.jackson.databind.json.JsonMapper.builder().build().valueToTree(data);
        ObjectNode body = com.fasterxml.jackson.databind.json.JsonMapper.builder().build().createObjectNode();
        body.put("eventType", topic);
        body.set("payload", payload);
        JsonNode created = restClient().post()
                .uri("/api/v1/app/{appId}/msg/", appId)
                .body(body)
                .retrieve().body(JsonNode.class);
        return created.get("id").asText();
    }

    public List<Map<String, Object>> listMessages(String appId, int limit) {
        JsonNode node = restClient().get()
                .uri("/api/v1/app/{appId}/msg/{limit}", appId, Math.min(Math.max(limit, 1), 100))
                .retrieve().body(JsonNode.class);
        List<Map<String, Object>> out = new ArrayList<>();
        if (node != null && node.has("data")) {
            for (JsonNode msg : node.get("data")) {
                out.add(Map.of(
                        "messageId", msg.get("id").asText(),
                        "topic", msg.get("eventType").asText(),
                        "timestamp", msg.get("timestamp").asText()));
            }
        }
        return out;
    }

    /** Delivery attempts of one message across all of the tenant's endpoints. */
    public List<Map<String, Object>> listAttempts(String appId, String messageId) {
        List<Map<String, Object>> out = new ArrayList<>();
        for (Map<String, Object> ep : listEndpoints(appId)) {
            String endpointId = String.valueOf(ep.get("endpointId"));
            try {
                JsonNode node = restClient().get()
                        .uri("/api/v1/app/{appId}/endpoint/{endpointId}/msg/{messageId}/attempt/{limit}",
                                appId, endpointId, messageId, 20)
                        .retrieve().body(JsonNode.class);
                if (node != null && node.has("data")) {
                    for (JsonNode attempt : node.get("data")) {
                        out.add(Map.of(
                                "endpointId", endpointId,
                                "status", attempt.get("status").asText(),
                                "responseStatusCode", attempt.get("responseStatusCodeValue").asInt(0),
                                "attempt", attempt.get("attempt").asInt()));
                    }
                }
            } catch (RestClientException e) {
                log.debug("no attempts for message {} endpoint {}: {}", messageId, endpointId, e.getMessage());
            }
        }
        return out;
    }
}
