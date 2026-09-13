package com.fina.platform.webhooks;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

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

    /** tenant → svix application id, resolved once per process. */
    private final ConcurrentHashMap<String, String> APP_ID_CACHE = new ConcurrentHashMap<>();

    private RestClient restClient() {
        return RestClient.builder()
                .baseUrl(props.getServerUrl())
                .defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer " + tokenService.bearerToken())
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .build();
    }

    // ── applications (our tenants) ──────────────────────────────────────

    /**
     * Returns the Svix application id for a tenant, creating it on first use.
     * The OSS svix-server build has no GET-by-uid route, so idempotency is
     * handled by treating 409 as "exists" and resolving the id via list.
     */
    public synchronized String ensureApplication(String tenantId) {
        String cached = APP_ID_CACHE.get(tenantId);
        if (cached != null) {
            return cached;
        }
        ObjectNode body = jsonMapper().createObjectNode();
        body.put("name", tenantId);
        body.put("uid", tenantId);
        JsonNode created;
        try {
            created = restClient().post()
                    .uri("/api/v1/app/")
                    .body(body)
                    .retrieve().body(JsonNode.class);
        } catch (HttpClientErrorException.Conflict e) {
            created = findApplicationByUid(tenantId);
            if (created == null) {
                throw new IllegalStateException(
                        "svix reports application conflict but it cannot be found: " + tenantId, e);
            }
        }
        String id = created.get("id").asText();
        APP_ID_CACHE.put(tenantId, id);
        log.info("resolved svix application for tenant {} -> {}", tenantId, id);
        return id;
    }

    private JsonNode findApplicationByUid(String tenantId) {
        JsonNode list = restClient().get()
                .uri("/api/v1/app/100/")
                .retrieve().body(JsonNode.class);
        if (list != null && list.has("data")) {
            for (JsonNode app : list.get("data")) {
                if (tenantId.equals(app.path("uid").asText(null))) {
                    return app;
                }
            }
        }
        return null;
    }

    private com.fasterxml.jackson.databind.ObjectMapper jsonMapper() {
        return com.fasterxml.jackson.databind.json.JsonMapper.builder().build();
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
        String endpointId = created.get("id").asText();
        // Svix does NOT return the signing secret in the creation response —
        // it lives behind the dedicated /secret/ endpoint.
        String secret = "";
        try {
            JsonNode secretNode = restClient().get()
                    .uri("/api/v1/app/{appId}/endpoint/{endpointId}/secret/", appId, endpointId)
                    .retrieve().body(JsonNode.class);
            if (secretNode != null && secretNode.hasNonNull("key")) {
                secret = secretNode.get("key").asText();
            }
        } catch (RestClientException e) {
            log.warn("could not fetch signing secret for endpoint {}: {}", endpointId, e.getMessage());
        }
        return Map.of(
                "endpointId", endpointId,
                "secret", secret,
                "url", created.get("url").asText(),
                "topics", topics);
    }

    public List<Map<String, Object>> listEndpoints(String appId) {
        JsonNode node = restClient().get()
                .uri("/api/v1/app/{appId}/endpoint/?limit=50", appId)
                .retrieve().body(JsonNode.class);
        List<Map<String, Object>> out = new ArrayList<>();
        if (node != null && node.has("data")) {
            for (JsonNode ep : node.get("data")) {
                List<String> topics = new ArrayList<>();
                if (ep.has("filterTypes") && ep.get("filterTypes").isArray()) {
                    ep.get("filterTypes").forEach(t -> topics.add(t.asText()));
                }
                out.add(Map.of(
                        "endpointId", ep.get("id").asText(),
                        "url", ep.get("url").asText(),
                        "topics", topics,
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
                .uri("/api/v1/app/{appId}/msg/?limit={limit}", appId, Math.min(Math.max(limit, 1), 100))
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

    /** Delivery status of one message across all of the tenant's endpoints. */
    public List<Map<String, Object>> listAttempts(String appId, String messageId) {
        JsonNode node = restClient().get()
                .uri("/api/v1/app/{appId}/msg/{messageId}/endpoint/?limit=50", appId, messageId)
                .retrieve().body(JsonNode.class);
        List<Map<String, Object>> out = new ArrayList<>();
        if (node != null && node.has("data")) {
            for (JsonNode ep : node.get("data")) {
                out.add(Map.of(
                        "endpointId", ep.get("id").asText(),
                        "url", ep.path("url").asText(""),
                        "status", ep.path("status").asText("unknown")));
            }
        }
        return out;
    }
}
