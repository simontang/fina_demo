package com.fina.platform.webhooks;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fina.platform.exception.ApiException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * HTTP adapter to svix-server. Translates our tenant model (X-Tenant-Id) and
 * event vocabulary onto Svix's application / endpoint / event-type / message
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
                // Translate the webhook backend's statuses into our error
                // contract: a missing destination/message is 404 (not 500), a
                // rejected payload is 400, a duplicate creation is 409.
                .defaultStatusHandler(org.springframework.http.HttpStatusCode::isError, (request, response) -> {
                    int sc = response.getStatusCode().value();
                    if (sc == 404) {
                        throw ApiException.notFound("webhook resource not found");
                    }
                    if (sc == 400 || sc == 422) {
                        throw ApiException.badRequest("webhook backend rejected the request");
                    }
                    if (sc == 409) {
                        throw new ApiException(409, "CONFLICT", "webhook resource already exists");
                    }
                    throw new ApiException(502, "WEBHOOK_BACKEND_ERROR",
                            "webhook backend returned HTTP " + sc);
                })
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
        JsonNode found = findApplicationByUid(tenantId);
        if (found == null) {
            ObjectNode body = jsonMapper().createObjectNode();
            body.put("name", tenantId);
            body.put("uid", tenantId);
            try {
                found = restClient().post()
                        .uri("/api/v1/app/")
                        .body(body)
                        .retrieve().body(JsonNode.class);
            } catch (ApiException e) {
                if (e.getStatus() != 409) {
                    throw e;
                }
                // lost a race with a concurrent create — look it up again
                found = findApplicationByUid(tenantId);
                if (found == null) {
                    throw new IllegalStateException(
                            "svix reports application conflict but it cannot be found: " + tenantId, e);
                }
            }
        }
        String id = found.get("id").asText();
        APP_ID_CACHE.put(tenantId, id);
        log.debug("resolved svix application for tenant {} -> {}", tenantId, id);
        return id;
    }

    /**
     * Resolves an existing application by uid. This svix-server build does not
     * serve limit-as-path-segment routes (`/app/{limit}/`), so we page the list
     * endpoint with `?limit=` (`iterator` is the continuation token) and match
     * on uid client-side.
     */
    private JsonNode findApplicationByUid(String tenantId) {
        String iterator = null;
        for (int page = 0; page < 10; page++) {
            String uri = "/api/v1/app/?limit=100" + (iterator == null ? "" : "&iterator=" + iterator);
            JsonNode list = restClient().get().uri(uri).retrieve().body(JsonNode.class);
            if (list == null) {
                return null;
            }
            if (list.has("data")) {
                for (JsonNode app : list.get("data")) {
                    if (tenantId.equals(app.path("uid").asText(null))) {
                        return app;
                    }
                }
            }
            boolean done = !list.hasNonNull("done") || list.get("done").asBoolean(true);
            iterator = list.path("iterator").asText(null);
            if (done || iterator == null) {
                break;
            }
        }
        log.warn("application for tenant {} not found in the first pages of the list", tenantId);
        return null;
    }

    private com.fasterxml.jackson.databind.ObjectMapper jsonMapper() {
        return com.fasterxml.jackson.databind.json.JsonMapper.builder().build();
    }

    // ── event types ─────────────────────────────────────────────────────

    /**
     * Register the event type if it is not there yet. A missing type comes
     * back as 404 from the backend, which our status handler surfaces as
     * ApiException(404) — catch that (not RestClientException) or every
     * publish to a brand-new eventType fails with 404.
     */
    public void ensureEventType(String eventType) {
        boolean exists;
        try {
            restClient().get()
                    .uri("/api/v1/event-type/{name}/", eventType)
                    .retrieve().toBodilessEntity();
            exists = true;
        } catch (ApiException e) {
            if (e.getStatus() != 404) {
                throw e;
            }
            exists = false;
        }
        if (exists) {
            return;
        }
        ObjectNode body = jsonMapper().createObjectNode();
        body.put("name", eventType);
        body.put("description", "factory event type: " + eventType);
        try {
            restClient().post()
                    .uri("/api/v1/event-type/")
                    .body(body)
                    .retrieve().toBodilessEntity();
            log.info("registered svix event type {}", eventType);
        } catch (ApiException e) {
            if (e.getStatus() != 409) {
                throw e;
            }
            log.debug("event type {} already registered (race)", eventType);
        }
    }

    // ── endpoints (our destinations) ────────────────────────────────────

    public Map<String, Object> createEndpoint(
            String appId,
            String url,
            List<String> filterTypes,
            List<String> channels,
            String description) {
        List<String> normalizedFilterTypes = normalizeStrings(filterTypes);
        List<String> normalizedChannels = WebhookChannels.normalize(channels);
        normalizedFilterTypes.forEach(this::ensureEventType);
        ObjectNode body = jsonMapper().createObjectNode();
        body.put("url", url);
        body.put("description", description == null ? "tenant destination" : description);
        if (!normalizedFilterTypes.isEmpty()) {
            ArrayNode filter = body.putArray("filterTypes");
            normalizedFilterTypes.forEach(filter::add);
        }
        if (!normalizedChannels.isEmpty()) {
            ArrayNode channelArray = body.putArray("channels");
            normalizedChannels.forEach(channelArray::add);
        }
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
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("endpointId", endpointId);
        response.put("secret", secret);
        response.put("url", created.get("url").asText());
        response.put("filterTypes", normalizedFilterTypes);
        response.put("channels", normalizedChannels);
        return response;
    }

    public List<Map<String, Object>> listEndpoints(String appId) {
        JsonNode node = restClient().get()
                .uri("/api/v1/app/{appId}/endpoint/?limit=50", appId)
                .retrieve().body(JsonNode.class);
        List<Map<String, Object>> out = new ArrayList<>();
        if (node != null && node.has("data")) {
            for (JsonNode ep : node.get("data")) {
                List<String> filterTypes = new ArrayList<>();
                if (ep.has("filterTypes") && ep.get("filterTypes").isArray()) {
                    ep.get("filterTypes").forEach(t -> filterTypes.add(t.asText()));
                }
                List<String> channels = new ArrayList<>();
                if (ep.has("channels") && ep.get("channels").isArray()) {
                    ep.get("channels").forEach(c -> channels.add(c.asText()));
                }
                out.add(Map.of(
                        "endpointId", ep.get("id").asText(),
                        "url", ep.get("url").asText(),
                        "filterTypes", filterTypes,
                        "channels", channels,
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

    public String publish(String appId, String eventType, Map<String, Object> payload, List<String> channels) {
        ensureEventType(eventType);
        ObjectNode payloadNode = jsonMapper().valueToTree(payload == null ? Map.of() : payload);
        ObjectNode body = jsonMapper().createObjectNode();
        body.put("eventType", eventType);
        body.set("payload", payloadNode);
        List<String> normalizedChannels = WebhookChannels.normalize(channels);
        if (!normalizedChannels.isEmpty()) {
            ArrayNode channelArray = body.putArray("channels");
            normalizedChannels.forEach(channelArray::add);
        }
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
                        "eventType", msg.get("eventType").asText(),
                        "timestamp", msg.get("timestamp").asText()));
            }
        }
        return out;
    }

    /**
     * Delivery status of one message across all of the tenant's endpoints.
     * Svix exposes a human-readable `statusText` (success/pending/failed/…)
     * alongside the numeric enum — we surface the words, not the number.
     */
    public List<Map<String, Object>> listAttempts(String appId, String messageId) {
        JsonNode node = restClient().get()
                .uri("/api/v1/app/{appId}/msg/{messageId}/endpoint/?limit=50", appId, messageId)
                .retrieve().body(JsonNode.class);
        List<Map<String, Object>> out = new ArrayList<>();
        if (node != null && node.has("data")) {
            for (JsonNode ep : node.get("data")) {
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("endpointId", ep.path("id").asText(""));
                row.put("url", ep.path("url").asText(""));
                row.put("status", ep.path("statusText").asText("unknown"));
                if (ep.hasNonNull("nextAttempt")) {
                    row.put("nextAttempt", ep.get("nextAttempt").asText());
                }
                if (ep.hasNonNull("disabled") && ep.get("disabled").asBoolean()) {
                    row.put("disabled", true);
                }
                out.add(row);
            }
        }
        return out;
    }

    private List<String> normalizeStrings(List<String> values) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        Set<String> normalized = new LinkedHashSet<>();
        for (String value : values) {
            if (value == null) {
                continue;
            }
            String trimmed = value.trim();
            if (!trimmed.isEmpty()) {
                normalized.add(trimmed);
            }
        }
        return List.copyOf(normalized);
    }
}
