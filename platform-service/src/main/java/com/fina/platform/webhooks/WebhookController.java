package com.fina.platform.webhooks;

import com.fina.platform.exception.ApiException;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Webhook management API in our tenant model: X-Tenant-Id scopes everything,
 * eventTypes/filterTypes/channels follow Svix's model. Svix stays behind this facade.
 */
@RestController
// Two mount points on purpose: the service's canonical path (/api/v1/webhooks)
// and the public prefix nginx exposes (/api/webhooks). The portal page calls
// the public one so the same code works both when it is served directly and
// when it sits behind the reverse proxy.
@RequestMapping({"/api/v1/webhooks", "/api/webhooks"})
@RequiredArgsConstructor
public class WebhookController {

    private final WebhookFacade facade;

    public record CreateDestinationRequest(
            String url,
            List<String> filterTypes,
            List<String> topics,
            List<String> channels,
            String description) {
    }

    @PostMapping("/destinations")
    public Map<String, Object> createDestination(@RequestBody CreateDestinationRequest req) {
        return facade.createDestination(
                req.url(),
                chooseList(req.filterTypes(), req.topics(), "filterTypes", "topics"),
                req.channels(),
                req.description());
    }

    @GetMapping("/destinations")
    public List<Map<String, Object>> listDestinations() {
        return facade.listDestinations();
    }

    @DeleteMapping("/destinations/{endpointId}")
    public Map<String, Object> deleteDestination(@PathVariable String endpointId) {
        facade.deleteDestination(endpointId);
        return Map.of("deleted", endpointId);
    }

    @PostMapping("/publish")
    public Map<String, Object> publish(@RequestBody PublishRequest req) {
        if (req.endpointIds() != null) {
            throw ApiException.badRequest("endpointIds is not supported by the Svix-compatible publish API; use channels");
        }
        String eventType = chooseString(req.eventType(), req.topic(), "eventType", "topic");
        if (eventType == null || eventType.isBlank()) {
            throw ApiException.badRequest("eventType is required");
        }
        return facade.publish(
                eventType,
                choosePayload(req.payload(), req.data()),
                req.channels());
    }

    public record PublishRequest(
            String eventType,
            Map<String, Object> payload,
            List<String> channels,
            String topic,
            Map<String, Object> data,
            List<String> endpointIds) {
    }

    private List<String> chooseList(List<String> primary, List<String> legacy, String primaryName, String legacyName) {
        if (primary != null && legacy != null && !Objects.equals(primary, legacy)) {
            throw ApiException.badRequest("send either " + primaryName + " or deprecated " + legacyName + ", not both");
        }
        return primary != null ? primary : legacy;
    }

    private String chooseString(String primary, String legacy, String primaryName, String legacyName) {
        if (primary != null && legacy != null && !Objects.equals(primary, legacy)) {
            throw ApiException.badRequest("send either " + primaryName + " or deprecated " + legacyName + ", not both");
        }
        return primary != null ? primary : legacy;
    }

    private Map<String, Object> choosePayload(Map<String, Object> primary, Map<String, Object> legacy) {
        if (primary != null && legacy != null && !Objects.equals(primary, legacy)) {
            throw ApiException.badRequest("send either payload or deprecated data, not both");
        }
        return primary != null ? primary : legacy;
    }

    @GetMapping("/messages")
    public List<Map<String, Object>> messages(@RequestParam(value = "limit", defaultValue = "20") int limit) {
        return facade.messages(limit);
    }

    @GetMapping("/messages/{messageId}/attempts")
    public List<Map<String, Object>> attempts(@PathVariable String messageId) {
        return facade.attempts(messageId);
    }
}
