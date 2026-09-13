package com.fina.platform.webhooks;

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

/**
 * Webhook management API in our tenant model: X-Tenant-Id scopes everything,
 * topics are our factory event names. Svix stays behind this facade.
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

    public record CreateDestinationRequest(String url, List<String> topics, String description) {
    }

    @PostMapping("/destinations")
    public Map<String, Object> createDestination(@RequestBody CreateDestinationRequest req) {
        return facade.createDestination(req.url(), req.topics(), req.description());
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
        return facade.publish(req.topic(), req.data());
    }

    public record PublishRequest(String topic, Map<String, Object> data) {
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
