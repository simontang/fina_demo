package com.fina.platform.webhooks;

import com.fina.platform.tenant.TenantContextHolder;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

/**
 * Tenant-aware facade over Svix. Tenant always comes from
 * TenantContextHolder (transparent multi-tenancy); callers address
 * everything in OUR vocabulary — topics and destinations — never in
 * Svix's application/endpoint terms.
 */
@Service
@RequiredArgsConstructor
public class WebhookFacade {

    private final SvixServerClient svix;

    private String appId() {
        String tenant = TenantContextHolder.getTenant();
        if (tenant == null || tenant.isBlank()) {
            throw new IllegalStateException("tenant context is missing");
        }
        return svix.ensureApplication(tenant);
    }

    public Map<String, Object> createDestination(String url, List<String> topics, String description) {
        return svix.createEndpoint(appId(), url, topics, description);
    }

    public List<Map<String, Object>> listDestinations() {
        return svix.listEndpoints(appId());
    }

    public void deleteDestination(String endpointId) {
        svix.deleteEndpoint(appId(), endpointId);
    }

    public Map<String, Object> publish(String topic, Map<String, Object> data) {
        String messageId = svix.publish(appId(), topic, data);
        return Map.of("messageId", messageId, "topic", topic);
    }

    public List<Map<String, Object>> messages(int limit) {
        return svix.listMessages(appId(), limit);
    }

    public List<Map<String, Object>> attempts(String messageId) {
        return svix.listAttempts(appId(), messageId);
    }
}
