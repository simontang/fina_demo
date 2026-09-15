package com.fina.platform.webhooks;

import com.fina.platform.tenant.TenantContextHolder;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Tenant-aware facade over Svix. Tenant always comes from
 * TenantContextHolder (transparent multi-tenancy); callers address
 * everything in Svix-compatible event and destination vocabulary — never in
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

    public Map<String, Object> createDestination(
            String url,
            List<String> filterTypes,
            List<String> channels,
            String description) {
        return svix.createEndpoint(appId(), url, filterTypes, channels, description);
    }

    public List<Map<String, Object>> listDestinations() {
        return svix.listEndpoints(appId());
    }

    public void deleteDestination(String endpointId) {
        svix.deleteEndpoint(appId(), endpointId);
    }

    public Map<String, Object> publish(String eventType, Map<String, Object> payload, List<String> channels) {
        List<String> normalizedChannels = WebhookChannels.normalize(channels);
        String messageId = svix.publish(appId(), eventType, payload, normalizedChannels);
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("messageId", messageId);
        response.put("eventType", eventType);
        response.put("channels", normalizedChannels);
        return response;
    }

    public List<Map<String, Object>> messages(int limit) {
        return svix.listMessages(appId(), limit);
    }

    public List<Map<String, Object>> attempts(String messageId) {
        return svix.listAttempts(appId(), messageId);
    }
}
