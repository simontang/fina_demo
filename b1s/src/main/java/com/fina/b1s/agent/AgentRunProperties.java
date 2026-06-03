package com.fina.b1s.agent;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.util.StringUtils;

@ConfigurationProperties(prefix = "agent.run")
public record AgentRunProperties(
        boolean enabled,
        String baseUrl,
        String bearerToken,
        String tenantId,
        String channelInstallationId,
        long connectTimeoutMs,
        long readTimeoutMs
) {

    public boolean isConfigured() {
        return enabled
                && StringUtils.hasText(baseUrl)
                && StringUtils.hasText(bearerToken)
                && StringUtils.hasText(tenantId)
                && StringUtils.hasText(channelInstallationId);
    }
}
