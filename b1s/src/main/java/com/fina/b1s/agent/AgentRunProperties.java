package com.fina.b1s.agent;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "agent.run")
public record AgentRunProperties(
        boolean enabled,
        String baseUrl,
        String bearerToken,
        String tenantId,
        String workspaceId,
        String projectId,
        String assistantId,
        String mode,
        boolean streaming,
        boolean background
) {
}
