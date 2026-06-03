package com.fina.b1s.agent;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;

@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public record AgentInboundRequest(
        String channel,
        String channelInstallationId,
        String tenantId,
        Sender sender,
        Content content
) {

    public record Sender(String id) {
    }

    public record Content(String text) {
    }
}
