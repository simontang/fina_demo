package com.fina.b1s.agent;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;

import java.util.Map;

@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public record AgentRunRequest(
        @JsonProperty("assistant_id")
        String assistantId,
        @JsonProperty("thread_id")
        String threadId,
        String message,
        Boolean streaming,
        Boolean background,
        String mode,
        @JsonProperty("custom_run_config")
        Map<String, Object> customRunConfig
) {
}
