package com.fina.b1s.agent;

public record AgentRunResult(
        boolean sent,
        String status,
        String runId,
        String rawResponse,
        String errorMessage
) {
}
