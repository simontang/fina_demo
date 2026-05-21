package com.fina.b1s.llm;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "llm.intent")
public record LlmIntentProperties(
        boolean enabled,
        String endpoint,
        String bearerToken,
        String model,
        int maxTokens,
        String anthropicVersion
) {
}
