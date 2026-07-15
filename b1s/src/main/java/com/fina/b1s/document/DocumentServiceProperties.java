package com.fina.b1s.document;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.util.StringUtils;

@ConfigurationProperties(prefix = "document-service")
public record DocumentServiceProperties(
        boolean enabled,
        String baseUrl,
        String engine,
        String mode,
        String languageHints,
        long connectTimeoutMs,
        long readTimeoutMs,
        long pollIntervalMs,
        long maxWaitMs
) {

    public boolean isConfigured() {
        return enabled && StringUtils.hasText(baseUrl);
    }
}
