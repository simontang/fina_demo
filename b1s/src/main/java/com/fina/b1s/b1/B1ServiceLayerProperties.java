package com.fina.b1s.b1;

import java.time.Duration;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "b1.service-layer")
public record B1ServiceLayerProperties(
        String baseUrl,
        String defaultCompanyDb,
        String defaultUsername,
        String defaultPassword,
        Duration connectTimeout,
        Duration readTimeout) {
}
