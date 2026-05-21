package com.fina.b1s.tos;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "tos")
public record TosProperties(
        boolean enabled,
        String endpoint,
        String region,
        String accessKey,
        String secretKey,
        String bucket,
        String keyPrefix
) {
}
