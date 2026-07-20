package com.fina.b1s.b1;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "b1.proxy")
public record B1ProxyProperties(
        boolean logRequestBodyOnError,
        int maxRequestBodyLogChars) {

    public B1ProxyProperties {
        if (maxRequestBodyLogChars <= 0) {
            maxRequestBodyLogChars = 8000;
        }
    }
}
