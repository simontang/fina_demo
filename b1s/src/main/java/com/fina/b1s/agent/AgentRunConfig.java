package com.fina.b1s.agent;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

import java.time.Duration;

@Configuration
@EnableConfigurationProperties(AgentRunProperties.class)
public class AgentRunConfig {

    @Bean
    public RestTemplate agentRunRestTemplate(RestTemplateBuilder builder, AgentRunProperties properties) {
        return builder
                .setConnectTimeout(Duration.ofMillis(timeoutOrDefault(properties.connectTimeoutMs(), 15_000)))
                .setReadTimeout(Duration.ofMillis(timeoutOrDefault(properties.readTimeoutMs(), 600_000)))
                .build();
    }

    private long timeoutOrDefault(long value, long defaultValue) {
        return value > 0 ? value : defaultValue;
    }
}
