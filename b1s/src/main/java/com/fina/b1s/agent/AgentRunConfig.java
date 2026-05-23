package com.fina.b1s.agent;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.net.http.HttpClient;
import java.time.Duration;

@Configuration
@EnableConfigurationProperties(AgentRunProperties.class)
public class AgentRunConfig {

    @Bean
    public HttpClient agentRunHttpClient(AgentRunProperties properties) {
        return HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(timeoutOrDefault(properties.connectTimeoutMs(), 15_000)))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .version(HttpClient.Version.HTTP_1_1)
                .build();
    }

    private long timeoutOrDefault(long value, long defaultValue) {
        return value > 0 ? value : defaultValue;
    }
}
