package com.fina.b1s.document;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.net.http.HttpClient;
import java.time.Duration;

@Configuration
@EnableConfigurationProperties(DocumentServiceProperties.class)
public class DocumentServiceConfig {

    @Bean
    public HttpClient documentServiceHttpClient(DocumentServiceProperties properties) {
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
