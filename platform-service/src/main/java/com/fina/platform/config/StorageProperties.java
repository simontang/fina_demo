package com.fina.platform.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.S3Configuration;

import java.net.URI;

/**
 * S3-compatible storage client (MinIO in the demo stack, any S3 in prod).
 * Env names mirror document_service's OBJECT_STORAGE_* convention.
 */
@Data
@Configuration
@ConfigurationProperties(prefix = "file.storage")
public class StorageProperties {

    private String endpoint;
    private String region;
    private String bucket;
    private String accessKey;
    private String secretKey;
    private boolean forcePathStyle;

    @Bean
    public S3Client s3Client() {
        return S3Client.builder()
                .endpointOverride(URI.create(endpoint))
                .region(Region.of(region == null || region.isBlank() ? "us-east-1" : region))
                .credentialsProvider(StaticCredentialsProvider.create(
                        AwsBasicCredentials.create(accessKey, secretKey)))
                .serviceConfiguration(S3Configuration.builder()
                        .pathStyleAccessEnabled(forcePathStyle)
                        .build())
                .build();
    }
}
