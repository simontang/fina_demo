package com.fina.b1s.tos;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.S3Configuration;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

import java.io.InputStream;
import java.net.URI;

@Slf4j
@Service
public class TosStorageServiceImpl implements TosStorageService {

    private final TosProperties properties;
    private volatile S3Client client;

    public TosStorageServiceImpl(TosProperties properties) {
        this.properties = properties;
    }

    @Override
    public UploadResult upload(String key, InputStream content, long contentLength) {
        if (!isConfigured()) {
            throw new IllegalStateException("TOS is not configured");
        }
        PutObjectRequest request = PutObjectRequest.builder()
                .bucket(properties.bucket())
                .key(key)
                .contentLength(contentLength)
                .build();
        getClient().putObject(request, RequestBody.fromInputStream(content, contentLength));
        return new UploadResult(properties.bucket(), key, buildObjectUrl(key));
    }

    private S3Client getClient() {
        S3Client existing = client;
        if (existing != null) {
            return existing;
        }
        synchronized (this) {
            if (client == null) {
                client = S3Client.builder()
                        .endpointOverride(URI.create(properties.endpoint()))
                        .region(Region.of(properties.region()))
                        .credentialsProvider(StaticCredentialsProvider.create(
                                AwsBasicCredentials.create(properties.accessKey(), properties.secretKey())))
                        .serviceConfiguration(S3Configuration.builder()
                                .pathStyleAccessEnabled(false)
                                .build())
                        .build();
                log.info("Initialized TOS client endpoint={} bucket={}", properties.endpoint(), properties.bucket());
            }
            return client;
        }
    }

    private boolean isConfigured() {
        return properties.enabled()
                && StringUtils.hasText(properties.endpoint())
                && StringUtils.hasText(properties.region())
                && StringUtils.hasText(properties.accessKey())
                && StringUtils.hasText(properties.secretKey())
                && StringUtils.hasText(properties.bucket());
    }

    private String buildObjectUrl(String key) {
        String endpoint = properties.endpoint();
        if (!StringUtils.hasText(endpoint)) {
            return null;
        }
        String normalized = endpoint.replaceFirst("^https?://", "");
        return "https://" + properties.bucket() + "." + normalized + "/" + key;
    }
}
