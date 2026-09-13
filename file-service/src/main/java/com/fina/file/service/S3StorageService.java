package com.fina.file.service;

import com.fina.file.config.StorageProperties;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.CreateBucketRequest;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.HeadBucketRequest;
import software.amazon.awssdk.services.s3.model.NoSuchBucketException;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

import java.io.InputStream;
import java.nio.file.Path;

/**
 * Thin wrapper over the S3 client. Storage objects are immutable: keys carry
 * a content-hash suffix and are never overwritten or deleted by this service.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class S3StorageService {

    private final S3Client s3Client;
    private final StorageProperties props;

    @PostConstruct
    void ensureBucket() {
        try {
            String bucket = props.getBucket();
            try {
                s3Client.headBucket(HeadBucketRequest.builder().bucket(bucket).build());
            } catch (NoSuchBucketException e) {
                s3Client.createBucket(CreateBucketRequest.builder().bucket(bucket).build());
                log.info("created bucket {}", bucket);
            }
        } catch (Exception e) {
            // Storage may not be reachable yet (e.g. compose dependency order);
            // fail only when actually used so the service can still boot.
            log.warn("bucket check/creation failed (will retry on first use): {}", e.getMessage());
        }
    }

    public void put(String key, Path file, long size, String mime) {
        s3Client.putObject(PutObjectRequest.builder()
                        .bucket(props.getBucket())
                        .key(key)
                        .contentType(mime != null ? mime : "application/octet-stream")
                        .contentLength(size)
                        .build(),
                RequestBody.fromFile(file));
    }

    public InputStream get(String key) {
        return s3Client.getObject(GetObjectRequest.builder()
                .bucket(props.getBucket())
                .key(key)
                .build());
    }
}
