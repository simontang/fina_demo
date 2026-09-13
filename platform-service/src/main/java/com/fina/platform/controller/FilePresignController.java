package com.fina.platform.controller;

import com.fina.platform.config.StorageProperties;
import com.fina.platform.exception.ApiException;
import com.fina.platform.service.FileObjectService;
import com.fina.platform.service.LinkModeResolver;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;
import software.amazon.awssdk.services.s3.presigner.model.GetObjectPresignRequest;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * One endpoint for "get me a download URL". Cloud/reachable storage →
 * storage-native presigned URL (bandwidth bypasses this service); internal
 * storage (self-hosted MinIO) → our own download URL, since a presigned URL
 * pointing at an internal host would be useless to the caller.
 */
@RestController
@RequestMapping("/api/v1/files")
@RequiredArgsConstructor
public class FilePresignController {

    private static final long DEFAULT_TTL = 3600;
    private static final long MAX_TTL = 604800;

    private final FileObjectService fileObjectService;
    private final LinkModeResolver modeResolver;
    private final S3Presigner presigner;
    private final StorageProperties storage;

    @org.springframework.beans.factory.annotation.Value("${file.link.public-base-url:}")
    private String publicBaseUrl;

    public record PresignRequest(String path, Integer version, Long ttlSeconds) {
    }

    @PostMapping("/presign")
    public Map<String, Object> presign(@RequestBody PresignRequest req, HttpServletRequest request) {
        if (req.path() == null || req.path().isBlank()) {
            throw ApiException.badRequest("path is required");
        }
        long ttl = req.ttlSeconds() == null ? DEFAULT_TTL
                : Math.min(Math.max(req.ttlSeconds(), 1), MAX_TTL);

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("path", req.path());
        out.put("expiresInSeconds", ttl);

        if (!modeResolver.usePresign()) {
            // self-hosted storage: our own download URL (tenant header applies)
            out.put("url", directUrl(req, request));
            out.put("kind", "direct");
            return out;
        }

        String storageKey = fileObjectService.storageKeyByPath(req.path(), req.version());
        String url = presigner.presignGetObject(GetObjectPresignRequest.builder()
                        .signatureDuration(Duration.ofSeconds(ttl))
                        .getObjectRequest(GetObjectRequest.builder()
                                .bucket(storage.getBucket())
                                .key(storageKey)
                                .build())
                        .build())
                .url().toString();
        out.put("url", modeResolver.rewritePublic(url));
        out.put("kind", "presigned");
        out.put("expiresAt", Instant.now().plusSeconds(ttl).toString());
        return out;
    }

    /** Absolute URL of our own download endpoint for this key. */
    private String directUrl(PresignRequest req, HttpServletRequest request) {
        String base = (publicBaseUrl != null && !publicBaseUrl.isBlank())
                ? publicBaseUrl.replaceAll("/+$", "")
                : request.getScheme() + "://" + request.getServerName()
                        + (isDefaultPort(request) ? "" : ":" + request.getServerPort());
        String key = req.path().replaceAll("^/+", "");
        StringBuilder url = new StringBuilder(base)
                .append("/api/v1/files/")
                .append(encodePath(key));
        if (req.version() != null) {
            url.append("?version=").append(req.version());
        }
        return url.toString();
    }

    private boolean isDefaultPort(HttpServletRequest request) {
        int port = request.getServerPort();
        return ("http".equals(request.getScheme()) && port == 80)
                || ("https".equals(request.getScheme()) && port == 443);
    }

    /** Percent-encode each segment but keep '/' separators. */
    private String encodePath(String key) {
        StringBuilder sb = new StringBuilder();
        for (String segment : key.split("/")) {
            if (sb.length() > 0) {
                sb.append('/');
            }
            sb.append(URLEncoder.encode(segment, StandardCharsets.UTF_8).replace("+", "%20"));
        }
        return sb.toString();
    }
}
