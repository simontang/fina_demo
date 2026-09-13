package com.fina.platform.service;

import com.fina.platform.config.StorageProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.util.regex.Pattern;

/**
 * Decides which download-link flavour a request gets.
 *
 * Rule of thumb (mode=auto): if the downloader could plausibly reach the
 * storage endpoint directly, hand out a storage-native presigned URL
 * (bandwidth offloads to TOS/S3). If storage lives on an internal host
 * (e.g. the compose-internal document-minio) a presigned URL would be
 * useless to the caller — return our own platform link instead. Setting
 * PUBLIC_FILE_BASE_URL signals that the storage is published behind a
 * known public host, which makes presign viable again.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class LinkModeResolver {

    private static final Pattern PRIVATE_IP = Pattern.compile(
            "^(10\\..*|192\\.168\\..*|172\\.(1[6-9]|2\\d|3[01])\\..*|127\\..*|0\\.0\\.0\\.0|\\[?::1\\]?)$");

    private final StorageProperties storage;

    @Value("${file.link.mode:auto}")
    private String mode;

    @Value("${file.link.public-base-url:}")
    private String publicBaseUrl;

    /** true → presign, false → our own ticket link. */
    public boolean usePresign() {
        if ("presign".equalsIgnoreCase(mode)) {
            return true;
        }
        if ("ticket".equalsIgnoreCase(mode)) {
            return false;
        }
        // auto
        if (publicBaseUrl != null && !publicBaseUrl.isBlank()) {
            return true;   // storage is published behind a public host
        }
        return isPubliclyReachable(storage.getEndpoint());
    }

    private boolean isPubliclyReachable(String endpoint) {
        if (endpoint == null || endpoint.isBlank()) {
            return false;
        }
        try {
            String host = URI.create(endpoint).getHost();
            if (host == null || host.isBlank()) {
                return false;
            }
            host = host.toLowerCase();
            if ("localhost".equals(host) || PRIVATE_IP.matcher(host).matches()) {
                return false;
            }
            // container/service names and bare internal hostnames carry no dot
            return host.contains(".")
                    && !host.endsWith(".internal")
                    && !host.endsWith(".local");
        } catch (Exception e) {
            log.debug("cannot parse storage endpoint {}: {}", endpoint, e.getMessage());
            return false;
        }
    }

    /** Rewrite a presigned URL's host when storage is published elsewhere. */
    public String rewritePublic(String url) {
        if (publicBaseUrl == null || publicBaseUrl.isBlank()) {
            return url;
        }
        try {
            URI base = URI.create(publicBaseUrl);
            URI original = URI.create(url);
            String rewritten = new URI(base.getScheme(), null, base.getHost(),
                    base.getPort(), original.getPath(), original.getQuery(), null).toString();
            log.debug("rewrote presigned url host {} -> {}", original.getHost(), base.getHost());
            return rewritten;
        } catch (Exception e) {
            log.warn("presigned url rewrite failed, returning original: {}", e.getMessage());
            return url;
        }
    }
}
