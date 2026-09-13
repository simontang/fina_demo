package com.fina.platform.service;

import com.fina.platform.dto.FileReceipt;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.Base64;

/**
 * Time-limited, unguessable download tickets: HMAC-SHA256 over
 * {key|version|expiry}. The ticket IS the authorization — downloads through
 * it need no tenant header (that is the point of a shareable link).
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class FileLinkService {

    private final FileObjectService fileObjectService;

    @Value("${file.link.secret:file-link-demo-secret-change-me}")
    private String secret;

    @Value("${file.link.default-ttl-seconds:3600}")
    private long defaultTtl;

    @Value("${file.link.max-ttl-seconds:604800}")
    private long maxTtl;

    public record Ticket(String uuid, Integer version, long expiresAtEpochSeconds, String token, boolean expired) {
    }

    /** Issue a ticket for a logical path (latest or specific version). */
    public Ticket ticketByPath(String path, Integer version, Long ttlSeconds) {
        FileReceipt r = fileObjectService.receiptByPath(path, version);
        return build(r.getUuid(), version, ttlSeconds);
    }

    /** Issue a ticket via the uuid handle. */
    public Ticket ticketByUuid(String uuid, Integer version, Long ttlSeconds) {
        FileReceipt r = fileObjectService.receiptByUuid(uuid);
        return build(r.getUuid(), version != null ? version : r.getVersion(), ttlSeconds);
    }

    private Ticket build(String uuid, Integer version, Long ttlSeconds) {
        long ttl = ttlSeconds == null ? defaultTtl : Math.min(Math.max(ttlSeconds, 1), maxTtl);
        long exp = Instant.now().getEpochSecond() + ttl;
        String payload = uuid + "|" + (version == null ? "" : version) + "|" + exp;
        String token = base64Url(payload.getBytes(StandardCharsets.UTF_8))
                + "." + base64Url(hmac(payload));
        return new Ticket(uuid, version, exp, token, false);
    }

    /** Validate a ticket; returns {key, version} or null when invalid/expired. */
    public Ticket parse(String token) {
        try {
            int dot = token.lastIndexOf('.');
            if (dot < 0) {
                return null;
            }
            String payloadB64 = token.substring(0, dot);
            String given = token.substring(dot + 1);
            String payload = new String(Base64.getUrlDecoder().decode(payloadB64), StandardCharsets.UTF_8);
            String[] parts = payload.split("\\|", -1);
            if (parts.length != 3) {
                return null;
            }
            String expected = base64Url(hmac(payload));
            if (!MessageDigest.isEqual(expected.getBytes(StandardCharsets.UTF_8), given.getBytes(StandardCharsets.UTF_8))) {
                return null;
            }
            long exp = Long.parseLong(parts[2]);
            Integer version = parts[1].isBlank() ? null : Integer.valueOf(parts[1]);
            boolean expired = Instant.now().getEpochSecond() > exp;
            return new Ticket(parts[0], version, exp, token, expired);
        } catch (Exception e) {
            log.debug("invalid download ticket: {}", e.getMessage());
            return null;
        }
    }

    /** URL the downloader uses, relative to this service's public base. */
    public String ticketUrl(String token) {
        return "/api/v1/files/ticket/" + URLEncoder.encode(token, StandardCharsets.UTF_8);
    }

    private byte[] hmac(String payload) {
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(secret.getBytes(StandardCharsets.UTF_8), "HmacSHA256"));
            return mac.doFinal(payload.getBytes(StandardCharsets.UTF_8));
        } catch (Exception e) {
            throw new IllegalStateException("hmac failed", e);
        }
    }

    private String base64Url(byte[] bytes) {
        return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
    }
}
