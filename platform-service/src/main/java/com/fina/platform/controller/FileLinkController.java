package com.fina.platform.controller;

import com.fina.platform.dto.FileReceipt;
import com.fina.platform.exception.ApiException;
import com.fina.platform.service.FileLinkService;
import com.fina.platform.service.FileObjectService;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;
import software.amazon.awssdk.services.s3.presigner.model.PresignedGetObjectRequest;

import java.time.Duration;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Download links (获取下载链接): shareable, time-limited, header-free URLs.
 * Two modes — "presign" delegates to the storage's native signed URL
 * (bandwidth offloads to MinIO/TOS; requires a downloader-reachable endpoint),
 * "ticket" streams through platform-service with an HMAC token (works with
 * any storage). Creating a link requires the tenant header; USING one does
 * not — the ticket is the capability.
 */
@RestController
@RequestMapping("/api/v1/files")
@RequiredArgsConstructor
public class FileLinkController {

    private final FileLinkService linkService;
    private final FileObjectService fileObjectService;
    private final S3Presigner presigner;
    private final com.fina.platform.config.StorageProperties storage;

    @Value("${file.link.mode:ticket}")
    private String mode;

    public record LinkRequest(String path, String uuid, Integer version, Long ttlSeconds) {
    }

    @PostMapping("/link")
    public Map<String, Object> link(@RequestBody LinkRequest req) {
        long ttl = req.ttlSeconds() == null ? -1
                : Math.min(Math.max(req.ttlSeconds(), 1), 604800);
        boolean presign = "presign".equalsIgnoreCase(mode);

        if (presign) {
            String storageKey = req.uuid() != null
                    ? fileObjectService.storageKeyByUuid(req.uuid())
                    : fileObjectService.storageKeyByPath(req.path(), req.version());
            PresignedGetObjectRequest pre = presigner.presignGetObject(
                    software.amazon.awssdk.services.s3.presigner.model.GetObjectPresignRequest.builder()
                            .signatureDuration(Duration.ofSeconds(ttl < 0 ? 3600 : ttl))
                            .getObjectRequest(GetObjectRequest.builder()
                                    .bucket(storage.getBucket())
                                    .key(storageKey)
                                    .build())
                            .build());
            return Map.of("url", pre.url().toString(), "kind", "presigned");
        }

        FileLinkService.Ticket t = req.uuid() != null
                ? linkService.ticketByUuid(req.uuid(), req.version(), req.ttlSeconds())
                : linkService.ticketByPath(req.path(), req.version(), req.ttlSeconds());
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("url", linkService.ticketUrl(t.token()));
        out.put("kind", "ticket");
        out.put("expiresAt", Instant.ofEpochSecond(t.expiresAtEpochSeconds()).toString());
        return out;
    }

    /** Header-free download via ticket: the token is the capability. */
    @GetMapping("/ticket/{token}")
    public void ticketDownload(@PathVariable String token,
                               @RequestParam(value = "bom", required = false, defaultValue = "false") boolean bom,
                               jakarta.servlet.http.HttpServletResponse response) throws java.io.IOException {
        FileLinkService.Ticket t = linkService.parse(token);
        if (t == null) {
            throw new ApiException(403, "LINK_INVALID", "download link is invalid");
        }
        if (t.expired()) {
            throw new ApiException(410, "LINK_EXPIRED", "download link has expired");
        }
        var row = fileObjectService.activeRowByUuidIgnoreTenant(t.uuid());
        if (row == null) {
            throw new ApiException(404, "LINK_GONE", "the linked file is no longer available");
        }
        com.fina.platform.tenant.TenantContextHolder.setTenant(row.getTenantId());
        try {
            fileObjectService.downloadRow(row, bom, response);
        } finally {
            com.fina.platform.tenant.TenantContextHolder.clear();
        }
    }
}
