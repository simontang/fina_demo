package com.fina.platform.dto;

import com.fina.platform.entity.FileObject;
import lombok.Builder;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * Upload/listing receipt. `fullPath` is the canonical address
 * ({path}/{filename}); id and uuid exist as compatibility aliases.
 */
@Data
@Builder
public class FileReceipt {

    private Long id;
    private String uuid;
    private String tenantId;
    private String fullPath;
    private String path;
    private String filename;
    private Integer version;
    private String sha256;
    private String md5;
    private Long size;
    private String mime;
    private String fileCategory;
    private String usage;
    private String meta;
    private String status;
    private String createdBy;
    private LocalDateTime createdAt;
    /** true when an identical file already existed at this path (no new version). */
    private boolean deduplicated;

    public static FileReceipt from(FileObject o, boolean deduplicated) {
        return FileReceipt.builder()
                .id(o.getId())
                .uuid(o.getUuid())
                .tenantId(o.getTenantId())
                .fullPath(o.fullPath())
                .path(o.getPath())
                .filename(o.getFilename())
                .version(o.getVersion())
                .sha256(o.getSha256())
                .md5(o.getMd5())
                .size(o.getSize())
                .mime(o.getMime())
                .fileCategory(o.getFileCategory())
                .usage(o.getUsage())
                .meta(o.getMeta())
                .status(o.getStatus())
                .createdBy(o.getCreatedBy())
                .createdAt(o.getCreatedAt())
                .deduplicated(deduplicated)
                .build();
    }
}
