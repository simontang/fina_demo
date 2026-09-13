package com.fina.platform.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * One immutable version of one file at one logical path. Never updated in
 * place: a re-upload appends version+1 (or is deduplicated on identical
 * content). Soft delete flips `status`, storage objects are retained.
 *
 * tenant_id is filled transparently by the TenantLine interceptor on INSERT
 * and appended to every query — the field exists here only for receipts.
 */
@Data
@TableName("file_objects")
public class FileObject {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String tenantId;

    /** Directory portion of the logical path, no leading/trailing slash. */
    private String path;

    private String filename;

    private Integer version;

    private String sha256;

    private String md5;

    private Long size;

    private String mime;

    private String fileCategory;

    private String usage;

    private String uuid;

    private String meta;

    private String storageKey;

    /** active | deleted */
    private String status;

    @TableField(fill = FieldFill.INSERT)
    private String createdBy;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;

    public String fullPath() {
        return path == null || path.isBlank() ? filename : path + "/" + filename;
    }
}
