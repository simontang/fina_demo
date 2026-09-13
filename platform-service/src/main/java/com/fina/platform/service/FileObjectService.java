package com.fina.platform.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fina.platform.dto.FileReceipt;
import com.fina.platform.dto.PathListing;
import com.fina.platform.entity.FileObject;
import com.fina.platform.exception.ApiException;
import com.fina.platform.mapper.FileObjectMapper;
import com.fina.platform.tenant.TenantContextHolder;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import software.amazon.awssdk.utils.IoUtils;

import java.io.IOException;
import java.io.InputStream;
import java.io.PushbackInputStream;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.TreeSet;
import java.util.UUID;

/**
 * Core file logic: immutable versioning by logical path, content dedupe,
 * pseudo-directory listing. Tenant comes exclusively from
 * TenantContextHolder; no method takes a tenant argument.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class FileObjectService {

    private static final long MAX_NAME_LENGTH = 255;

    private final FileObjectMapper mapper;
    private final S3StorageService storage;

    public FileReceipt upload(MultipartFile file, String pathDir, String fileName,
                              String fileCategory, String usage, String meta) throws IOException {
        if (file == null || file.isEmpty()) {
            throw ApiException.badRequest("file part is empty");
        }
        String[] parts = splitLogicalPath(pathDir, fileName != null && !fileName.isBlank()
                ? fileName : file.getOriginalFilename());
        return store(parts[0], parts[1], fileCategory, usage, meta,
                file.getContentType(), null, file.getInputStream());
    }

    /** Core store: hashes, dedupes, versions and persists one object. */
    public FileReceipt store(String pathDir, String fileName, String fileCategory, String usage, String meta,
                             String contentType, String explicitUuid, InputStream stream) {
        if (stream == null) {
            throw ApiException.badRequest("empty stream");
        }
        String tenant = requireTenant();
        String dir = normalizeDir(pathDir);
        String name = sanitizeFilename(fileName);

        Path temp = null;
        try {
            Hashes hashes;
            long size;
            Path tmp = temp = Files.createTempFile("upload-", ".bin");
            try (InputStream in = stream) {
                MessageDigest sha = MessageDigest.getInstance("SHA-256");
                DigestInputStream shaIn = new DigestInputStream(in, sha);
                size = IoUtils.copy(shaIn, Files.newOutputStream(tmp));
                if (size == 0) {
                    throw ApiException.badRequest("file is empty");
                }
                hashes = new Hashes(HexFormat.of().formatHex(sha.digest()), null);
            }
            try (InputStream in = Files.newInputStream(tmp)) {
                MessageDigest md5 = MessageDigest.getInstance("MD5");
                DigestInputStream md5In = new DigestInputStream(in, md5);
                IoUtils.copy(md5In, java.io.OutputStream.nullOutputStream());
                hashes = new Hashes(hashes.sha256(), HexFormat.of().formatHex(md5.digest()));
            }

            FileObject latest = findLatest(dir, name);
            if (latest != null && hashes.sha256().equals(latest.getSha256())) {
                log.info("deduplicated upload {} (identical content at version {})",
                        latest.fullPath(), latest.getVersion());
                return FileReceipt.from(latest, true);
            }

            // Discrete storage: the object IS its uuid. The key is flat
            // ({tenant}/{uuid}) — S3-compatible storage has no directories, so
            // shard prefixes buy nothing here; the logical path lives purely
            // as metadata, which is what makes renames free.
            String uuid = explicitUuid != null && !explicitUuid.isBlank()
                    ? explicitUuid : UUID.randomUUID().toString().replace("-", "");
            String storageKey = tenant + "/" + uuid;
            storage.put(storageKey, tmp, size, contentType);

            FileObject row = new FileObject();
            row.setTenantId(tenant);
            row.setPath(dir);
            row.setFilename(name);
            row.setVersion(latest == null ? 1 : latest.getVersion() + 1);
            row.setSha256(hashes.sha256());
            row.setMd5(hashes.md5());
            row.setSize(size);
            row.setMime(contentType);
            row.setFileCategory(fileCategory);
            row.setUsage(usage);
            row.setUuid(uuid);
            row.setMeta(meta);
            row.setStorageKey(storageKey);
            row.setStatus("active");
            mapper.insert(row);
            log.info("stored {} v{} at discrete key {}", row.fullPath(), row.getVersion(), storageKey);
            return FileReceipt.from(row, false);
        } catch (Exception e) {
            if (e instanceof ApiException api) {
                throw api;
            }
            throw new ApiException(500, "UPLOAD_FAILED", String.valueOf(e.getMessage()));
        } finally {
            if (temp != null) {
                try {
                    Files.deleteIfExists(temp);
                } catch (IOException ignored) {
                }
            }
        }
    }

    public void download(String uuid, boolean bom,
                         jakarta.servlet.http.HttpServletResponse response) throws IOException {
        downloadRow(resolveByUuid(uuid), bom, response);
    }

    public void downloadByPath(String fullPath, Integer version, boolean bom,
                               jakarta.servlet.http.HttpServletResponse response) throws IOException {
        downloadRow(resolveByPath(fullPath, version), bom, response);
    }

    /** Stream one row (used by both path download and ticket redemption). */
    public void downloadRow(FileObject row, boolean bom,
                            jakarta.servlet.http.HttpServletResponse response) throws IOException {
        boolean addBom = bom && isTextLike(row);

        response.setContentType(row.getMime() != null ? row.getMime() : "application/octet-stream");
        String encoded = URLEncoder.encode(row.getFilename(), StandardCharsets.UTF_8);
        response.setHeader("Content-Disposition", "Attachment;Filename*=utf-8''" + encoded);
        response.setHeader("X-File-Version", String.valueOf(row.getVersion()));

        try (InputStream raw = storage.get(row.getStorageKey())) {
            var out = response.getOutputStream();
            if (addBom) {
                PushbackInputStream pb = new PushbackInputStream(raw, 3);
                byte[] head = pb.readNBytes(3);
                boolean hasBom = head.length == 3 && head[0] == (byte) 0xEF
                        && head[1] == (byte) 0xBB && head[2] == (byte) 0xBF;
                if (!hasBom) {
                    out.write(new byte[]{(byte) 0xEF, (byte) 0xBB, (byte) 0xBF});
                }
                if (head.length > 0) {
                    pb.unread(head);
                }
                pb.transferTo(out);
            } else {
                raw.transferTo(out);
            }
            out.flush();
        }
    }

    /** Max files per page; caller may request fewer, never more. */
    public static final int MAX_PAGE_SIZE = 1000;
    private static final int MAX_DIRECTORIES = 1000;

    /**
     * One entry point for "find files under a folder": scope to a path,
     * optionally recurse into descendants, optionally filter by name
     * substring / category / usage / time range. Newest first, keyset
     * paginated (the cursor is the previous page's last id, base64-wrapped).
     * Non-recursive calls also return the next-level folder names so a UI can
     * drill down.
     */
    public PathListing query(String path, Boolean recursive, String q, String fileCategory,
                             String usage, String from, String to, Integer limit, String cursor) {
        String dir = normalizeDir(path);
        boolean recurse = Boolean.TRUE.equals(recursive);
        int pageSize = limit == null ? 100 : Math.min(Math.max(limit, 1), MAX_PAGE_SIZE);

        List<FileObject> rows = mapper.query(
                dir,
                dir.isEmpty() ? "%" : dir + "/%",
                recurse,
                blankToNull(q),
                blankToNull(fileCategory),
                blankToNull(usage),
                parseDate(from, false),
                parseDate(to, true),
                decodeCursor(cursor),
                pageSize + 1);
        boolean truncated = rows.size() > pageSize;
        if (truncated) {
            rows = rows.subList(0, pageSize);
        }

        List<String> directories = List.of();
        if (!recurse) {
            int startPos = dir.isEmpty() ? 1 : dir.length() + 2;
            List<String> found = mapper.listSubdirectories(
                    startPos, dir.isEmpty() ? "%" : dir + "/%", MAX_DIRECTORIES + 1);
            if (found.size() > MAX_DIRECTORIES) {
                directories = found.subList(0, MAX_DIRECTORIES);
                truncated = true;
            } else {
                directories = found;
            }
        }

        return PathListing.builder()
                .path(dir)
                .recursive(recurse)
                .query(blankToNull(q))
                .directories(directories)
                .files(rows.stream().map(r -> FileReceipt.from(r, false)).toList())
                .limit(pageSize)
                .truncated(truncated)
                .nextCursor(rows.isEmpty() ? null : encodeCursor(rows.get(rows.size() - 1).getId()))
                .build();
    }

    /** Accepts YYYY-MM-DD (or full ISO timestamp). endOfDay widens a bare
     *  date to its last instant so `to=2026-09-30` includes that day. */
    private java.time.LocalDateTime parseDate(String value, boolean endOfDay) {
        if (value == null || value.isBlank()) {
            return null;
        }
        String v = value.trim();
        try {
            if (v.length() == 10) {
                java.time.LocalDate d = java.time.LocalDate.parse(v);
                return endOfDay ? d.atTime(23, 59, 59, 999_000_000) : d.atStartOfDay();
            }
            return java.time.LocalDateTime.parse(v.replace(' ', 'T'));
        } catch (Exception e) {
            throw ApiException.badRequest("invalid date: " + value + " (expected YYYY-MM-DD)");
        }
    }

    private String encodeCursor(Long id) {
        return id == null ? null
                : java.util.Base64.getUrlEncoder().withoutPadding()
                        .encodeToString(String.valueOf(id).getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }

    private Long decodeCursor(String cursor) {
        if (cursor == null || cursor.isBlank()) {
            return null;
        }
        try {
            return Long.valueOf(new String(java.util.Base64.getUrlDecoder().decode(cursor),
                    java.nio.charset.StandardCharsets.UTF_8));
        } catch (Exception e) {
            throw ApiException.badRequest("invalid cursor");
        }
    }

    private String blankToNull(String v) {
        return v == null || v.isBlank() ? null : v.trim();
    }

    public FileReceipt receiptByPath(String fullPath, Integer version) {
        return FileReceipt.from(resolveByPath(fullPath, version), false);
    }

    /** Tenant-scoped lookup regardless of status (used by PUT to inherit
     *  metadata for an existing address). null when unknown. */
    public FileReceipt findAnyByUuid(String uuid) {
        FileObject row = mapper.selectOne(new LambdaQueryWrapper<FileObject>()
                .eq(FileObject::getUuid, uuid)
                .orderByDesc(FileObject::getVersion)
                .last("LIMIT 1"));
        return row == null ? null : FileReceipt.from(row, false);
    }

    public FileReceipt receiptByUuid(String uuid) {
        return FileReceipt.from(resolveByUuid(uuid), false);
    }

    public String storageKeyByUuid(String uuid) {
        return resolveByUuid(uuid).getStorageKey();
    }

    public String storageKeyByPath(String path, Integer version) {
        return resolveByPath(path, version).getStorageKey();
    }

    /** Tenant-exempt lookup (download tickets / presign without a header). */
    public FileObject activeRowByUuidIgnoreTenant(String uuid) {
        return mapper.selectActiveByUuidIgnoreTenant(uuid);
    }

    /** Soft delete; uuid pins the exact version, so no version arg is needed. */
    public int delete(String uuid) {
        FileObject row = mapper.selectOne(new LambdaQueryWrapper<FileObject>()
                .eq(FileObject::getUuid, uuid)
                .eq(FileObject::getStatus, "active"));
        if (row == null) {
            return 0;
        }
        row.setStatus("deleted");
        return mapper.updateById(row);
    }

    // ── helpers ─────────────────────────────────────────────────────────

    /** uuids are per stored version, so they pin exactly one row. */
    private FileObject resolveByUuid(String uuid) {
        return requireRow(new LambdaQueryWrapper<FileObject>()
                .eq(FileObject::getUuid, uuid)
                .eq(FileObject::getStatus, "active"));
    }

    private FileObject resolveByPath(String fullPath, Integer version) {
        String[] parts = splitPath(normalizeDir(fullPath));
        LambdaQueryWrapper<FileObject> qw = new LambdaQueryWrapper<FileObject>()
                .eq(FileObject::getPath, parts[0])
                .eq(FileObject::getFilename, parts[1])
                .eq(FileObject::getStatus, "active");
        if (version != null) {
            qw.eq(FileObject::getVersion, version);
        }
        return requireRow(qw.orderByDesc(FileObject::getVersion).last("LIMIT 1"));
    }

    private FileObject findLatest(String dir, String name) {
        return mapper.selectOne(new LambdaQueryWrapper<FileObject>()
                .eq(FileObject::getPath, dir)
                .eq(FileObject::getFilename, name)
                .orderByDesc(FileObject::getVersion)
                .last("LIMIT 1"));
    }

    private FileObject requireRow(LambdaQueryWrapper<FileObject> qw) {
        FileObject row = mapper.selectOne(qw);
        if (row == null) {
            throw ApiException.notFound("no active file matches the given address");
        }
        return row;
    }

    private String requireTenant() {
        String tenant = TenantContextHolder.getTenant();
        if (tenant == null || tenant.isBlank()) {
            throw new IllegalStateException("tenant context is missing");
        }
        return tenant;
    }

    /** Directory segment: no leading/trailing slashes, no "..", no backslashes. */
    String normalizeDir(String raw) {
        if (raw == null) {
            return "";
        }
        String cleaned = raw.trim().replaceAll("/+", "/").replaceAll("^/+|/+$", "");
        if (cleaned.contains("..") || cleaned.contains("\\")) {
            throw ApiException.badRequest("path must not contain '..' or backslash");
        }
        if (cleaned.length() > 512) {
            throw ApiException.badRequest("path too long");
        }
        return cleaned;
    }

    /** Resolve a request into {directory, filename}: fileName wins when
     *  given, otherwise fullPath is the complete logical path. */
    public String[] splitLogicalPath(String pathDir, String fileName) {
        if (fileName != null && !fileName.isBlank()) {
            return new String[]{normalizeDir(pathDir), sanitizeFilename(fileName)};
        }
        return splitPath(normalizeDir(pathDir));
    }

    /** Full logical path → {directory, filename}. */
    private String[] splitPath(String normalizedFullPath) {
        int idx = normalizedFullPath.lastIndexOf('/');
        String dir = idx >= 0 ? normalizedFullPath.substring(0, idx) : "";
        String filename = idx >= 0 ? normalizedFullPath.substring(idx + 1) : normalizedFullPath;
        if (filename.isBlank()) {
            throw ApiException.badRequest("path must point to a file, not a directory");
        }
        return new String[]{dir, filename};
    }

    private String sanitizeFilename(String raw) {
        if (raw == null || raw.isBlank()) {
            throw ApiException.badRequest("fileName is required");
        }
        String cleaned = raw.trim().replaceAll("[/\\\\]", "_");
        if (cleaned.length() > MAX_NAME_LENGTH) {
            throw ApiException.badRequest("fileName too long");
        }
        return cleaned;
    }

    private boolean isTextLike(FileObject row) {
        String mime = row.getMime();
        String name = row.getFilename().toLowerCase();
        return (mime != null && (mime.contains("text") || mime.contains("csv")))
                || name.endsWith(".csv") || name.endsWith(".txt") || name.endsWith(".tsv");
    }

    private record Hashes(String sha256, String md5) {
    }
}
