package com.fina.file.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fina.file.dto.FileReceipt;
import com.fina.file.dto.PathListing;
import com.fina.file.entity.FileObject;
import com.fina.file.exception.ApiException;
import com.fina.file.mapper.FileObjectMapper;
import com.fina.file.tenant.TenantContextHolder;
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
                              String fileCategory, String usage, String meta) {
        if (file == null || file.isEmpty()) {
            throw ApiException.badRequest("file part is empty");
        }
        String tenant = requireTenant();
        String dir = normalizeDir(pathDir);
        String name = sanitizeFilename(fileName != null && !fileName.isBlank()
                ? fileName : file.getOriginalFilename());

        Path temp = null;
        try {
            Hashes hashes;
            long size;
            Path tmp = temp = Files.createTempFile("upload-", ".bin");
            try (InputStream in = file.getInputStream()) {
                MessageDigest sha = MessageDigest.getInstance("SHA-256");
                DigestInputStream shaIn = new DigestInputStream(in, sha);
                size = IoUtils.copy(shaIn, Files.newOutputStream(tmp));
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

            String storageKey = tenant + "/"
                    + (dir.isEmpty() ? "" : dir + "/")
                    + name + "@" + hashes.sha256().substring(0, 8);
            storage.put(storageKey, tmp, size, file.getContentType());

            FileObject row = new FileObject();
            row.setTenantId(tenant);
            row.setPath(dir);
            row.setFilename(name);
            row.setVersion(latest == null ? 1 : latest.getVersion() + 1);
            row.setSha256(hashes.sha256());
            row.setMd5(hashes.md5());
            row.setSize(size);
            row.setMime(file.getContentType());
            row.setFileCategory(fileCategory);
            row.setUsage(usage);
            row.setUuid(UUID.randomUUID().toString().replace("-", ""));
            row.setMeta(meta);
            row.setStorageKey(storageKey);
            row.setStatus("active");
            mapper.insert(row);
            log.info("stored {} v{} as {}", row.fullPath(), row.getVersion(), storageKey);
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

    public void download(String fullPath, Integer version, boolean bom,
                         jakarta.servlet.http.HttpServletResponse response) throws IOException {
        FileObject row = resolveByPath(fullPath, version);
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

    public PathListing list(String prefix) {
        String dir = normalizeDir(prefix);
        LambdaQueryWrapper<FileObject> qw = new LambdaQueryWrapper<FileObject>()
                .eq(FileObject::getStatus, "active")
                .last("LIMIT 2000");
        if (!dir.isEmpty()) {
            qw.and(w -> w.eq(FileObject::getPath, dir).or().likeRight(FileObject::getPath, dir + "/"));
        }
        List<FileObject> rows = mapper.selectList(qw);

        TreeSet<String> directories = new TreeSet<>();
        List<FileReceipt> files = new ArrayList<>();
        for (FileObject row : rows) {
            if (row.getPath().equals(dir)) {
                files.add(FileReceipt.from(row, false));
            } else {
                String remainder = dir.isEmpty() ? row.getPath() : row.getPath().substring(dir.length() + 1);
                directories.add(remainder.split("/", 2)[0]);
            }
        }
        return PathListing.builder().prefix(dir).directories(new ArrayList<>(directories)).files(files).build();
    }

    public FileReceipt receiptByPath(String fullPath, Integer version) {
        return FileReceipt.from(resolveByPath(fullPath, version), false);
    }

    public FileReceipt receiptById(long id) {
        return FileReceipt.from(requireRow(
                new LambdaQueryWrapper<FileObject>().eq(FileObject::getId, id)), false);
    }

    public FileReceipt receiptByUuid(String uuid) {
        return FileReceipt.from(requireRow(
                new LambdaQueryWrapper<FileObject>().eq(FileObject::getUuid, uuid)), false);
    }

    /** Soft delete: flips status, never touches the storage object or history rows. */
    public int delete(String fullPath, Integer version) {
        String[] parts = splitPath(normalizeDir(fullPath));
        LambdaQueryWrapper<FileObject> qw = new LambdaQueryWrapper<FileObject>()
                .eq(FileObject::getPath, parts[0])
                .eq(FileObject::getFilename, parts[1])
                .eq(FileObject::getStatus, "active");
        if (version != null) {
            qw.eq(FileObject::getVersion, version);
        }
        List<FileObject> rows = mapper.selectList(qw);
        int count = 0;
        for (FileObject row : rows) {
            row.setStatus("deleted");
            count += mapper.updateById(row);
        }
        return count;
    }

    // ── helpers ─────────────────────────────────────────────────────────

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
