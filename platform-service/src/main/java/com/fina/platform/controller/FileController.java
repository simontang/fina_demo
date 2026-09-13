package com.fina.platform.controller;

import com.fina.platform.dto.FileReceipt;
import com.fina.platform.dto.PathListing;
import com.fina.platform.exception.ApiException;
import com.fina.platform.service.FileObjectService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * File API addressed by uuid. A uuid identifies one stored version, so it
 * pins the object unambiguously — no path encoding, no query-param
 * addressing, and renames never invalidate a reference. The logical path
 * (dir/filename) is metadata: set at upload, used for listing, never an
 * address.
 *
 *   POST   /api/v1/files/upload       multipart upload (server picks uuid)
 *   PUT    /api/v1/files/{uuid}       raw-stream upload (client picks uuid)
 *   GET    /api/v1/files/{uuid}       metadata (JSON)
 *   GET    /api/v1/files/{uuid}/download   download bytes
 *   HEAD   /api/v1/files/{uuid}       metadata as headers
 *   DELETE /api/v1/files/{uuid}       soft delete
 *   GET    /api/v1/files?prefix=      list by logical-path prefix
 *   POST   /api/v1/files/presign      download URL (presigned / direct)
 */
@RestController
@RequestMapping("/api/v1/files")
@RequiredArgsConstructor
public class FileController {

    private static final Pattern UUID_PATTERN = Pattern.compile("^[0-9a-fA-F]{32}$");

    private final FileObjectService service;

    // ── upload ──────────────────────────────────────────────────────────

    /** Multipart upload; the server generates the uuid. Logical path and
     *  filename are metadata. */
    @PostMapping("/upload")
    public FileReceipt upload(@RequestParam("file") MultipartFile file,
                              @RequestParam(value = "path", required = false, defaultValue = "") String path,
                              @RequestParam(value = "fileName", required = false) String fileName,
                              @RequestParam(value = "fileCategory", required = false) String fileCategory,
                              @RequestParam(value = "usage", required = false) String usage,
                              @RequestParam(value = "meta", required = false) String meta) throws IOException {
        return service.upload(file, path, fileName, fileCategory, usage, meta);
    }

    /** Raw-stream upload at a client-chosen uuid (idempotent re-upload of the
     *  same address). Metadata rides X-File-* headers. */
    @PutMapping("/{uuid}")
    public FileReceipt put(@PathVariable String uuid, HttpServletRequest request) throws IOException {
        requireUuid(uuid);
        FileReceipt existing = service.findAnyByUuid(uuid);
        String path = request.getHeader("X-File-Path");
        String name = request.getHeader("X-File-Name");
        if (existing != null) {
            path = path != null ? path : existing.getPath();
            name = name != null ? name : existing.getFilename();
        }
        if (name == null || name.isBlank()) {
            throw ApiException.badRequest("X-File-Name header is required for a new uuid");
        }
        String[] parts = service.splitLogicalPath(path == null ? "" : path, name);
        return service.store(parts[0], parts[1],
                request.getHeader("X-File-Category"),
                request.getHeader("X-File-Usage"),
                request.getHeader("X-File-Meta"),
                request.getContentType(),
                uuid,
                request.getInputStream());
    }

    // ── object operations ───────────────────────────────────────────────

    /** Metadata (JSON) — GET does not download. */
    @GetMapping("/{uuid}")
    public FileReceipt metadata(@PathVariable String uuid) {
        requireUuid(uuid);
        return service.receiptByUuid(uuid);
    }

    /** Download the object's bytes. */
    @GetMapping("/{uuid}/download")
    public void download(@PathVariable String uuid,
                         @RequestParam(value = "bom", required = false, defaultValue = "false") boolean bom,
                         HttpServletResponse response) throws IOException {
        requireUuid(uuid);
        service.download(uuid, bom, response);
    }

    @RequestMapping(value = "/{uuid}", method = RequestMethod.HEAD)
    public void head(@PathVariable String uuid, HttpServletResponse response) {
        requireUuid(uuid);
        FileReceipt r = service.receiptByUuid(uuid);
        response.setIntHeader("Content-Length", r.getSize() == null ? 0 : r.getSize().intValue());
        response.setHeader("Content-Type", r.getMime() != null ? r.getMime() : "application/octet-stream");
        response.setHeader("ETag", '"' + r.getSha256() + '"');
        response.setHeader("X-File-Md5", r.getMd5());
        response.setHeader("X-File-Version", String.valueOf(r.getVersion()));
        response.setHeader("X-File-Path", r.getFullPath());
        if (r.getFileCategory() != null) response.setHeader("X-File-Category", r.getFileCategory());
        if (r.getUsage() != null) response.setHeader("X-File-Usage", r.getUsage());
        if (r.getMeta() != null) response.setHeader("X-File-Meta", r.getMeta());
    }

    @DeleteMapping("/{uuid}")
    public Map<String, Object> delete(@PathVariable String uuid) {
        requireUuid(uuid);
        return Map.of("deleted", service.delete(uuid));
    }

    // ── find files under a folder ───────────────────────────────────────

    /**
     * Find files under a folder. `path` picks the folder (default: root);
     * `q` filters by name substring; `recursive=true` includes descendants;
     * the rest are attribute/time filters. Newest first, page-number
     * paginated (`page` is 1-based, `size` rows per page; `total` reports the
     * full match count so a UI can render page links).
     */
    @GetMapping
    public PathListing list(@RequestParam(value = "path", required = false, defaultValue = "") String path,
                            @RequestParam(value = "q", required = false) String q,
                            @RequestParam(value = "recursive", required = false, defaultValue = "false") boolean recursive,
                            @RequestParam(value = "fileCategory", required = false) String fileCategory,
                            @RequestParam(value = "usage", required = false) String usage,
                            @RequestParam(value = "from", required = false) String from,
                            @RequestParam(value = "to", required = false) String to,
                            @RequestParam(value = "page", required = false) Integer page,
                            @RequestParam(value = "size", required = false) Integer size) {
        return service.query(path, recursive, q, fileCategory, usage, from, to, page, size);
    }

    private void requireUuid(String uuid) {
        if (uuid == null || !UUID_PATTERN.matcher(uuid).matches()) {
            throw ApiException.badRequest("uuid must be 32 hex characters");
        }
    }
}
