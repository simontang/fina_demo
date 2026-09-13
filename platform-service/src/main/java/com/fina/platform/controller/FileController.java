package com.fina.platform.controller;

import com.fina.platform.dto.FileReceipt;
import com.fina.platform.dto.PathListing;
import com.fina.platform.exception.ApiException;
import com.fina.platform.service.FileObjectService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.util.AntPathMatcher;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.HandlerMapping;

import java.io.IOException;
import java.util.Map;

/**
 * File API — S3-style resource addressing merged with fina-ai's practical
 * interactions. The object key is the URL path: PUT/GET/HEAD/DELETE operate
 * on /api/v1/files/{key...}, metadata travels as standard headers
 * (ETag = sha256, X-File-*), listing follows list-objects semantics
 * (prefix + delimiter), and the uuid handle + POST receipt keep the
 * fina-ai heritage.
 */
@RestController
@RequestMapping("/api/v1/files")
@RequiredArgsConstructor
public class FileController {

    private final FileObjectService service;

    // ── upload ──────────────────────────────────────────────────────────

    /** Multipart upload (fina-ai style): form fields carry path and metadata. */
    @PostMapping("/upload")
    public FileReceipt upload(@RequestParam("file") MultipartFile file,
                              @RequestParam(value = "path", required = false, defaultValue = "") String path,
                              @RequestParam(value = "fileName", required = false) String fileName,
                              @RequestParam(value = "fileCategory", required = false) String fileCategory,
                              @RequestParam(value = "usage", required = false) String usage,
                              @RequestParam(value = "meta", required = false) String meta) throws IOException {
        String effectiveName = (fileName != null && !fileName.isBlank())
                ? fileName : file.getOriginalFilename();
        String[] parts = service.splitLogicalPath(path, effectiveName);
        return service.store(parts[0], parts[1], fileCategory, usage, meta,
                file.getContentType(), file.getSize(), file.getInputStream());
    }

    /** Raw-stream upload (S3 PutObject style): the key is the URL path,
     *  metadata rides X-File-* headers. */
    @PutMapping("/**")
    public FileReceipt put(HttpServletRequest request) throws IOException {
        String key = objectKey(request);
        String[] parts = service.splitLogicalPath(key, null);
        return service.store(parts[0], parts[1],
                request.getHeader("X-File-Category"),
                request.getHeader("X-File-Usage"),
                request.getHeader("X-File-Meta"),
                request.getContentType(),
                request.getContentLengthLong(),
                request.getInputStream());
    }

    // ── download / metadata / delete ────────────────────────────────────

    @GetMapping("/**")
    public void download(HttpServletRequest request, HttpServletResponse response,
                         @RequestParam(value = "version", required = false) Integer version,
                         @RequestParam(value = "bom", required = false, defaultValue = "false") boolean bom)
            throws IOException {
        String key = objectKey(request);
        service.download(key, version, bom, response);
    }

    /** S3 HeadObject style: metadata as response headers, empty body. */
    @RequestMapping(value = "/**", method = RequestMethod.HEAD)
    public void head(HttpServletRequest request, HttpServletResponse response,
                     @RequestParam(value = "version", required = false) Integer version) {
        FileReceipt r = service.receiptByPath(objectKey(request), version);
        response.setIntHeader("Content-Length", r.getSize().intValue());
        response.setHeader("Content-Type", r.getMime() != null ? r.getMime() : "application/octet-stream");
        response.setHeader("ETag", '"' + r.getSha256() + '"');
        response.setHeader("X-File-Md5", r.getMd5());
        response.setHeader("X-File-Uuid", r.getUuid());
        response.setHeader("X-File-Version", String.valueOf(r.getVersion()));
        if (r.getFileCategory() != null) response.setHeader("X-File-Category", r.getFileCategory());
        if (r.getUsage() != null) response.setHeader("X-File-Usage", r.getUsage());
        if (r.getMeta() != null) response.setHeader("X-File-Meta", r.getMeta());
    }

    @DeleteMapping("/**")
    public Map<String, Object> delete(HttpServletRequest request,
                                      @RequestParam(value = "version", required = false) Integer version) {
        String key = objectKey(request);
        return Map.of("deleted", service.delete(key, version));
    }

    // ── listing ─────────────────────────────────────────────────────────

    /** List-objects style: prefix + delimiter (delimiter accepted for S3
     *  familiarity; pseudo-directory aggregation is always on). */
    @GetMapping
    public PathListing list(@RequestParam(value = "prefix", required = false, defaultValue = "") String prefix,
                            @RequestParam(value = "delimiter", required = false) String delimiter) {
        return service.list(prefix);
    }

    // ── receipts ────────────────────────────────────────────────────────

    /** JSON receipt by path (fina-ai download2ByPath interaction style:
     *  POST + JSON body, keeps the wildcard route unambiguous). */
    @PostMapping("/receipt")
    public FileReceipt receiptByPath(@RequestBody ReceiptRequest req) {
        if (req.path() == null || req.path().isBlank()) {
            throw ApiException.badRequest("path is required");
        }
        return service.receiptByPath(req.path(), req.version());
    }

    public record ReceiptRequest(String path, Integer version) {
    }

    @GetMapping("/uuid/{uuid}/receipt")
    public FileReceipt receiptByUuid(@PathVariable String uuid) {
        return service.receiptByUuid(uuid);
    }

    /** Download by uuid handle (fina-ai downloadByUuid heritage). */
    @GetMapping("/uuid/{uuid}")
    public void downloadByUuid(@PathVariable String uuid,
                               @RequestParam(value = "bom", required = false, defaultValue = "false") boolean bom,
                               HttpServletResponse response) throws IOException {
        FileReceipt receipt = service.receiptByUuid(uuid);
        service.download(receipt.getFullPath(), receipt.getVersion(), bom, response);
    }

    /** Extract the wildcard part of /api/v1/files/** as the object key.
     *  The matcher hands back the still-encoded path — decode it (UriUtils
     *  keeps '+' intact, unlike URLDecoder). */
    private String objectKey(HttpServletRequest request) {
        String pattern = (String) request.getAttribute(HandlerMapping.BEST_MATCHING_PATTERN_ATTRIBUTE);
        String path = (String) request.getAttribute(HandlerMapping.PATH_WITHIN_HANDLER_MAPPING_ATTRIBUTE);
        String key = new AntPathMatcher().extractPathWithinPattern(pattern, path);
        if (key.startsWith("/")) {
            key = key.substring(1);
        }
        return org.springframework.web.util.UriUtils.decode(key, java.nio.charset.StandardCharsets.UTF_8);
    }
}
