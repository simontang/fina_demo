package com.fina.file.controller;

import com.fina.file.dto.FileReceipt;
import com.fina.file.dto.PathListing;
import com.fina.file.entity.FileObject;
import com.fina.file.exception.ApiException;
import com.fina.file.mapper.FileObjectMapper;
import com.fina.file.service.FileObjectService;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;

/**
 * Path-first file API. `path` always means the full logical path
 * ({directory}/{filename}); id/uuid endpoints are compatibility aliases,
 * mirroring the fina-ai file service where download(id) is deprecated in
 * favour of path addressing.
 */
@RestController
@RequestMapping("/api/v1/files")
@RequiredArgsConstructor
public class FileController {

    private final FileObjectService service;
    private final FileObjectMapper mapper;

    @PostMapping("/upload")
    public FileReceipt upload(@RequestParam("file") MultipartFile file,
                              @RequestParam(value = "path", required = false, defaultValue = "") String path,
                              @RequestParam(value = "fileName", required = false) String fileName,
                              @RequestParam(value = "fileCategory", required = false) String fileCategory,
                              @RequestParam(value = "usage", required = false) String usage,
                              @RequestParam(value = "meta", required = false) String meta) {
        return service.upload(file, path, fileName, fileCategory, usage, meta);
    }

    @GetMapping("/download")
    public void download(@RequestParam("path") String path,
                         @RequestParam(value = "version", required = false) Integer version,
                         @RequestParam(value = "bom", required = false, defaultValue = "false") boolean bom,
                         HttpServletResponse response) throws IOException {
        service.download(path, version, bom, response);
    }

    @GetMapping
    public PathListing list(@RequestParam(value = "prefix", required = false, defaultValue = "") String prefix) {
        return service.list(prefix);
    }

    @GetMapping("/receipt")
    public FileReceipt receiptByPath(@RequestParam("path") String path,
                                     @RequestParam(value = "version", required = false) Integer version) {
        return service.receiptByPath(path, version);
    }

    @GetMapping("/{id}/receipt")
    public FileReceipt receiptById(@PathVariable long id) {
        return service.receiptById(id);
    }

    @GetMapping("/{id}/download")
    public void downloadById(@PathVariable long id,
                             @RequestParam(value = "bom", required = false, defaultValue = "false") boolean bom,
                             HttpServletResponse response) throws IOException {
        FileObject row = mapper.selectById(id);
        if (row == null || !"active".equals(row.getStatus())) {
            throw ApiException.notFound("no active file matches the given address");
        }
        service.download(row.fullPath(), row.getVersion(), bom, response);
    }

    @GetMapping("/uuid/{uuid}/download")
    public void downloadByUuid(@PathVariable String uuid,
                               @RequestParam(value = "bom", required = false, defaultValue = "false") boolean bom,
                               HttpServletResponse response) throws IOException {
        FileReceipt receipt = service.receiptByUuid(uuid);
        service.download(receipt.getFullPath(), receipt.getVersion(), bom, response);
    }

    @DeleteMapping
    public Map<String, Object> delete(@RequestParam("path") String path,
                                      @RequestParam(value = "version", required = false) Integer version) {
        return Map.of("deleted", service.delete(path, version));
    }
}
