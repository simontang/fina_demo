package com.fina.metrics.controller;

import com.fina.metrics.dto.ApiResponse;
import com.fina.metrics.dto.MetricsMetaObjectRequest;
import com.fina.metrics.dto.MetricsMetaObjectVO;
import com.fina.metrics.dto.PageResult;
import com.fina.metrics.service.MetricsMetaObjectService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/meta/objects")
public class MetricsMetaObjectController {

    private final MetricsMetaObjectService service;

    @GetMapping
    public ApiResponse<PageResult<MetricsMetaObjectVO>> list(
            @RequestParam(required = false) Long datasourceId,
            @RequestParam(required = false) String objectType,
            @RequestParam(required = false) String objectKey,
            @RequestParam(required = false) Integer page,
            @RequestParam(required = false) Integer pageSize) {
        return ApiResponse.ok(service.list(datasourceId, objectType, objectKey, page, pageSize));
    }

    @GetMapping("/{id}")
    public ApiResponse<MetricsMetaObjectVO> get(@PathVariable Long id) {
        return ApiResponse.ok(service.getById(id));
    }

    @PostMapping
    public ApiResponse<MetricsMetaObjectVO> create(
            @Valid @RequestBody MetricsMetaObjectRequest request) {
        return ApiResponse.ok(service.create(request));
    }

    @PutMapping("/{id}")
    public ApiResponse<MetricsMetaObjectVO> update(
            @PathVariable Long id,
            @Valid @RequestBody MetricsMetaObjectRequest request) {
        return ApiResponse.ok(service.update(id, request));
    }

    @DeleteMapping("/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        service.delete(id);
        return ApiResponse.ok();
    }
}
