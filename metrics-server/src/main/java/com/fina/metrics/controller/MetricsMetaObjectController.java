package com.fina.metrics.controller;

import com.fina.metrics.dto.ApiResponse;
import com.fina.metrics.dto.MetricsMetaObjectRequest;
import com.fina.metrics.dto.MetricsMetaObjectVO;
import com.fina.metrics.dto.PageResult;
import com.fina.metrics.service.DataSourceApiKeyAuthService;
import com.fina.metrics.service.DataSourceApiKeyPermission;
import com.fina.metrics.service.MetricsAdminAuthService;
import com.fina.metrics.service.MetricsMetaObjectService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/meta/objects")
public class MetricsMetaObjectController {

    private final MetricsMetaObjectService service;
    private final MetricsAdminAuthService adminAuth;
    private final DataSourceApiKeyAuthService datasourceAuth;

    @GetMapping
    public ApiResponse<PageResult<MetricsMetaObjectVO>> list(
            @RequestParam(required = false) Long datasourceId,
            @RequestParam(required = false) String objectType,
            @RequestParam(required = false) String objectKey,
            @RequestParam(required = false) Integer page,
            @RequestParam(required = false) Integer pageSize,
            HttpServletRequest httpRequest) {
        requireRead(httpRequest, datasourceId);
        return ApiResponse.ok(service.list(datasourceId, objectType, objectKey, page, pageSize));
    }

    @GetMapping("/{id}")
    public ApiResponse<MetricsMetaObjectVO> get(@PathVariable Long id, HttpServletRequest httpRequest) {
        MetricsMetaObjectVO object = service.getById(id);
        requireRead(httpRequest, object.getDatasourceId());
        return ApiResponse.ok(object);
    }

    @PostMapping
    public ApiResponse<MetricsMetaObjectVO> create(
            @Valid @RequestBody MetricsMetaObjectRequest request,
            HttpServletRequest httpRequest) {
        requireWrite(httpRequest, request.getDatasourceId());
        return ApiResponse.ok(service.create(request));
    }

    @PutMapping("/{id}")
    public ApiResponse<MetricsMetaObjectVO> update(
            @PathVariable Long id,
            @Valid @RequestBody MetricsMetaObjectRequest request,
            HttpServletRequest httpRequest) {
        MetricsMetaObjectVO existing = service.getById(id);
        Long datasourceId = request.getDatasourceId() != null ? request.getDatasourceId() : existing.getDatasourceId();
        if (existing.getDatasourceId() != null && request.getDatasourceId() != null
                && !existing.getDatasourceId().equals(request.getDatasourceId())) {
            throw new IllegalArgumentException("meta object datasourceId cannot be changed");
        }
        requireWrite(httpRequest, datasourceId);
        request.setDatasourceId(datasourceId);
        return ApiResponse.ok(service.update(id, request));
    }

    @DeleteMapping("/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id, HttpServletRequest httpRequest) {
        MetricsMetaObjectVO existing = service.getById(id);
        requireWrite(httpRequest, existing.getDatasourceId());
        service.delete(id);
        return ApiResponse.ok();
    }

    private void requireRead(HttpServletRequest request, Long datasourceId) {
        if (datasourceId == null) {
            adminAuth.requireAdmin(request);
        } else {
            datasourceAuth.requirePermission(request, datasourceId, DataSourceApiKeyPermission.META_READ);
        }
    }

    private void requireWrite(HttpServletRequest request, Long datasourceId) {
        if (datasourceId == null) {
            adminAuth.requireAdmin(request);
        } else {
            datasourceAuth.requirePermission(request, datasourceId, DataSourceApiKeyPermission.META_WRITE);
        }
    }
}
