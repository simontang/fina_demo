package com.fina.metrics.controller;

import com.fasterxml.jackson.databind.JsonNode;
import com.fina.metrics.dto.*;
import com.fina.metrics.service.DataSourceTableAccessService;
import com.fina.metrics.service.MetricsMetaObjectService;
import com.fina.metrics.service.MetricsMetaObjectTypes;
import com.fina.metrics.util.TenantHeaderResolver;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.*;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/datasources/{dsId}/meta")
public class DataSourceMetaController {

    private static final List<String> TABLE_META_TYPES = List.of(
            MetricsMetaObjectTypes.TABLE_CATALOG,
            MetricsMetaObjectTypes.TABLE_VIEW_DETAIL);
    private static final List<String> METRIC_META_TYPES = List.of(
            MetricsMetaObjectTypes.METRIC_INDEX,
            MetricsMetaObjectTypes.METRIC_DETAIL);

    private final MetricsMetaObjectService metaObjectService;
    private final DataSourceTableAccessService tableAccessService;

    @GetMapping("/tables")
    public ApiResponse<PageResult<MetricsMetaObjectVO>> listTableMeta(
            @PathVariable Long dsId,
            @RequestParam(required = false) String objectType,
            @RequestParam(required = false) String objectKey,
            @RequestParam(required = false) Integer page,
            @RequestParam(required = false) Integer pageSize) {
        return ApiResponse.ok(metaObjectService.listByDatasourceAndTypes(
                dsId, resolveTypes(objectType, TABLE_META_TYPES), objectKey, page, pageSize));
    }

    @GetMapping("/tables/{tableKey}")
    public ApiResponse<List<MetricsMetaObjectVO>> getTableMeta(
            @PathVariable Long dsId,
            @PathVariable String tableKey,
            @RequestParam(required = false) String objectType) {
        return ApiResponse.ok(metaObjectService.listByDatasourceAndTypes(
                dsId, resolveTypes(objectType, TABLE_META_TYPES), tableKey, 1, TABLE_META_TYPES.size())
                .getItems());
    }

    @PostMapping("/tables")
    public ApiResponse<DataSourcePublishedMetaVO> createTableMeta(
            @PathVariable Long dsId,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId,
            @Valid @RequestBody DataSourcePublishedMetaRequest request) {
        String objectType = resolveObjectType(request.getObjectType(), MetricsMetaObjectTypes.TABLE_VIEW_DETAIL, TABLE_META_TYPES);
        String objectKey = resolveObjectKey(request, "tableName", "viewName");
        MetricsMetaObjectVO metaObject = metaObjectService.create(toMetaObjectRequest(dsId, objectType, objectKey, request));
        DataSourceTableGrantVO grant = ensureTableGrant(resolveTenant(tenantId), dsId, request, objectKey);
        return ApiResponse.ok(DataSourcePublishedMetaVO.builder()
                .metaObject(metaObject)
                .tableGrant(grant)
                .build());
    }

    @PutMapping("/tables/{tableKey}")
    public ApiResponse<DataSourcePublishedMetaVO> updateTableMeta(
            @PathVariable Long dsId,
            @PathVariable String tableKey,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId,
            @Valid @RequestBody DataSourcePublishedMetaRequest request) {
        String objectType = resolveObjectType(request.getObjectType(), MetricsMetaObjectTypes.TABLE_VIEW_DETAIL, TABLE_META_TYPES);
        MetricsMetaObjectVO metaObject = metaObjectService.updateByDatasourceTypeKey(
                dsId, objectType, tableKey, toMetaObjectRequest(dsId, objectType, tableKey, request));
        DataSourceTableGrantVO grant = ensureTableGrant(resolveTenant(tenantId), dsId, request, tableKey);
        return ApiResponse.ok(DataSourcePublishedMetaVO.builder()
                .metaObject(metaObject)
                .tableGrant(grant)
                .build());
    }

    @DeleteMapping("/tables/{tableKey}")
    public ApiResponse<Void> deleteTableMeta(
            @PathVariable Long dsId,
            @PathVariable String tableKey,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId,
            @RequestParam(required = false) String objectType) {
        List<MetricsMetaObjectVO> objects = metaObjectService.listByDatasourceAndTypes(
                dsId, resolveTypes(objectType, TABLE_META_TYPES), tableKey, 1, TABLE_META_TYPES.size())
                .getItems();
        if (objects.isEmpty()) {
            throw new IllegalArgumentException("Datasource table meta not found: " + tableKey);
        }
        Set<String> tablePatterns = new LinkedHashSet<>();
        tablePatterns.add(tableKey);
        for (MetricsMetaObjectVO object : objects) {
            addTextField(tablePatterns, object.getPayload(), "tableName");
            addTextField(tablePatterns, object.getPayload(), "viewName");
            metaObjectService.delete(object.getId());
        }
        deleteMatchingTableGrants(resolveTenant(tenantId), dsId, tablePatterns);
        return ApiResponse.ok();
    }

    @GetMapping("/metrics")
    public ApiResponse<PageResult<MetricsMetaObjectVO>> listMetricMeta(
            @PathVariable Long dsId,
            @RequestParam(required = false) String objectType,
            @RequestParam(required = false) String objectKey,
            @RequestParam(required = false) Integer page,
            @RequestParam(required = false) Integer pageSize) {
        return ApiResponse.ok(metaObjectService.listByDatasourceAndTypes(
                dsId, resolveTypes(objectType, METRIC_META_TYPES), objectKey, page, pageSize));
    }

    @GetMapping("/metrics/{metricKey}")
    public ApiResponse<List<MetricsMetaObjectVO>> getMetricMeta(
            @PathVariable Long dsId,
            @PathVariable String metricKey,
            @RequestParam(required = false) String objectType) {
        return ApiResponse.ok(metaObjectService.listByDatasourceAndTypes(
                dsId, resolveTypes(objectType, METRIC_META_TYPES), metricKey, 1, METRIC_META_TYPES.size())
                .getItems());
    }

    @PostMapping("/metrics")
    public ApiResponse<MetricsMetaObjectVO> createMetricMeta(
            @PathVariable Long dsId,
            @Valid @RequestBody DataSourcePublishedMetaRequest request) {
        String objectType = resolveObjectType(request.getObjectType(), MetricsMetaObjectTypes.METRIC_DETAIL, METRIC_META_TYPES);
        String objectKey = resolveObjectKey(request, "metric_name", "metricName");
        return ApiResponse.ok(metaObjectService.create(toMetaObjectRequest(dsId, objectType, objectKey, request)));
    }

    @PutMapping("/metrics/{metricKey}")
    public ApiResponse<MetricsMetaObjectVO> updateMetricMeta(
            @PathVariable Long dsId,
            @PathVariable String metricKey,
            @Valid @RequestBody DataSourcePublishedMetaRequest request) {
        String objectType = resolveObjectType(request.getObjectType(), MetricsMetaObjectTypes.METRIC_DETAIL, METRIC_META_TYPES);
        return ApiResponse.ok(metaObjectService.updateByDatasourceTypeKey(
                dsId, objectType, metricKey, toMetaObjectRequest(dsId, objectType, metricKey, request)));
    }

    @DeleteMapping("/metrics/{metricKey}")
    public ApiResponse<Void> deleteMetricMeta(
            @PathVariable Long dsId,
            @PathVariable String metricKey,
            @RequestParam(required = false) String objectType) {
        List<MetricsMetaObjectVO> objects = metaObjectService.listByDatasourceAndTypes(
                dsId, resolveTypes(objectType, METRIC_META_TYPES), metricKey, 1, METRIC_META_TYPES.size())
                .getItems();
        if (objects.isEmpty()) {
            throw new IllegalArgumentException("Datasource metric meta not found: " + metricKey);
        }
        for (MetricsMetaObjectVO object : objects) {
            metaObjectService.delete(object.getId());
        }
        return ApiResponse.ok();
    }

    private MetricsMetaObjectRequest toMetaObjectRequest(
            Long datasourceId,
            String objectType,
            String objectKey,
            DataSourcePublishedMetaRequest request) {
        MetricsMetaObjectRequest metaRequest = new MetricsMetaObjectRequest();
        metaRequest.setDatasourceId(datasourceId);
        metaRequest.setObjectType(objectType);
        metaRequest.setObjectKey(objectKey);
        metaRequest.setPayload(request.getPayload());
        metaRequest.setStatus(request.getStatus() != null ? request.getStatus() : 1);
        return metaRequest;
    }

    private DataSourceTableGrantVO ensureTableGrant(
            String tenantId,
            Long datasourceId,
            DataSourcePublishedMetaRequest request,
            String objectKey) {
        DataSourceTableGrantRequest grantRequest = request.getAccessGrant() != null
                ? request.getAccessGrant()
                : defaultGrantRequest(request, objectKey);
        normalizeGrantRequest(grantRequest);
        List<DataSourceTableGrantVO> grants = tableAccessService.listGrants(tenantId, datasourceId);
        for (DataSourceTableGrantVO grant : grants) {
            if (sameGrant(grant, grantRequest)) {
                if (!Objects.equals(grant.getStatus(), grantRequest.getStatus())) {
                    return tableAccessService.updateGrant(tenantId, datasourceId, grant.getId(), grantRequest);
                }
                return grant;
            }
        }
        return tableAccessService.createGrant(tenantId, datasourceId, grantRequest);
    }

    private DataSourceTableGrantRequest defaultGrantRequest(
            DataSourcePublishedMetaRequest request,
            String objectKey) {
        DataSourceTableGrantRequest grant = new DataSourceTableGrantRequest();
        grant.setSchemaName(textField(request.getPayload(), "schemaName"));
        grant.setTablePattern(firstNonBlank(
                textField(request.getPayload(), "tableName"),
                textField(request.getPayload(), "viewName"),
                objectKey));
        grant.setPatternType("EXACT");
        grant.setCaseSensitive(false);
        grant.setStatus(request.getStatus() != null ? request.getStatus() : 1);
        return grant;
    }

    private void normalizeGrantRequest(DataSourceTableGrantRequest request) {
        if (!StringUtils.hasText(request.getPatternType())) {
            request.setPatternType("PREFIX");
        }
        if (request.getCaseSensitive() == null) {
            request.setCaseSensitive(false);
        }
        if (request.getStatus() == null) {
            request.setStatus(1);
        }
    }

    private void deleteMatchingTableGrants(String tenantId, Long datasourceId, Set<String> tablePatterns) {
        for (DataSourceTableGrantVO grant : tableAccessService.listGrants(tenantId, datasourceId)) {
            if (tablePatterns.contains(grant.getTablePattern())) {
                tableAccessService.deleteGrant(tenantId, datasourceId, grant.getId());
            }
        }
    }

    private boolean sameGrant(DataSourceTableGrantVO existing, DataSourceTableGrantRequest request) {
        return Objects.equals(normalizeBlank(existing.getSchemaName()), normalizeBlank(request.getSchemaName()))
                && existing.getTablePattern().equals(request.getTablePattern())
                && existing.getPatternType().equalsIgnoreCase(request.getPatternType())
                && Objects.equals(Boolean.TRUE.equals(existing.getCaseSensitive()), Boolean.TRUE.equals(request.getCaseSensitive()));
    }

    private List<String> resolveTypes(String objectType, List<String> defaults) {
        if (!StringUtils.hasText(objectType)) {
            return defaults;
        }
        return List.of(resolveObjectType(objectType, null, defaults));
    }

    private String resolveObjectType(String objectType, String defaultType, List<String> allowed) {
        String resolved = StringUtils.hasText(objectType) ? objectType.trim() : defaultType;
        if (!allowed.contains(resolved)) {
            throw new IllegalArgumentException("Unsupported datasource meta object type: " + resolved);
        }
        return resolved;
    }

    private String resolveObjectKey(DataSourcePublishedMetaRequest request, String... payloadFields) {
        String key = firstNonBlank(request.getObjectKey(), textField(request.getPayload(), payloadFields));
        if (!StringUtils.hasText(key)) {
            throw new IllegalArgumentException("objectKey is required or must be derivable from payload");
        }
        return key;
    }

    private String textField(JsonNode payload, String... fieldNames) {
        if (payload == null || !payload.isObject()) {
            return null;
        }
        for (String fieldName : fieldNames) {
            String value = payload.path(fieldName).asText(null);
            if (StringUtils.hasText(value)) {
                return value.trim();
            }
        }
        return null;
    }

    private void addTextField(Set<String> values, JsonNode payload, String fieldName) {
        String value = textField(payload, fieldName);
        if (StringUtils.hasText(value)) {
            values.add(value);
        }
    }

    private String firstNonBlank(String... values) {
        for (String value : values) {
            if (StringUtils.hasText(value)) {
                return value.trim();
            }
        }
        return null;
    }

    private String normalizeBlank(String value) {
        return StringUtils.hasText(value) ? value : null;
    }

    private String resolveTenant(String tenantId) {
        return TenantHeaderResolver.resolve(tenantId);
    }
}
