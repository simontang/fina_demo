package com.fina.metrics.controller;

import com.fasterxml.jackson.databind.JsonNode;
import com.fina.metrics.dto.*;
import com.fina.metrics.exception.ForbiddenException;
import com.fina.metrics.service.DataSourceTableAccessService;
import com.fina.metrics.service.MetricsMetaObjectService;
import com.fina.metrics.service.MetricsMetaObjectTypes;
import com.fina.metrics.util.ReadOnlySqlValidator;
import com.fina.metrics.util.SqlIdentifierUtils;
import com.fina.metrics.util.SqlIdentifierUtils.TableIdentifier;
import com.fina.metrics.util.SqlTableReferenceExtractor;
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
        validateTableMeta(dsId, request, objectKey);
        MetricsMetaObjectVO metaObject = metaObjectService.create(toMetaObjectRequest(dsId, objectType, objectKey, request));
        return ApiResponse.ok(DataSourcePublishedMetaVO.builder()
                .metaObject(metaObject)
                .build());
    }

    @PutMapping("/tables/{tableKey}")
    public ApiResponse<DataSourcePublishedMetaVO> updateTableMeta(
            @PathVariable Long dsId,
            @PathVariable String tableKey,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId,
            @Valid @RequestBody DataSourcePublishedMetaRequest request) {
        String objectType = resolveObjectType(request.getObjectType(), MetricsMetaObjectTypes.TABLE_VIEW_DETAIL, TABLE_META_TYPES);
        validateTableMeta(dsId, request, tableKey);
        MetricsMetaObjectVO metaObject = metaObjectService.updateByDatasourceTypeKey(
                dsId, objectType, tableKey, toMetaObjectRequest(dsId, objectType, tableKey, request));
        return ApiResponse.ok(DataSourcePublishedMetaVO.builder()
                .metaObject(metaObject)
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
        for (MetricsMetaObjectVO object : objects) {
            metaObjectService.delete(object.getId());
        }
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
        validateLegacyAccessGrant(dsId, request.getAccessGrant());
        return ApiResponse.ok(metaObjectService.create(toMetaObjectRequest(dsId, objectType, objectKey, request)));
    }

    @PutMapping("/metrics/{metricKey}")
    public ApiResponse<MetricsMetaObjectVO> updateMetricMeta(
            @PathVariable Long dsId,
            @PathVariable String metricKey,
            @Valid @RequestBody DataSourcePublishedMetaRequest request) {
        String objectType = resolveObjectType(request.getObjectType(), MetricsMetaObjectTypes.METRIC_DETAIL, METRIC_META_TYPES);
        validateLegacyAccessGrant(dsId, request.getAccessGrant());
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

    private void validateTableMeta(
            Long datasourceId,
            DataSourcePublishedMetaRequest request,
            String objectKey) {
        JsonNode payload = request.getPayload();
        String schema = textField(payload, "schemaName");
        for (String field : List.of("tableName", "viewName")) {
            String target = textField(payload, field);
            if (StringUtils.hasText(target)) SqlIdentifierUtils.parseTableIdentifier(schema, target);
        }
        Set<TableIdentifier> tables = new LinkedHashSet<>();
        addTableReference(tables, schema, textField(payload, "tableName"));
        addTableReference(tables, schema, textField(payload, "viewName"));
        if (tables.isEmpty()) {
            SqlIdentifierUtils.parseTableIdentifier(schema, objectKey);
            addTableReference(tables, schema, objectKey);
        }
        addTableReference(tables, schema, textField(payload, "mainTable"));
        addTableReference(tables, schema, textField(payload, "lineTable"));
        String selectSql = textField(payload, "selectSql");
        if (StringUtils.hasText(selectSql)) {
            ReadOnlySqlValidator.validate(selectSql);
            for (SqlTableReferenceExtractor.TableReference reference : SqlTableReferenceExtractor.extract(selectSql)) {
                // Unqualified SQL resolves through the datasource, not the metadata payload schema.
                tables.add(new TableIdentifier(reference.schemaName(), reference.tableName()));
            }
        }
        for (TableIdentifier table : tables) {
            if (!tableAccessService.isTableAuthorized(null, datasourceId, table.schemaName(), table.tableName())) {
                throw new ForbiddenException("Table meta references a table outside datasource visible scope: "
                        + (table.schemaName() == null ? "" : table.schemaName() + ".") + table.tableName());
            }
        }
        validateLegacyAccessGrant(datasourceId, request.getAccessGrant());
    }

    private void addTableReference(Set<TableIdentifier> tables, String schema, String tableName) {
        if (!StringUtils.hasText(tableName)) {
            return;
        }
        TableIdentifier table = SqlIdentifierUtils.parseTableIdentifier(null, tableName);
        tables.add(new TableIdentifier(firstNonBlank(table.schemaName(), schema), table.tableName()));
    }

    /**
     * Legacy accessGrant is an assertion of an identical active rule, never a scope edit.
     * No prefix containment or implicit rule is inferred, even in ALL mode. Callers
     * without an identical rule must omit accessGrant; concrete tables are checked above.
     */
    private void validateLegacyAccessGrant(Long datasourceId, DataSourceTableGrantRequest request) {
        if (request == null) {
            return;
        }
        boolean covered = tableAccessService.listActiveGrants(null, datasourceId).stream()
                .anyMatch(existing -> sameGrant(existing, request));
        if (!covered) {
            throw new ForbiddenException("accessGrant must match an existing active datasource scope rule");
        }
    }

    private boolean sameGrant(DataSourceTableGrantVO existing, DataSourceTableGrantRequest request) {
        return Objects.equals(existing.getStatus(), 1)
                && Objects.equals(request.getStatus() == null ? 1 : request.getStatus(), 1)
                && Objects.equals(normalizeBlank(existing.getSchemaName()), normalizeBlank(request.getSchemaName()))
                && Objects.equals(existing.getTablePattern(), normalizeBlank(request.getTablePattern()))
                && firstNonBlank(request.getPatternType(), "PREFIX").equalsIgnoreCase(existing.getPatternType())
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

    private String firstNonBlank(String... values) {
        for (String value : values) {
            if (StringUtils.hasText(value)) {
                return value.trim();
            }
        }
        return null;
    }

    private String normalizeBlank(String value) {
        return StringUtils.hasText(value) ? value.trim() : null;
    }
}
