package com.fina.metrics.controller;

import com.fina.metrics.dto.*;
import com.fina.metrics.service.DataSourceService;
import com.fina.metrics.service.DataSourceTableAccessService;
import com.fina.metrics.util.TenantHeaderResolver;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * SAP B1 HANA Datasource Management API
 *
 * ┌──────────────────────────────────────────────────────────────────────────┐
 * │ Method  │ Path                               │ Description               │
 * ├──────────────────────────────────────────────────────────────────────────┤
 * │ GET     │ /api/v1/datasources                │ List all (incl. inactive) │
 * │ GET     │ /api/v1/datasources/active         │ List active only          │
 * │ GET     │ /api/v1/datasources/{id}           │ Get one by id             │
 * │ POST    │ /api/v1/datasources                │ Create new datasource     │
 * │ PUT     │ /api/v1/datasources/{id}           │ Update (password optional)│
 * │ DELETE  │ /api/v1/datasources/{id}           │ Soft-delete               │
 * ├──────────────────────────────────────────────────────────────────────────┤
 * │ PATCH   │ /api/v1/datasources/{id}/status    │ Enable / disable          │
 * │ POST    │ /api/v1/datasources/{id}/enable    │ Enable shortcut           │
 * │ POST    │ /api/v1/datasources/{id}/disable   │ Disable shortcut          │
 * ├──────────────────────────────────────────────────────────────────────────┤
 * │ POST    │ /api/v1/datasources/test           │ Test with given creds     │
 * │ POST    │ /api/v1/datasources/{id}/test      │ Test a saved datasource   │
 * ├──────────────────────────────────────────────────────────────────────────┤
 * │ POST    │ /api/v1/datasources/{id}/reload    │ Reload pool from DB       │
 * │ GET     │ /api/v1/datasources/{id}/pool      │ Pool metrics (HikariCP)   │
 * └──────────────────────────────────────────────────────────────────────────┘
 */
@Slf4j
@RestController
@RequestMapping("/api/v1/datasources")
@RequiredArgsConstructor
public class DataSourceController {

    private final DataSourceService dataSourceService;
    private final DataSourceTableAccessService tableAccessService;

    // ─── Query ────────────────────────────────────────────────────────────────

    @GetMapping
    public ApiResponse<List<DataSourceVO>> listAll() {
        return ApiResponse.ok(dataSourceService.listAll());
    }

    @GetMapping("/active")
    public ApiResponse<List<DataSourceVO>> listActive() {
        return ApiResponse.ok(dataSourceService.listActive());
    }

    @GetMapping("/{id}")
    public ApiResponse<DataSourceVO> getById(@PathVariable Long id) {
        return ApiResponse.ok(dataSourceService.getById(id));
    }

    // ─── Create / Update / Delete ─────────────────────────────────────────────

    /**
     * Create a new SAP B1 HANA datasource.
     * All fields are required. Password is plain-text and will be AES-encrypted at rest.
     */
    @PostMapping
    public ApiResponse<DataSourceVO> create(@Valid @RequestBody DataSourceRequest request) {
        log.debug("Create datasource name={} url={}", request.getName(), request.getUrl());
        return ApiResponse.ok(dataSourceService.create(request));
    }

    /**
     * Update a datasource.
     * Password is optional — omit or leave blank to keep the current password.
     */
    @PutMapping("/{id}")
    public ApiResponse<DataSourceVO> update(
            @PathVariable Long id,
            @Valid @RequestBody DataSourceUpdateRequest request) {
        log.debug("Update datasource id={} name={}", id, request.getName());
        return ApiResponse.ok(dataSourceService.update(id, request));
    }

    /**
     * Soft-delete a datasource. The connection pool is closed immediately.
     */
    @DeleteMapping("/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        log.debug("Delete datasource id={}", id);
        dataSourceService.delete(id);
        return ApiResponse.ok();
    }

    // ─── Status toggle ────────────────────────────────────────────────────────

    /**
     * Set datasource status (1=active, 0=inactive).
     * Activating registers the pool; deactivating closes it.
     */
    @PatchMapping("/{id}/status")
    public ApiResponse<DataSourceVO> setStatus(
            @PathVariable Long id,
            @Valid @RequestBody DataSourceStatusRequest request) {
        return ApiResponse.ok(dataSourceService.setStatus(id, request.getStatus()));
    }

    /** Enable shortcut — equivalent to PATCH /{id}/status with status=1 */
    @PostMapping("/{id}/enable")
    public ApiResponse<DataSourceVO> enable(@PathVariable Long id) {
        return ApiResponse.ok(dataSourceService.enable(id));
    }

    /** Disable shortcut — equivalent to PATCH /{id}/status with status=0 */
    @PostMapping("/{id}/disable")
    public ApiResponse<DataSourceVO> disable(@PathVariable Long id) {
        return ApiResponse.ok(dataSourceService.disable(id));
    }

    // ─── Connection test ──────────────────────────────────────────────────────

    /**
     * Test connectivity using the provided credentials (nothing is saved).
     * Useful before creating a new datasource.
     */
    @PostMapping("/test")
    public ApiResponse<Map<String, Object>> testConnection(
            @Valid @RequestBody DataSourceRequest request) {
        log.debug("Test connection url={}", request.getUrl());
        boolean ok = dataSourceService.testConnection(request);
        log.info("Connection test url={} result={}", request.getUrl(), ok);
        return ApiResponse.ok(Map.of(
                "connected", ok,
                "message", ok ? "Connection successful" : "Connection failed"
        ));
    }

    /**
     * Test connectivity of an already-saved datasource using its stored credentials.
     * Returns {connected, message, datasourceId}.
     */
    @PostMapping("/{id}/test")
    public ApiResponse<Map<String, Object>> testConnectionById(@PathVariable Long id) {
        return ApiResponse.ok(dataSourceService.testConnectionById(id));
    }

    // ─── Pool management ──────────────────────────────────────────────────────

    /**
     * Reload the connection pool from the current DB config.
     * Use this after externally updating credentials.
     */
    @PostMapping("/{id}/reload")
    public ApiResponse<Void> reload(@PathVariable Long id) {
        dataSourceService.reload(id);
        return ApiResponse.ok();
    }

    /**
     * Get HikariCP pool metrics for a datasource.
     * Returns: registered, poolName, totalConnections, activeConnections,
     *          idleConnections, pendingThreads.
     */
    @GetMapping("/{id}/pool")
    public ApiResponse<Map<String, Object>> poolStatus(@PathVariable Long id) {
        return ApiResponse.ok(dataSourceService.getPoolStatus(id));
    }

    // ─── Tenant-scoped table access ──────────────────────────────────────────

    @GetMapping("/{id}/table-grants")
    public ApiResponse<List<DataSourceTableGrantVO>> listTableGrants(
            @PathVariable Long id,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId) {
        return ApiResponse.ok(tableAccessService.listGrants(resolveTenant(tenantId), id));
    }

    @PostMapping("/{id}/table-grants")
    public ApiResponse<DataSourceTableGrantVO> createTableGrant(
            @PathVariable Long id,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId,
            @Valid @RequestBody DataSourceTableGrantRequest request) {
        return ApiResponse.ok(tableAccessService.createGrant(resolveTenant(tenantId), id, request));
    }

    @PutMapping("/{id}/table-grants/{grantId}")
    public ApiResponse<DataSourceTableGrantVO> updateTableGrant(
            @PathVariable Long id,
            @PathVariable Long grantId,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId,
            @Valid @RequestBody DataSourceTableGrantRequest request) {
        return ApiResponse.ok(tableAccessService.updateGrant(resolveTenant(tenantId), id, grantId, request));
    }

    @DeleteMapping("/{id}/table-grants/{grantId}")
    public ApiResponse<Void> deleteTableGrant(
            @PathVariable Long id,
            @PathVariable Long grantId,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId) {
        tableAccessService.deleteGrant(resolveTenant(tenantId), id, grantId);
        return ApiResponse.ok();
    }

    @GetMapping("/{id}/schema/tables")
    public ApiResponse<List<DataSourceTableVO>> listSchemaTables(
            @PathVariable Long id,
            @RequestParam(required = false) String schemaName) {
        return ApiResponse.ok(tableAccessService.listPhysicalTables(id, schemaName));
    }

    @PostMapping("/{id}/query")
    public ApiResponse<MetricsQueryData> queryDatasource(
            @PathVariable Long id,
            @Valid @RequestBody SqlProbeRequest request) {
        return ApiResponse.ok(tableAccessService.queryDatasource(id, request));
    }

    @GetMapping("/{id}/tables")
    public ApiResponse<List<DataSourceTableVO>> listTables(
            @PathVariable Long id,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId) {
        return ApiResponse.ok(tableAccessService.listAuthorizedTables(resolveTenant(tenantId), id));
    }

    @GetMapping("/{id}/tables/{tableName}/columns")
    public ApiResponse<List<DataSourceColumnVO>> listColumns(
            @PathVariable Long id,
            @PathVariable String tableName,
            @RequestParam(required = false) String schemaName,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId) {
        return ApiResponse.ok(tableAccessService.listAuthorizedColumns(
                resolveTenant(tenantId), id, schemaName, tableName));
    }

    @PostMapping("/{id}/sql/probe")
    public ApiResponse<MetricsQueryData> probeSql(
            @PathVariable Long id,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId,
            @Valid @RequestBody SqlProbeRequest request) {
        return ApiResponse.ok(tableAccessService.probeSql(resolveTenant(tenantId), id, request));
    }

    private String resolveTenant(String tenantId) {
        return TenantHeaderResolver.resolve(tenantId);
    }
}
