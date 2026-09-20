package com.fina.metrics.controller;

import com.fina.metrics.dto.*;
import com.fina.metrics.service.DataSourceApiKeyAuthService;
import com.fina.metrics.service.DataSourceApiKeyPermission;
import com.fina.metrics.service.DataSourceApiKeyService;
import com.fina.metrics.service.DataSourceService;
import com.fina.metrics.service.DataSourceTableAccessService;
import com.fina.metrics.service.MetricsAdminAuthService;
import com.fina.metrics.util.TenantHeaderResolver;
import jakarta.servlet.http.HttpServletRequest;
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
    private final MetricsAdminAuthService adminAuth;
    private final DataSourceApiKeyAuthService datasourceAuth;
    private final DataSourceApiKeyService datasourceKeyService;

    @GetMapping("/current")
    public ApiResponse<DataSourceVO> getCurrentDatasource(HttpServletRequest httpRequest) {
        DataSourceApiKeyAuthService.AuthContext auth = datasourceAuth.requireAnyDatasourceKey(httpRequest);
        return ApiResponse.ok(dataSourceService.getById(auth.datasourceId()));
    }

    @GetMapping("/authorized")
    public ApiResponse<List<DataSourceVO>> listAuthorized(HttpServletRequest httpRequest) {
        DataSourceApiKeyAuthService.AuthContext auth = datasourceAuth.requireAnyDatasourceKey(httpRequest);
        return ApiResponse.ok(List.of(dataSourceService.getById(auth.datasourceId())));
    }

    @GetMapping("/{id:\\d+}/api-keys")
    public ApiResponse<List<DataSourceApiKeyVO>> listApiKeys(
            @PathVariable Long id,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(datasourceKeyService.list(id));
    }

    @PostMapping("/{id:\\d+}/api-keys")
    public ApiResponse<DataSourceApiKeyVO> createApiKey(
            @PathVariable Long id,
            @Valid @RequestBody DataSourceApiKeyRequest request,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(datasourceKeyService.create(id, request));
    }

    @PostMapping("/{id:\\d+}/api-keys/{keyId:\\d+}/disable")
    public ApiResponse<DataSourceApiKeyVO> disableApiKey(
            @PathVariable Long id,
            @PathVariable Long keyId,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(datasourceKeyService.disable(id, keyId));
    }

    @GetMapping("/{id:\\d+}/visible-scopes")
    public ApiResponse<List<DataSourceVisibleScopeVO>> listVisibleScopes(
            @PathVariable Long id,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(tableAccessService.listGrants(null, id).stream()
                .map(DataSourceVisibleScopeVO::from).toList());
    }

    @PostMapping("/{id:\\d+}/visible-scopes")
    public ApiResponse<DataSourceVisibleScopeVO> createVisibleScope(
            @PathVariable Long id,
            @Valid @RequestBody DataSourceVisibleScopeRequest request,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(DataSourceVisibleScopeVO.from(tableAccessService.createGrant(null, id, request)));
    }

    @PutMapping("/{id:\\d+}/visible-scopes/{scopeId:\\d+}")
    public ApiResponse<DataSourceVisibleScopeVO> updateVisibleScope(
            @PathVariable Long id, @PathVariable Long scopeId,
            @Valid @RequestBody DataSourceVisibleScopeRequest request,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(DataSourceVisibleScopeVO.from(tableAccessService.updateGrant(null, id, scopeId, request)));
    }

    @DeleteMapping("/{id:\\d+}/visible-scopes/{scopeId:\\d+}")
    public ApiResponse<Void> deleteVisibleScope(
            @PathVariable Long id,
            @PathVariable Long scopeId,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        tableAccessService.deleteGrant(null, id, scopeId);
        return ApiResponse.ok();
    }

    // ─── Query ────────────────────────────────────────────────────────────────

    @GetMapping
    public ApiResponse<List<DataSourceVO>> listAll(HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(dataSourceService.listAll());
    }

    @GetMapping("/active")
    public ApiResponse<List<DataSourceVO>> listActive(HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(dataSourceService.listActive());
    }

    @GetMapping("/{id:\\d+}")
    public ApiResponse<DataSourceVO> getById(@PathVariable Long id, HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(dataSourceService.getById(id));
    }

    // ─── Create / Update / Delete ─────────────────────────────────────────────

    /**
     * Create a new SAP B1 HANA datasource.
     * All fields are required. Password is plain-text and will be AES-encrypted at rest.
     */
    @PostMapping
    public ApiResponse<DataSourceVO> create(
            @Valid @RequestBody DataSourceRequest request,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        log.debug("Create datasource name={} url={}", request.getName(), request.getUrl());
        return ApiResponse.ok(dataSourceService.create(request));
    }

    /**
     * Update a datasource.
     * Password is optional — omit or leave blank to keep the current password.
     */
    @PutMapping("/{id:\\d+}")
    public ApiResponse<DataSourceVO> update(
            @PathVariable Long id,
            @Valid @RequestBody DataSourceUpdateRequest request,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        log.debug("Update datasource id={} name={}", id, request.getName());
        return ApiResponse.ok(dataSourceService.update(id, request));
    }

    /**
     * Soft-delete a datasource. The connection pool is closed immediately.
     */
    @DeleteMapping("/{id:\\d+}")
    public ApiResponse<Void> delete(@PathVariable Long id, HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        log.debug("Delete datasource id={}", id);
        dataSourceService.delete(id);
        return ApiResponse.ok();
    }

    // ─── Status toggle ────────────────────────────────────────────────────────

    /**
     * Set datasource status (1=active, 0=inactive).
     * Activating registers the pool; deactivating closes it.
     */
    @PatchMapping("/{id:\\d+}/status")
    public ApiResponse<DataSourceVO> setStatus(
            @PathVariable Long id,
            @Valid @RequestBody DataSourceStatusRequest request,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(dataSourceService.setStatus(id, request.getStatus()));
    }

    /** Enable shortcut — equivalent to PATCH /{id}/status with status=1 */
    @PostMapping("/{id:\\d+}/enable")
    public ApiResponse<DataSourceVO> enable(@PathVariable Long id, HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(dataSourceService.enable(id));
    }

    /** Disable shortcut — equivalent to PATCH /{id}/status with status=0 */
    @PostMapping("/{id:\\d+}/disable")
    public ApiResponse<DataSourceVO> disable(@PathVariable Long id, HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(dataSourceService.disable(id));
    }

    // ─── Connection test ──────────────────────────────────────────────────────

    /**
     * Test connectivity using the provided credentials (nothing is saved).
     * Useful before creating a new datasource.
     */
    @PostMapping("/test")
    public ApiResponse<Map<String, Object>> testConnection(
            @Valid @RequestBody DataSourceRequest request,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
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
    @PostMapping("/{id:\\d+}/test")
    public ApiResponse<Map<String, Object>> testConnectionById(
            @PathVariable Long id,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(dataSourceService.testConnectionById(id));
    }

    // ─── Pool management ──────────────────────────────────────────────────────

    /**
     * Reload the connection pool from the current DB config.
     * Use this after externally updating credentials.
     */
    @PostMapping("/{id:\\d+}/reload")
    public ApiResponse<Void> reload(@PathVariable Long id, HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        dataSourceService.reload(id);
        return ApiResponse.ok();
    }

    /**
     * Get HikariCP pool metrics for a datasource.
     * Returns: registered, poolName, totalConnections, activeConnections,
     *          idleConnections, pendingThreads.
     */
    @GetMapping("/{id:\\d+}/pool")
    public ApiResponse<Map<String, Object>> poolStatus(@PathVariable Long id, HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(dataSourceService.getPoolStatus(id));
    }

    // Legacy scope routes retain their request shape; tenant headers do not affect visibility.

    @GetMapping("/{id:\\d+}/table-grants")
    public ApiResponse<List<DataSourceTableGrantVO>> listTableGrants(
            @PathVariable Long id,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(tableAccessService.listGrants(resolveTenant(tenantId), id));
    }

    @PostMapping("/{id:\\d+}/table-grants")
    public ApiResponse<DataSourceTableGrantVO> createTableGrant(
            @PathVariable Long id,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId,
            @Valid @RequestBody DataSourceTableGrantRequest request,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(tableAccessService.createGrant(resolveTenant(tenantId), id, request));
    }

    @PutMapping("/{id:\\d+}/table-grants/{grantId:\\d+}")
    public ApiResponse<DataSourceTableGrantVO> updateTableGrant(
            @PathVariable Long id,
            @PathVariable Long grantId,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId,
            @Valid @RequestBody DataSourceTableGrantRequest request,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        return ApiResponse.ok(tableAccessService.updateGrant(resolveTenant(tenantId), id, grantId, request));
    }

    @DeleteMapping("/{id:\\d+}/table-grants/{grantId:\\d+}")
    public ApiResponse<Void> deleteTableGrant(
            @PathVariable Long id,
            @PathVariable Long grantId,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId,
            HttpServletRequest httpRequest) {
        adminAuth.requireAdmin(httpRequest);
        tableAccessService.deleteGrant(resolveTenant(tenantId), id, grantId);
        return ApiResponse.ok();
    }

    @GetMapping("/{id:\\d+}/schema/tables")
    public ApiResponse<List<DataSourceTableVO>> listSchemaTables(
            @PathVariable Long id,
            @RequestParam(required = false) String schemaName,
            HttpServletRequest httpRequest) {
        datasourceAuth.requirePermission(httpRequest, id, DataSourceApiKeyPermission.QUERY);
        return ApiResponse.ok(tableAccessService.listPhysicalTables(id, schemaName));
    }

    @PostMapping("/{id:\\d+}/query")
    public ApiResponse<MetricsQueryData> queryDatasource(
            @PathVariable Long id,
            @Valid @RequestBody SqlProbeRequest request,
            HttpServletRequest httpRequest) {
        datasourceAuth.requirePermission(httpRequest, id, DataSourceApiKeyPermission.QUERY);
        return ApiResponse.ok(tableAccessService.queryDatasource(id, request));
    }

    @GetMapping("/{id:\\d+}/tables")
    public ApiResponse<List<DataSourceTableVO>> listTables(
            @PathVariable Long id,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId,
            HttpServletRequest httpRequest) {
        datasourceAuth.requirePermission(httpRequest, id, DataSourceApiKeyPermission.QUERY);
        return ApiResponse.ok(tableAccessService.listAuthorizedTables(resolveTenant(tenantId), id));
    }

    @GetMapping("/{id:\\d+}/tables/{tableName}/columns")
    public ApiResponse<List<DataSourceColumnVO>> listColumns(
            @PathVariable Long id,
            @PathVariable String tableName,
            @RequestParam(required = false) String schemaName,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId,
            HttpServletRequest httpRequest) {
        datasourceAuth.requirePermission(httpRequest, id, DataSourceApiKeyPermission.QUERY);
        return ApiResponse.ok(tableAccessService.listAuthorizedColumns(
                resolveTenant(tenantId), id, schemaName, tableName));
    }

    @PostMapping("/{id:\\d+}/sql/probe")
    public ApiResponse<MetricsQueryData> probeSql(
            @PathVariable Long id,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId,
            @Valid @RequestBody SqlProbeRequest request,
            HttpServletRequest httpRequest) {
        datasourceAuth.requirePermission(httpRequest, id, DataSourceApiKeyPermission.QUERY);
        return ApiResponse.ok(tableAccessService.probeSql(resolveTenant(tenantId), id, request));
    }

    private String resolveTenant(String tenantId) {
        return TenantHeaderResolver.resolve(tenantId);
    }
}
