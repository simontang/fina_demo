package com.fina.metrics.controller;

import com.fina.metrics.dto.*;
import com.fina.metrics.service.MetricsService;
import com.fina.metrics.util.TenantHeaderResolver;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * Metrics API — discovery, definition management, and query execution.
 *
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │ Agent discovery flow (in order):                                            │
 * │  1. GET  /api/v1/datasources/{dsId}/metrics/index         → find metrics    │
 * │  2. GET  /api/v1/datasources/{dsId}/metrics/{name}/detail → understand them │
 * │  3. POST /api/v1/metrics/query                            → query data      │
 * ├─────────────────────────────────────────────────────────────────────────────┤
 * │ One-shot meta:                                                               │
 * │  GET    /api/v1/datasources/{dsId}/meta                  → index + details │
 * ├─────────────────────────────────────────────────────────────────────────────┤
 * │ Definition management (CRUD):                                               │
 * │  GET    /api/v1/datasources/{dsId}/metrics                → list DB defs   │
 * │  GET    /api/v1/datasources/{dsId}/metrics/{code}         → single DB def  │
 * │  POST   /api/v1/datasources/{dsId}/metrics                → create         │
 * │  PUT    /api/v1/metrics/{id}                              → update         │
 * │  DELETE /api/v1/metrics/{id}                              → delete         │
 * └─────────────────────────────────────────────────────────────────────────────┘
 */
@Slf4j
@RestController
@RequiredArgsConstructor
public class MetricsController {

    private final MetricsService metricsService;

    // ── Agent discovery ───────────────────────────────────────────────────────

    /**
     * Lightweight index: all catalog metrics enriched with datasource availability.
     *
     * registered=true  → SQL configured in t_metrics_meta, can query immediately
     * registered=false → in catalog but SQL not yet configured for this datasource
     */
    @GetMapping("/api/v1/datasources/{dsId}/metrics/index")
    public ApiResponse<MetricsIndexResponse> getMetricsIndex(
            @PathVariable Long dsId,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId) {
        return ApiResponse.ok(metricsService.getMetricsIndex(dsId, resolveTenant(tenantId)));
    }

    /**
     * Full agent context for a single metric.
     *
     * Returns catalog semantics (synonyms, thresholds, diagnostic workflow,
     * human-readable explanation) merged with the DB-stored SQL definition.
     * query_info is null when the metric is not registered for this datasource.
     */
    @GetMapping("/api/v1/datasources/{dsId}/metrics/{metricName}/detail")
    public ApiResponse<MetricsDetailResponse> getMetricDetail(
            @PathVariable Long dsId,
            @PathVariable String metricName,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId) {
        return ApiResponse.ok(metricsService.getMetricDetail(dsId, metricName, resolveTenant(tenantId)));
    }

    /**
     * One-shot: return metrics index and full detail for every metric.
     * GET /api/v1/datasources/{dsId}/meta
     */
    @GetMapping("/api/v1/datasources/{dsId}/meta")
    public ApiResponse<MetricsMetaFullResponse> getMetricsMeta(
            @PathVariable Long dsId,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId) {
        return ApiResponse.ok(metricsService.getMetricsMeta(dsId, resolveTenant(tenantId)));
    }

    // ── Query execution ───────────────────────────────────────────────────────

    /**
     * Execute a semantic metric query (BI Metrics Query API).
     *
     * Semantic mode (metrics[]): one SQL for all metrics (same table_view required).
     * Response data: semanticModel, columns (name+type), rows (value arrays), debug.
     *
     * Ad-hoc mode (custom_sql): executes SQL directly; semanticModel is "adhoc".
     */
    @PostMapping("/api/v1/metrics/query")
    public ApiResponse<MetricsQueryData> query(
            @Valid @RequestBody SemanticQueryRequest request,
            @RequestHeader(value = "X-Tenant-Id", required = false) String tenantId) {
        if (log.isInfoEnabled()) {
            log.info("Query datasource={} metrics={} customSql={}",
                    request.getDatasourceId(),
                    request.getMetrics(),
                    request.getCustomSql() != null ? "[adhoc]" : null);
        }
        return ApiResponse.ok(metricsService.query(request, resolveTenant(tenantId)));
    }

    // ── Metric definition CRUD ────────────────────────────────────────────────

    /** List all SQL-level metric definitions stored for a datasource */
    @GetMapping("/api/v1/datasources/{dsId}/metrics")
    public ApiResponse<List<MetricsMetaVO>> listMetrics(@PathVariable Long dsId) {
        return ApiResponse.ok(metricsService.listByDatasource(dsId));
    }

    /** Get a single SQL-level metric definition */
    @GetMapping("/api/v1/datasources/{dsId}/metrics/{code}")
    public ApiResponse<MetricsMetaVO> getMetric(
            @PathVariable Long dsId,
            @PathVariable String code) {
        return ApiResponse.ok(metricsService.getMetricMeta(dsId, code));
    }

    @PostMapping("/api/v1/datasources/{dsId}/metrics")
    public ApiResponse<MetricsMetaVO> createMetric(
            @PathVariable Long dsId,
            @Valid @RequestBody MetricsMetaRequest request) {
        request.setDatasourceId(dsId);
        return ApiResponse.ok(metricsService.createMetricMeta(request));
    }

    @PutMapping("/api/v1/metrics/{id}")
    public ApiResponse<MetricsMetaVO> updateMetric(
            @PathVariable Long id,
            @Valid @RequestBody MetricsMetaRequest request) {
        return ApiResponse.ok(metricsService.updateMetricMeta(id, request));
    }

    @DeleteMapping("/api/v1/metrics/{id}")
    public ApiResponse<Void> deleteMetric(@PathVariable Long id) {
        metricsService.deleteMetricMeta(id);
        return ApiResponse.ok();
    }

    private String resolveTenant(String tenantId) {
        return TenantHeaderResolver.resolve(tenantId);
    }
}
