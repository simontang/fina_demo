package com.fina.metrics.controller;

import com.fina.metrics.dto.*;
import com.fina.metrics.service.MetricsService;
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
    public ApiResponse<MetricsIndexResponse> getMetricsIndex(@PathVariable Long dsId) {
        return ApiResponse.ok(metricsService.getMetricsIndex(dsId));
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
            @PathVariable String metricName) {
        return ApiResponse.ok(metricsService.getMetricDetail(dsId, metricName));
    }

    /**
     * One-shot: return metrics index and full detail for every metric.
     * GET /api/v1/datasources/{dsId}/meta
     */
    @GetMapping("/api/v1/datasources/{dsId}/meta")
    public ApiResponse<MetricsMetaFullResponse> getMetricsMeta(@PathVariable Long dsId) {
        return ApiResponse.ok(metricsService.getMetricsMeta(dsId));
    }

    // ── Query execution ───────────────────────────────────────────────────────

    /**
     * Execute a semantic metric query against a SAP B1 HANA datasource.
     *
     * Semantic mode (metrics[] populated):
     *   One HANA SQL is generated per metric from the catalog definition.
     *   All queries run in parallel; results are returned as response.results[].
     *   A single metric failure is captured in MetricResult.error without
     *   aborting the other metrics.
     *
     * Ad-hoc mode (custom_sql populated):
     *   Executes the SQL directly. metrics/groupBy/filters/orderBy are ignored.
     *   Use :paramName placeholders and supply values in params.
     *
     * See SemanticQueryRequest for supported operators and groupBy granularity syntax.
     */
    @PostMapping("/api/v1/metrics/query")
    public ApiResponse<SemanticQueryResponse> query(
            @Valid @RequestBody SemanticQueryRequest request) {
        if (log.isInfoEnabled()) {
            log.info("Query datasource={} metrics={} customSql={}",
                    request.getDatasourceId(),
                    request.getMetrics(),
                    request.getCustomSql() != null ? "[adhoc]" : null);
        }
        return ApiResponse.ok(metricsService.query(request));
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
}
