package com.fina.metrics.service;

import com.fina.metrics.dto.MetricsDetailResponse;
import com.fina.metrics.dto.MetricsIndexResponse;
import com.fina.metrics.dto.MetricsMetaFullResponse;
import com.fina.metrics.dto.MetricsMetaRequest;
import com.fina.metrics.dto.MetricsMetaVO;
import com.fina.metrics.dto.MetricsQueryData;
import com.fina.metrics.dto.SemanticQueryRequest;

import java.util.List;

public interface MetricsService {

    // ── Discovery / meta ──────────────────────────────────────────────────────

    /**
     * Returns the lightweight metrics index for a datasource.
     *
     * Merges the static catalog (metrics-index-meta.json) with DB registration
     * status: registered=true means a SQL definition exists in t_metrics_meta
     * for this datasource and the metric is immediately queryable.
     */
    MetricsIndexResponse getMetricsIndex(Long datasourceId);

    /**
     * Returns the full agent context for a single metric within a datasource.
     *
     * Merges catalog semantics (ai_agent_context, thresholds, synonyms) from
     * metrics-detail-meta.json with the DB-stored query definition
     * (SQL, parameter schema, value_column) from t_metrics_meta.
     *
     * query_info is null when the metric is not yet registered for this datasource.
     */
    MetricsDetailResponse getMetricDetail(Long datasourceId, String metricName);

    /**
     * Returns index and full detail for all metrics in one response.
     * GET /api/v1/datasources/{dsId}/meta
     */
    MetricsMetaFullResponse getMetricsMeta(Long datasourceId);

    // ── Metric definition CRUD ────────────────────────────────────────────────

    /** List all metric definitions stored in t_metrics_meta for a datasource */
    List<MetricsMetaVO> listByDatasource(Long datasourceId);

    /** Get a single metric definition by datasource + code */
    MetricsMetaVO getMetricMeta(Long datasourceId, String metricCode);

    /** Create a metric definition */
    MetricsMetaVO createMetricMeta(MetricsMetaRequest request);

    /** Update a metric definition */
    MetricsMetaVO updateMetricMeta(Long id, MetricsMetaRequest request);

    /** Soft-delete a metric definition */
    void deleteMetricMeta(Long id);

    // ── Query execution ───────────────────────────────────────────────────────

    /**
     * Execute a semantic query against the target datasource.
     *
     * Semantic mode (metrics[] set):
     *   One SQL is built for all requested metrics (same table_view required).
     *   Returns one result set: data.semanticModel, data.columns, data.rows, data.debug.
     *
     * Ad-hoc mode (custom_sql set):
     *   The SQL is executed directly; semanticModel is "adhoc".
     */
    MetricsQueryData query(SemanticQueryRequest request);
}
