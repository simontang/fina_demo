package com.fina.b1s.service;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.List;
import java.util.Optional;

/**
 * Provides read-only access to the static metric catalog JSON files.
 *
 * metrics-index-meta.json  — lightweight discovery: display names, domains, keywords
 * metrics-detail-meta.json — full AI agent context per metric
 *
 * Both files are loaded once at startup and cached in memory.
 */
public interface MetaCatalogService {

    /** All metric entries from metrics-index-meta.json */
    List<JsonNode> getIndexItems();

    /** Catalog version string, e.g. "1.0" */
    String getCatalogVersion();

    /** Domain category list from the index file */
    List<String> getDomainCategories();

    /**
     * Find a single index item by metric_name.
     * Returns empty if the metric is not in the catalog.
     */
    Optional<JsonNode> findIndexItem(String metricName);

    /**
     * Find the full detail entry by metric_name from metrics-detail-meta.json.
     * Returns empty if the metric is not in the catalog.
     */
    Optional<JsonNode> findDetailItem(String metricName);
}
