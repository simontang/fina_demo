package com.fina.metrics.service;

import java.util.Set;

public final class MetricsMetaObjectTypes {
    public static final String CATALOG_CONFIG = "catalog_config";
    public static final String METRIC_INDEX = "metric_index";
    public static final String METRIC_DETAIL = "metric_detail";
    public static final String TABLE_CATALOG = "table_catalog";
    public static final String TABLE_VIEW_DETAIL = "table_view_detail";

    public static final Set<String> SUPPORTED = Set.of(
            CATALOG_CONFIG,
            METRIC_INDEX,
            METRIC_DETAIL,
            TABLE_CATALOG,
            TABLE_VIEW_DETAIL);

    private MetricsMetaObjectTypes() {
    }
}
