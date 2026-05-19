package com.fina.b1s.dto;

import lombok.Data;

import java.util.List;
import java.util.Map;

/**
 * Result of a metric query execution.
 */
@Data
public class MetricsQueryResult {

    private Long datasourceId;
    private String metricCode;

    /** Column names in result order */
    private List<String> columns;

    /** Each row is a map of columnName → value */
    private List<Map<String, Object>> rows;

    private int rowCount;
    private long executionTimeMs;

    /** SQL that was actually executed (sanitized, no password info) */
    private String executedSql;
}
