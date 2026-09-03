package com.fina.metrics.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;
import lombok.Data;

import java.util.List;
import java.util.Map;

/**
 * Response data for POST /api/v1/metrics/query (BI Metrics Query API doc).
 *
 * data.semanticModel — semantic model name (e.g. source.table_view)
 * data.columns      — column metadata (name + type), order matches each row
 * data.rows         — array of value arrays (not objects), same order as columns
 * data.rowCount     — number of returned rows, when populated by probe-style APIs
 * data.debug        — present only when request.debug=true (sql, params, etc.)
 */
@Data
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class MetricsQueryData {

    private String semanticModel;
    private List<ColumnMeta> columns;
    private List<List<Object>> rows;
    private Integer rowCount;
    /**
     * Only when request.debug == true. May contain "sql", "params", etc.
     */
    private Map<String, Object> debug;
}
