package com.fina.metrics.dto;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class DataSourcePublishedMetaRequest {

    /**
     * Optional object type. Table meta accepts table_catalog/table_view_detail.
     * Metric meta accepts metric_index/metric_detail.
     */
    private String objectType;

    /**
     * Optional stable key. If omitted, table meta derives it from tableName/viewName
     * and metric meta derives it from metric_name.
     */
    private String objectKey;

    @NotNull(message = "payload is required")
    private JsonNode payload;

    /** Defaults to active. */
    private Integer status;

    /**
     * Deprecated compatibility field, validated only; publishing table or metric
     * metadata never creates, updates, or deletes datasource scope rules.
     * When supplied, must match an existing active rule exactly (schema, pattern,
     * pattern type, case sensitivity, and active status). No broader or implicit
     * rule is inferred, even in ALL mode. Omit when no identical rule exists.
     */
    @Valid
    private DataSourceTableGrantRequest accessGrant;
}
