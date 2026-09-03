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
     * Optional table access backing. If omitted for table meta, an EXACT grant is
     * derived from tableName/viewName/objectKey.
     */
    @Valid
    private DataSourceTableGrantRequest accessGrant;
}
