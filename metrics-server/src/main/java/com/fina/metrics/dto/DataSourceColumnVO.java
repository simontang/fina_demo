package com.fina.metrics.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class DataSourceColumnVO {

    private String schemaName;
    private String tableName;
    private String columnName;
    private Integer ordinalPosition;
    private Integer dataType;
    private String typeName;
    private Integer columnSize;
    private Boolean nullable;
    private String remarks;
}
