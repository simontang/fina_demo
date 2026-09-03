package com.fina.metrics.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class DataSourceTableVO {

    private String schemaName;
    private String tableName;
    private String tableType;
    private String remarks;
}
