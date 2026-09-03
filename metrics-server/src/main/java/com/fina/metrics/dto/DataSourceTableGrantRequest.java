package com.fina.metrics.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class DataSourceTableGrantRequest {

    private String schemaName;

    @NotBlank(message = "tablePattern is required")
    private String tablePattern;

    private String patternType = "PREFIX";

    private Boolean caseSensitive = false;

    private Integer status = 1;
}
