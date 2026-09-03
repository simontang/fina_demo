package com.fina.metrics.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.util.Map;

@Data
public class SqlProbeRequest {

    @NotBlank(message = "sql is required")
    private String sql;

    private Map<String, Object> params;

    private Integer maxRows;

    private Boolean debug;
}
