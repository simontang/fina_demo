package com.fina.metrics.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * Public view of a datasource — password is never exposed.
 */
@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
public class DataSourceVO {
    private Long id;
    private String name;
    private String url;
    private String username;
    private String schemaName;
    private String sourceType;
    private String description;
    private Integer status;
    private String statusLabel;
    private Boolean connected;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
