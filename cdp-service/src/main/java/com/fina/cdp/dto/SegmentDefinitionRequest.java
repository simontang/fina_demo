package com.fina.cdp.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class SegmentDefinitionRequest {

    private String threadId;

    @NotBlank(message = "name is required")
    private String name;

    private String description;

    @NotNull(message = "datasourceId is required")
    private Long datasourceId;

    @NotBlank(message = "querySql is required")
    private String querySql;

    private Integer status;
}
