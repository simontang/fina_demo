package com.fina.cdp.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class SegmentDataRequest {

    @NotNull(message = "definitionId is required")
    private Long definitionId;

    private String runId;

    @NotBlank(message = "dataJson is required")
    private String dataJson;
}
