package com.fina.cdp.dto;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class SegmentDataVO {

    private Long id;
    private String tenantId;
    private Long definitionId;
    private String runId;
    private String dataJson;
    private Integer rowCount;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
