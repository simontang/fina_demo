package com.fina.metrics.dto;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class RuntimeMetaChangeState {

    private String sourceName;
    private Long totalCount;
    private Long activeCount;
    private Long maxId;
    private LocalDateTime maxUpdatedAt;
    private String contentFingerprint;
}
