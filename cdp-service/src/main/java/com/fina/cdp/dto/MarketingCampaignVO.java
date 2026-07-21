package com.fina.cdp.dto;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.databind.JsonNode;
import lombok.Data;

import java.time.LocalDateTime;

@Data
public class MarketingCampaignVO {

    private Long id;
    private String tenantId;
    private String threadId;
    private String name;
    private String description;
    private String type;
    private String status;
    private String goal;
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime startTime;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime endTime;
    private Long mainSegmentDataId;
    private JsonNode segmentationStrategy;
    private JsonNode controlGroupStrategy;
    private JsonNode contentChannelStrategy;
    private JsonNode offerStrategy;
    private JsonNode waveStrategy;
    private JsonNode abTestStrategy;
    private JsonNode statistics;
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime actualStartedAt;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime actualStoppedAt;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createdAt;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updatedAt;
}
