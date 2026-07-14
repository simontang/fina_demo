package com.fina.cdp.dto;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.time.LocalDateTime;

@Data
public class MarketingCampaignRequest {

    @NotBlank(message = "name is required")
    private String name;

    private String description;

    @NotBlank(message = "type is required")
    private String type;

    private String status;

    @NotBlank(message = "goal is required")
    private String goal;

    @NotNull(message = "startTime is required")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime startTime;

    @NotNull(message = "endTime is required")
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
}
