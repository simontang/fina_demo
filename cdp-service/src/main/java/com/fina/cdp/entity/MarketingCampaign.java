package com.fina.cdp.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("t_marketing_campaign")
public class MarketingCampaign {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String tenantId;
    private String name;
    private String description;
    private String type;
    private String status;
    private String goal;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private Long mainSegmentDataId;
    private String segmentationStrategyJson;
    private String controlGroupStrategyJson;
    private String contentChannelStrategyJson;
    private String offerStrategyJson;
    private String waveStrategyJson;
    private String abTestStrategyJson;
    private String statisticsJson;
    private LocalDateTime actualStartedAt;
    private LocalDateTime actualStoppedAt;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;

    @TableLogic
    private Integer deleted;
}
