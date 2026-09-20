package com.fina.metrics.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("t_datasource_api_key")
public class DataSourceApiKey {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long datasourceId;

    private String keyName;

    private String keyHash;

    private String permissionsJson;

    private Integer status;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;

    @TableLogic
    private Integer deleted;
}
