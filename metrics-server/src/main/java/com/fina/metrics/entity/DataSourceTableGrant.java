package com.fina.metrics.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("t_datasource_table_grant")
public class DataSourceTableGrant {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String tenantId;

    private Long datasourceId;

    private String schemaName;

    private String tablePattern;

    private String patternType;

    private Boolean caseSensitive;

    private Integer status;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;

    @TableLogic
    private Integer deleted;
}
