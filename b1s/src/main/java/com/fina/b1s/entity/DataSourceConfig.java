package com.fina.b1s.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("t_datasource_config")
public class DataSourceConfig {

    @TableId(type = IdType.AUTO)
    private Long id;

    /** Human-readable name, e.g. "SAP B1 Production" */
    private String name;

    private String url;

    private String username;

    private String password;

    private String schemaName;

    @TableField(exist = false)
    private String instanceType = "SQLSERVER";

    private String description;

    /** 1 = active, 0 = inactive */
    private Integer status;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;

    @TableLogic
    private Integer deleted;
}
