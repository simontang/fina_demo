package com.fina.metrics.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * SAP B1 HANA datasource configuration stored in the master PostgreSQL database.
 * Each row represents one dynamic HANA connection that can be used to query metrics.
 * Passwords are AES-encrypted at rest via EncryptUtil.
 */
@Data
@TableName("t_datasource_config")
public class DataSourceConfig {

    @TableId(type = IdType.AUTO)
    private Long id;

    /** Human-readable name, e.g. "SAP B1 Production" */
    private String name;

    /** SAP HANA JDBC URL, e.g. jdbc:sap://host:39015?currentSchema=MYSCHEMA */
    private String url;

    private String username;

    /** AES-encrypted HANA password */
    private String password;

    /** Optional default schema to set on every new connection */
    private String schemaName;

    /** Runtime datasource type, e.g. sap_b1_hana or cdp_postgres */
    private String sourceType;

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
