package com.fina.metrics.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * Metric definition: a named query bound to a specific datasource.
 *
 * querySql supports named parameters using :paramName syntax,
 * e.g. SELECT * FROM SALES WHERE SALE_DATE >= :startDate AND SALE_DATE <= :endDate
 *
 * parameters is a JSON array describing each parameter:
 * [{"name":"startDate","type":"STRING","required":true,"description":"Start date (yyyy-MM-dd)"},...]
 */
@Data
@TableName("t_metrics_meta")
public class MetricsMeta {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long datasourceId;

    /** Unique code within a datasource, e.g. "SALES_BY_DATE" */
    private String metricCode;

    private String metricName;

    private String description;

    /** SQL with :paramName placeholders */
    private String querySql;

    /** JSON array of parameter definitions */
    private String parameters;

    /** Result column to use as the primary metric value (optional) */
    private String valueColumn;

    /** 1 = active, 0 = inactive */
    private Integer status;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;

    @TableLogic
    private Integer deleted;
}
