package com.fina.metrics.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * Flexible semantic metadata overlay stored in the master database.
 *
 * Static classpath metadata remains the base catalog. Active rows in this table
 * add or override catalog objects without changing the existing JSON files.
 */
@Data
@TableName("t_metrics_meta_object")
public class MetricsMetaObject {

    @TableId(type = IdType.AUTO)
    private Long id;

    /** Null means global metadata, otherwise scoped to one datasource. */
    private Long datasourceId;

    /** catalog_config | metric_index | metric_detail | table_catalog | table_view_detail */
    private String objectType;

    /** Stable object key, e.g. metric_name or table/view name. */
    private String objectKey;

    /** JSON object/array payload matching the existing static meta shapes. */
    private String payloadJson;

    /** 1 = active, 0 = inactive */
    private Integer status;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;

    @TableLogic
    private Integer deleted;
}
