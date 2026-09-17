package com.fina.metrics.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class DataSourcePublishedMetaVO {
    private MetricsMetaObjectVO metaObject;

    /** Legacy response field; null because publishing metadata never creates or changes scope rules. */
    private DataSourceTableGrantVO tableGrant;
}
