package com.fina.metrics.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class DataSourcePublishedMetaVO {
    private MetricsMetaObjectVO metaObject;
    private DataSourceTableGrantVO tableGrant;
}
