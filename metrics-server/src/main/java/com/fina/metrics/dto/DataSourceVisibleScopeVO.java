package com.fina.metrics.dto;

import lombok.Data;
import org.springframework.beans.BeanUtils;

import java.time.LocalDateTime;

/** Public datasource scope; deliberately contains no tenant identity. */
@Data
public class DataSourceVisibleScopeVO {
    private Long id;
    private Long datasourceId;
    private String schemaName;
    private String tablePattern;
    private String patternType;
    private Boolean caseSensitive;
    private Integer status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public static DataSourceVisibleScopeVO from(DataSourceTableGrantVO legacy) {
        DataSourceVisibleScopeVO result = new DataSourceVisibleScopeVO();
        BeanUtils.copyProperties(legacy, result);
        return result;
    }
}
