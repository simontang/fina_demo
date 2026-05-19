package com.fina.b1s.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * Full detail for one table/view returned by GET .../meta (tablesDetails).
 * Aligns with view-*.json structure; when built from CSV, selectSql is null
 * and columns have name, description, type only.
 */
@Data
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class TableViewDetailResponse {

    private String tableName;
    private String docType;
    private String docTypeEn;
    private Integer objTypeCode;
    private String mainTable;
    private String lineTable;
    /** Present when source is view-*.json */
    private String selectSql;
    private List<ColumnMeta> columns;

    @Data
    @Builder
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class ColumnMeta {
        private String name;
        private String label;
        private String description;
        private String type;
        private String example;
    }
}
