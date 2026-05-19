package com.fina.b1s.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;
import lombok.Data;

/**
 * One entry in the Table/View index returned by GET .../meta (tablesIndex).
 * Lightweight discovery: which tables/views exist and basic identity.
 */
@Data
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class TableViewIndexItem {

    /** View or table name, e.g. VW_ORDN */
    private String tableName;

    /** Human-readable display name, e.g. docTypeEn or docType from view-*.json */
    private String displayName;

    /** Document type (Chinese), e.g. 退货 */
    private String docType;

    /** Document type (English), e.g. Sales Return */
    private String docTypeEn;

    /** Main SAP table code, e.g. ORDN */
    private String mainTable;

    /** Line table code, e.g. RDN1 */
    private String lineTable;

    /** Number of columns in this table/view */
    private Integer columnCount;

    /** Optional short description */
    private String shortDesc;
}
