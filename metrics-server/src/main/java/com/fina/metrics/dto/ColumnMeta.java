package com.fina.metrics.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Column metadata for metrics query response (BI API doc).
 * Each item: { "name": "&lt;列名&gt;", "type": "&lt;数据类型&gt;" }
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ColumnMeta {

    private String name;
    private String type;
}
