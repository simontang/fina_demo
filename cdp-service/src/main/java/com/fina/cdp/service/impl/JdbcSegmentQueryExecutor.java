package com.fina.cdp.service.impl;

import com.fina.cdp.config.DynamicDataSourceManager;
import com.fina.cdp.service.SegmentQueryExecutor;
import lombok.RequiredArgsConstructor;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.stereotype.Service;

import java.sql.ResultSetMetaData;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class JdbcSegmentQueryExecutor implements SegmentQueryExecutor {

    private final DynamicDataSourceManager dataSourceManager;

    @Override
    public List<Map<String, Object>> query(Long datasourceId, String sql, Map<String, Object> params) {
        Map<String, Object> safeParams = params != null ? params : Map.of();
        return dataSourceManager.getNamedJdbcTemplate(datasourceId)
                .query(sql, new MapSqlParameterSource(safeParams), rs -> {
                    ResultSetMetaData meta = rs.getMetaData();
                    int columnCount = meta.getColumnCount();
                    List<Map<String, Object>> rows = new ArrayList<>();
                    while (rs.next()) {
                        Map<String, Object> row = new LinkedHashMap<>();
                        for (int i = 1; i <= columnCount; i++) {
                            row.put(meta.getColumnLabel(i), rs.getObject(i));
                        }
                        rows.add(row);
                    }
                    return rows;
                });
    }
}
