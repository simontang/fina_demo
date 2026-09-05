package com.fina.metrics.mapper;

import com.fina.metrics.dto.RuntimeMetaChangeState;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface RuntimeMetaChangeMapper {

    @Select("""
            SELECT 'datasource_config' AS source_name,
                   COUNT(*) AS total_count,
                   COALESCE(SUM(CASE WHEN deleted = 0 AND status = 1 THEN 1 ELSE 0 END), 0) AS active_count,
                   COALESCE(MAX(id), 0) AS max_id,
                   MAX(updated_at) AS max_updated_at,
                   MD5(COALESCE(STRING_AGG(TO_JSONB(source_row)::TEXT, ',' ORDER BY id), '')) AS content_fingerprint
              FROM t_datasource_config source_row
            UNION ALL
            SELECT 'datasource_table_grant' AS source_name,
                   COUNT(*) AS total_count,
                   COALESCE(SUM(CASE WHEN deleted = 0 AND status = 1 THEN 1 ELSE 0 END), 0) AS active_count,
                   COALESCE(MAX(id), 0) AS max_id,
                   MAX(updated_at) AS max_updated_at,
                   MD5(COALESCE(STRING_AGG(TO_JSONB(source_row)::TEXT, ',' ORDER BY id), '')) AS content_fingerprint
              FROM t_datasource_table_grant source_row
            UNION ALL
            SELECT 'metrics_meta' AS source_name,
                   COUNT(*) AS total_count,
                   COALESCE(SUM(CASE WHEN deleted = 0 AND status = 1 THEN 1 ELSE 0 END), 0) AS active_count,
                   COALESCE(MAX(id), 0) AS max_id,
                   MAX(updated_at) AS max_updated_at,
                   MD5(COALESCE(STRING_AGG(TO_JSONB(source_row)::TEXT, ',' ORDER BY id), '')) AS content_fingerprint
              FROM t_metrics_meta source_row
            UNION ALL
            SELECT 'metrics_meta_object' AS source_name,
                   COUNT(*) AS total_count,
                   COALESCE(SUM(CASE WHEN deleted = 0 AND status = 1 THEN 1 ELSE 0 END), 0) AS active_count,
                   COALESCE(MAX(id), 0) AS max_id,
                   MAX(updated_at) AS max_updated_at,
                   MD5(COALESCE(STRING_AGG(TO_JSONB(source_row)::TEXT, ',' ORDER BY id), '')) AS content_fingerprint
              FROM t_metrics_meta_object source_row
            """)
    List<RuntimeMetaChangeState> selectRuntimeMetaChangeState();
}
