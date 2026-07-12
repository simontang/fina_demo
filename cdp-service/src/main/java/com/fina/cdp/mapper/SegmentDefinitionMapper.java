package com.fina.cdp.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.fina.cdp.entity.SegmentDefinition;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;

@Mapper
public interface SegmentDefinitionMapper extends BaseMapper<SegmentDefinition> {

    @Select("""
            SELECT *
            FROM t_segment_definition
            WHERE tenant_id = #{tenantId}
              AND deleted = 0
            ORDER BY updated_at DESC, id DESC
            """)
    List<SegmentDefinition> selectByTenant(@Param("tenantId") String tenantId);

    @Select("""
            SELECT *
            FROM t_segment_definition
            WHERE tenant_id = #{tenantId}
              AND id = #{id}
              AND deleted = 0
            """)
    SegmentDefinition selectByTenantAndId(@Param("tenantId") String tenantId, @Param("id") Long id);

    @Select("""
            SELECT COUNT(1)
            FROM t_segment_definition
            WHERE tenant_id = #{tenantId}
              AND id = #{id}
              AND deleted = 0
            """)
    int existsByTenantAndId(@Param("tenantId") String tenantId, @Param("id") Long id);

    @Update("""
            UPDATE t_segment_definition
            SET deleted = 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE tenant_id = #{tenantId}
              AND id = #{id}
              AND deleted = 0
            """)
    int softDeleteByTenant(@Param("tenantId") String tenantId, @Param("id") Long id);
}
