package com.fina.cdp.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.fina.cdp.entity.SegmentData;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;

@Mapper
public interface SegmentDataMapper extends BaseMapper<SegmentData> {

    @Select("""
            <script>
            SELECT COUNT(1)
            FROM t_segment_data
            WHERE tenant_id = #{tenantId}
              AND deleted = 0
              <if test="definitionId != null">
              AND definition_id = #{definitionId}
              </if>
            </script>
            """)
    long countByTenant(@Param("tenantId") String tenantId, @Param("definitionId") Long definitionId);

    @Select("""
            <script>
            SELECT *
            FROM t_segment_data
            WHERE tenant_id = #{tenantId}
              AND deleted = 0
              <if test="definitionId != null">
              AND definition_id = #{definitionId}
              </if>
            ORDER BY created_at DESC, id DESC
            LIMIT #{limit}
            OFFSET #{offset}
            </script>
            """)
    List<SegmentData> selectPageByTenant(
            @Param("tenantId") String tenantId,
            @Param("definitionId") Long definitionId,
            @Param("limit") int limit,
            @Param("offset") int offset);

    @Select("""
            SELECT *
            FROM t_segment_data
            WHERE tenant_id = #{tenantId}
              AND id = #{id}
              AND deleted = 0
            """)
    SegmentData selectByTenantAndId(@Param("tenantId") String tenantId, @Param("id") Long id);

    @Update("""
            UPDATE t_segment_data
            SET deleted = 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE tenant_id = #{tenantId}
              AND id = #{id}
              AND deleted = 0
            """)
    int softDeleteByTenant(@Param("tenantId") String tenantId, @Param("id") Long id);
}
