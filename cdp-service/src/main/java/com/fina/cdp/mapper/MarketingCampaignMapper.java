package com.fina.cdp.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.fina.cdp.entity.MarketingCampaign;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface MarketingCampaignMapper extends BaseMapper<MarketingCampaign> {

    @Select("""
            <script>
            SELECT COUNT(1)
            FROM t_marketing_campaign
            WHERE tenant_id = #{tenantId}
              AND deleted = 0
              <if test="type != null and type != ''">
              AND type = #{type}
              </if>
              <if test="status != null and status != ''">
              AND status = #{status}
              </if>
              <if test="keyword != null and keyword != ''">
              AND LOWER(name) LIKE CONCAT('%', LOWER(#{keyword}), '%') ESCAPE '\\'
              </if>
            </script>
            """)
    long countByTenant(
            @Param("tenantId") String tenantId,
            @Param("type") String type,
            @Param("status") String status,
            @Param("keyword") String keyword);

    @Select("""
            <script>
            SELECT *
            FROM t_marketing_campaign
            WHERE tenant_id = #{tenantId}
              AND deleted = 0
              <if test="type != null and type != ''">
              AND type = #{type}
              </if>
              <if test="status != null and status != ''">
              AND status = #{status}
              </if>
              <if test="keyword != null and keyword != ''">
              AND LOWER(name) LIKE CONCAT('%', LOWER(#{keyword}), '%') ESCAPE '\\'
              </if>
            ORDER BY updated_at DESC, id DESC
            LIMIT #{limit}
            OFFSET #{offset}
            </script>
            """)
    List<MarketingCampaign> selectPageByTenant(
            @Param("tenantId") String tenantId,
            @Param("type") String type,
            @Param("status") String status,
            @Param("keyword") String keyword,
            @Param("limit") int limit,
            @Param("offset") long offset);

    @Select("""
            SELECT *
            FROM t_marketing_campaign
            WHERE tenant_id = #{tenantId}
              AND id = #{id}
              AND deleted = 0
            """)
    MarketingCampaign selectByTenantAndId(@Param("tenantId") String tenantId, @Param("id") Long id);

    @Update("""
            UPDATE t_marketing_campaign
            SET deleted = 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE tenant_id = #{tenantId}
              AND id = #{id}
              AND deleted = 0
            """)
    int softDeleteByTenant(@Param("tenantId") String tenantId, @Param("id") Long id);

    @Select("""
            SELECT *
            FROM t_marketing_campaign
            WHERE status = 'scheduled'
              AND deleted = 0
              AND start_time <= #{now}
              AND end_time > #{now}
            """)
    List<MarketingCampaign> selectScheduledDue(@Param("now") LocalDateTime now);

    @Select("""
            SELECT *
            FROM t_marketing_campaign
            WHERE status = 'running'
              AND deleted = 0
              AND end_time <= #{now}
            """)
    List<MarketingCampaign> selectRunningExpired(@Param("now") LocalDateTime now);
}
