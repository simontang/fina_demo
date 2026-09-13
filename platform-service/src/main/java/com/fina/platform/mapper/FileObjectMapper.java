package com.fina.platform.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.fina.platform.entity.FileObject;
import com.baomidou.mybatisplus.annotation.InterceptorIgnore;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface FileObjectMapper extends BaseMapper<FileObject> {

    /**
     * Ticket-based downloads arrive without a tenant header, so the tenant
     * line interceptor must not fire here. Safe: the uuid is an unguessable
     * capability (128-bit random), and the row must be active.
     */
    @InterceptorIgnore(tenantLine = "true")
    @Select("SELECT * FROM file_objects WHERE uuid = #{uuid} AND status = 'active' LIMIT 1")
    FileObject selectActiveByUuidIgnoreTenant(@Param("uuid") String uuid);

    /**
     * Immediate sub-directory names under a path prefix, computed in SQL.
     * startPos is the 1-based char position after "{prefix}/".
     */
    @Select("""
            <script>
            SELECT DISTINCT split_part(substring(path from #{startPos}), '/', 1) AS segment
            FROM file_objects
            WHERE status = 'active'
              AND path LIKE #{like}
              <if test="excludeEmpty">AND path &lt;&gt; ''</if>
              AND split_part(substring(path from #{startPos}), '/', 1) &lt;&gt; ''
            ORDER BY segment
            LIMIT #{limit}
            </script>
            """)
    List<String> listSubdirectories(@Param("startPos") int startPos,
                                    @Param("like") String like,
                                    @Param("excludeEmpty") boolean excludeEmpty,
                                    @Param("limit") int limit);

    /** Files directly under a directory, keyset-paginated by filename. */
    @Select("""
            <script>
            SELECT * FROM file_objects
            WHERE status = 'active' AND path = #{dir}
              <if test="cursor != null and cursor != ''">AND filename &gt; #{cursor}</if>
            ORDER BY filename
            LIMIT #{limit}
            </script>
            """)
    List<FileObject> listFilesInDir(@Param("dir") String dir,
                                    @Param("cursor") String cursor,
                                    @Param("limit") int limit);

    /**
     * Substring + attribute search, newest first, keyset-paginated by id.
     * Filters are all optional; the tenant interceptor adds tenant_id.
     */
    @Select("""
            <script>
            SELECT * FROM file_objects
            WHERE status = 'active'
              <if test="q != null and q != ''">
                AND (filename ILIKE '%' || #{q} || '%' OR path ILIKE '%' || #{q} || '%')
              </if>
              <if test="prefix != null and prefix != ''">
                AND (path = #{prefix} OR path LIKE #{prefix} || '/%')
              </if>
              <if test="fileCategory != null and fileCategory != ''">
                AND file_category = #{fileCategory}
              </if>
              <if test="usage != null and usage != ''">
                AND usage = #{usage}
              </if>
              <if test="from != null">AND created_at &gt;= #{from}</if>
              <if test="to != null">AND created_at &lt;= #{to}</if>
              <if test="beforeId != null">AND id &lt; #{beforeId}</if>
            ORDER BY id DESC
            LIMIT #{limit}
            </script>
            """)
    List<FileObject> search(@Param("q") String q,
                            @Param("prefix") String prefix,
                            @Param("fileCategory") String fileCategory,
                            @Param("usage") String usage,
                            @Param("from") java.time.LocalDateTime from,
                            @Param("to") java.time.LocalDateTime to,
                            @Param("beforeId") Long beforeId,
                            @Param("limit") int limit);
}
