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
     * The one listing/search query: scope to a folder, optionally recurse,
     * optionally filter by name substring / attributes / time, newest first,
     * offset-paginated.
     */
    @Select("""
            <script>
            SELECT * FROM file_objects
            WHERE status = 'active'
              <choose>
                <when test="recursive">AND (path = #{path} OR path LIKE #{pathPrefix})</when>
                <otherwise>AND path = #{path}</otherwise>
              </choose>
              <if test="q != null and q != ''">AND filename ILIKE '%' || #{q} || '%'</if>
              <if test="fileCategory != null and fileCategory != ''">AND file_category = #{fileCategory}</if>
              <if test="usage != null and usage != ''">AND usage = #{usage}</if>
              <if test="meta != null and meta != ''">AND meta @&gt; CAST(#{meta} AS jsonb)</if>
              <if test="from != null">AND created_at &gt;= #{from}</if>
              <if test="to != null">AND created_at &lt;= #{to}</if>
            ORDER BY id DESC
            LIMIT #{size} OFFSET #{offset}
            </script>
            """)
    List<FileObject> query(@Param("path") String path,
                           @Param("pathPrefix") String pathPrefix,
                           @Param("recursive") boolean recursive,
                           @Param("q") String q,
                           @Param("fileCategory") String fileCategory,
                           @Param("usage") String usage,
                           @Param("meta") String meta,
                           @Param("from") java.time.LocalDateTime from,
                           @Param("to") java.time.LocalDateTime to,
                           @Param("size") int size,
                           @Param("offset") int offset);

    /** Total hits for the same filters, so a page-number UI can render. */
    @Select("""
            <script>
            SELECT count(*) FROM file_objects
            WHERE status = 'active'
              <choose>
                <when test="recursive">AND (path = #{path} OR path LIKE #{pathPrefix})</when>
                <otherwise>AND path = #{path}</otherwise>
              </choose>
              <if test="q != null and q != ''">AND filename ILIKE '%' || #{q} || '%'</if>
              <if test="fileCategory != null and fileCategory != ''">AND file_category = #{fileCategory}</if>
              <if test="usage != null and usage != ''">AND usage = #{usage}</if>
              <if test="meta != null and meta != ''">AND meta @&gt; CAST(#{meta} AS jsonb)</if>
              <if test="from != null">AND created_at &gt;= #{from}</if>
              <if test="to != null">AND created_at &lt;= #{to}</if>
            </script>
            """)
    long countQuery(@Param("path") String path,
                    @Param("pathPrefix") String pathPrefix,
                    @Param("recursive") boolean recursive,
                    @Param("q") String q,
                    @Param("fileCategory") String fileCategory,
                    @Param("usage") String usage,
                    @Param("meta") String meta,
                    @Param("from") java.time.LocalDateTime from,
                    @Param("to") java.time.LocalDateTime to);

    /** Next-level folder names under a path (used by the non-recursive view). */
    @Select("""
            <script>
            SELECT DISTINCT split_part(substring(path from #{startPos}), '/', 1) AS segment
            FROM file_objects
            WHERE status = 'active' AND path LIKE #{like}
              AND split_part(substring(path from #{startPos}), '/', 1) &lt;&gt; ''
            ORDER BY segment
            LIMIT #{limit}
            </script>
            """)
    List<String> listSubdirectories(@Param("startPos") int startPos,
                                    @Param("like") String like,
                                    @Param("limit") int limit);
}
