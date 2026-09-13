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
}
