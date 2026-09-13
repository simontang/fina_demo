package com.fina.platform.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.fina.platform.entity.FileObject;
import com.baomidou.mybatisplus.annotation.InterceptorIgnore;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

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
}
