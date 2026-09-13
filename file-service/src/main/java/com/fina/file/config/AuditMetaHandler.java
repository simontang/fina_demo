package com.fina.file.config;

import com.baomidou.mybatisplus.core.handlers.MetaObjectHandler;
import com.fina.file.tenant.TenantContextHolder;
import org.apache.ibatis.reflection.MetaObject;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;

/**
 * Auto-fill audit columns on insert/update (bjy_crm_ai MetaObjectHandler
 * pattern). Fields are only filled when null — never overwrite explicit values.
 */
@Component
public class AuditMetaHandler implements MetaObjectHandler {

    @Override
    public void insertFill(MetaObject metaObject) {
        strictInsertFill(metaObject, "createdAt", LocalDateTime.class, LocalDateTime.now());
        strictInsertFill(metaObject, "updatedAt", LocalDateTime.class, LocalDateTime.now());
        String user = TenantContextHolder.getUser();
        strictInsertFill(metaObject, "createdBy", String.class, user != null ? user : "api");
    }

    @Override
    public void updateFill(MetaObject metaObject) {
        strictUpdateFill(metaObject, "updatedAt", LocalDateTime.class, LocalDateTime.now());
    }
}
