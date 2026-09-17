package com.fina.platform.bo;

import java.util.List;
import java.util.Map;

public final class BusinessObjectDtos {
    private BusinessObjectDtos() {
    }

    public record StoreRequest(
            String storeKey,
            String name,
            String description,
            String jdbcUrl,
            String schemaName,
            String username,
            String password,
            Integer status
    ) {
    }

    public record StoreResponse(
            Long id,
            String storeKey,
            String name,
            String description,
            String jdbcUrl,
            String schemaName,
            String username,
            Integer status
    ) {
    }

    public record StoreGrantRequest(
            String granteeKey,
            Boolean canRead,
            Boolean canWrite,
            Boolean canManage,
            Integer status
    ) {
    }

    public record StoreGrantResponse(
            Long id,
            Long storeId,
            String storeKey,
            String granteeKey,
            Boolean canRead,
            Boolean canWrite,
            Boolean canManage,
            Integer status
    ) {
    }

    public record ObjectDefinitionRequest(
            String storeKey,
            String objectKey,
            String displayName,
            String description,
            List<FieldDefinition> fields,
            List<IndexDefinition> indexes,
            Integer status
    ) {
    }

    public record ObjectDefinitionResponse(
            Long id,
            Long storeId,
            String storeKey,
            String objectKey,
            String tableName,
            String displayName,
            String description,
            List<FieldDefinition> fields,
            List<IndexDefinition> indexes,
            Integer status
    ) {
    }

    public record FieldDefinition(
            String key,
            String type,
            Boolean required,
            Integer maxLength,
            Integer precision,
            Integer scale,
            String description
    ) {
    }

    public record IndexDefinition(
            String name,
            List<String> fields,
            Boolean unique
    ) {
    }

    public record RecordRequest(Map<String, Object> data) {
    }

    public record RecordResponse(
            String id,
            String objectKey,
            Map<String, Object> data
    ) {
    }

    public record QueryRequest(
            List<Filter> filters,
            List<Sort> sort,
            Integer page,
            Integer pageSize
    ) {
    }

    public record Filter(
            String field,
            String op,
            Object value
    ) {
    }

    public record Sort(
            String field,
            String direction
    ) {
    }

    public record QueryResponse(
            String objectKey,
            Integer page,
            Integer pageSize,
            Long total,
            List<Map<String, Object>> rows
    ) {
    }
}
