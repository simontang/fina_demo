package com.fina.platform.bo;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.time.LocalDateTime;
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

    public record StoreApiKeyRequest(
            String keyName,
            String rawKey,
            List<String> permissions,
            Integer status
    ) {
    }

    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record StoreApiKeyResponse(
            Long id,
            Long storeId,
            String storeKey,
            String keyName,
            List<String> permissions,
            Integer status,
            LocalDateTime createdAt,
            LocalDateTime updatedAt,
            String rawKey
    ) {
    }

    public record ObjectDefinitionRequest(
            String storeKey,
            String objectKey,
            String displayName,
            String description,
            List<FieldDefinition> fields,
            List<IndexDefinition> indexes,
            Integer status,
            String deleteMode
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
            Integer status,
            String deleteMode
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

    public record BatchRecordRequest(List<Map<String, Object>> records) {
    }

    public record BatchDeleteRequest(List<String> ids) {
    }

    public record BatchCreateResponse(
            String objectKey,
            int created,
            List<String> ids
    ) {
    }

    public record BatchDeleteResponse(
            String objectKey,
            int deleted,
            List<String> ids
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
