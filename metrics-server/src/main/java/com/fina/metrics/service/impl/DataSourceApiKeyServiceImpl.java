package com.fina.metrics.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.metrics.dto.DataSourceApiKeyRequest;
import com.fina.metrics.dto.DataSourceApiKeyVO;
import com.fina.metrics.entity.DataSourceApiKey;
import com.fina.metrics.entity.DataSourceConfig;
import com.fina.metrics.mapper.DataSourceApiKeyMapper;
import com.fina.metrics.mapper.DataSourceConfigMapper;
import com.fina.metrics.service.DataSourceApiKeyPermission;
import com.fina.metrics.service.DataSourceApiKeyService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

import java.security.SecureRandom;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class DataSourceApiKeyServiceImpl implements DataSourceApiKeyService {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final SecureRandom RANDOM = new SecureRandom();

    private final DataSourceApiKeyMapper keyMapper;
    private final DataSourceConfigMapper datasourceMapper;

    @Override
    public List<DataSourceApiKeyVO> list(Long datasourceId) {
        requireDatasource(datasourceId);
        return keyMapper.selectList(new LambdaQueryWrapper<DataSourceApiKey>()
                        .eq(DataSourceApiKey::getDatasourceId, datasourceId)
                        .eq(DataSourceApiKey::getDeleted, 0)
                        .orderByAsc(DataSourceApiKey::getKeyName))
                .stream()
                .map(this::toVO)
                .toList();
    }

    @Override
    @Transactional
    public DataSourceApiKeyVO create(Long datasourceId, DataSourceApiKeyRequest request) {
        requireDatasource(datasourceId);
        List<String> permissions = normalizePermissions(request.getPermissions());
        String rawKey = StringUtils.hasText(request.getRawKey()) ? request.getRawKey().trim() : generateKey();
        DataSourceApiKey key = new DataSourceApiKey();
        key.setDatasourceId(datasourceId);
        key.setKeyName(request.getKeyName().trim());
        key.setKeyHash(DataSourceApiKeyAuthServiceImpl.hash(rawKey));
        key.setPermissionsJson(toJson(permissions));
        key.setStatus(1);
        key.setDeleted(0);
        keyMapper.insert(key);
        DataSourceApiKeyVO vo = toVO(key);
        if (!StringUtils.hasText(request.getRawKey())) {
            vo.setRawKey(rawKey);
        }
        return vo;
    }

    @Override
    @Transactional
    public DataSourceApiKeyVO disable(Long datasourceId, Long keyId) {
        requireDatasource(datasourceId);
        DataSourceApiKey key = requireKey(datasourceId, keyId);
        key.setStatus(0);
        keyMapper.updateById(key);
        return toVO(key);
    }

    private DataSourceApiKey requireKey(Long datasourceId, Long keyId) {
        DataSourceApiKey key = keyMapper.selectOne(new LambdaQueryWrapper<DataSourceApiKey>()
                .eq(DataSourceApiKey::getId, keyId)
                .eq(DataSourceApiKey::getDatasourceId, datasourceId)
                .eq(DataSourceApiKey::getDeleted, 0));
        if (key == null) {
            throw new IllegalArgumentException("Datasource API key not found: " + keyId);
        }
        return key;
    }

    private void requireDatasource(Long datasourceId) {
        DataSourceConfig datasource = datasourceMapper.selectOne(new LambdaQueryWrapper<DataSourceConfig>()
                .eq(DataSourceConfig::getId, datasourceId)
                .eq(DataSourceConfig::getDeleted, 0));
        if (datasource == null) {
            throw new IllegalArgumentException("Datasource not found: " + datasourceId);
        }
    }

    private List<String> normalizePermissions(List<String> requested) {
        Set<String> allowed = Arrays.stream(DataSourceApiKeyPermission.values())
                .map(Enum::name)
                .collect(Collectors.toSet());
        List<String> permissions = requested == null || requested.isEmpty()
                ? Arrays.stream(DataSourceApiKeyPermission.values()).map(Enum::name).toList()
                : requested.stream()
                        .filter(StringUtils::hasText)
                        .map(value -> value.trim().toUpperCase())
                        .distinct()
                        .toList();
        for (String permission : permissions) {
            if (!allowed.contains(permission)) {
                throw new IllegalArgumentException("Unsupported datasource key permission: " + permission);
            }
        }
        return permissions;
    }

    private String generateKey() {
        byte[] bytes = new byte[24];
        RANDOM.nextBytes(bytes);
        return "mds_" + HexFormat.of().formatHex(bytes);
    }

    private String toJson(List<String> permissions) {
        try {
            return MAPPER.writeValueAsString(permissions);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to serialize datasource key permissions", e);
        }
    }

    private List<String> fromJson(String permissionsJson) {
        try {
            return MAPPER.readValue(permissionsJson, new TypeReference<>() {});
        } catch (Exception e) {
            throw new IllegalStateException("Invalid datasource key permissions JSON");
        }
    }

    private DataSourceApiKeyVO toVO(DataSourceApiKey key) {
        DataSourceApiKeyVO vo = new DataSourceApiKeyVO();
        vo.setId(key.getId());
        vo.setDatasourceId(key.getDatasourceId());
        vo.setKeyName(key.getKeyName());
        vo.setPermissions(fromJson(key.getPermissionsJson()));
        vo.setStatus(key.getStatus());
        vo.setCreatedAt(key.getCreatedAt());
        vo.setUpdatedAt(key.getUpdatedAt());
        return vo;
    }
}
