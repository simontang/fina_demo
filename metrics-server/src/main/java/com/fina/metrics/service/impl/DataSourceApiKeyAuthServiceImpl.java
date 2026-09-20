package com.fina.metrics.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.metrics.entity.DataSourceApiKey;
import com.fina.metrics.exception.ForbiddenException;
import com.fina.metrics.exception.UnauthorizedException;
import com.fina.metrics.mapper.DataSourceApiKeyMapper;
import com.fina.metrics.service.DataSourceApiKeyAuthService;
import com.fina.metrics.service.DataSourceApiKeyPermission;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Set;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class DataSourceApiKeyAuthServiceImpl implements DataSourceApiKeyAuthService {

    private static final String BEARER_PREFIX = "Bearer ";
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final DataSourceApiKeyMapper mapper;

    @Override
    public AuthContext requirePermission(
            HttpServletRequest request,
            Long expectedDatasourceId,
            DataSourceApiKeyPermission permission) {
        DataSourceApiKey key = requireKey(request);
        if (!key.getDatasourceId().equals(expectedDatasourceId)) {
            throw new ForbiddenException("Datasource key is not authorized for datasourceId=" + expectedDatasourceId);
        }
        if (!permissions(key).contains(permission.name())) {
            throw new ForbiddenException("Datasource key lacks permission: " + permission.name());
        }
        return new AuthContext(key.getDatasourceId(), key.getKeyName());
    }

    @Override
    public AuthContext requireAnyDatasourceKey(HttpServletRequest request) {
        DataSourceApiKey key = requireKey(request);
        return new AuthContext(key.getDatasourceId(), key.getKeyName());
    }

    private DataSourceApiKey requireKey(HttpServletRequest request) {
        String rawKey = resolveProvidedKey(request);
        if (!StringUtils.hasText(rawKey)) {
            throw new UnauthorizedException("Metrics datasource API key is required");
        }
        DataSourceApiKey key = mapper.selectOne(new LambdaQueryWrapper<DataSourceApiKey>()
                .eq(DataSourceApiKey::getKeyHash, hash(rawKey.trim()))
                .eq(DataSourceApiKey::getStatus, 1)
                .eq(DataSourceApiKey::getDeleted, 0));
        if (key == null) {
            throw new UnauthorizedException("Metrics datasource API key is invalid");
        }
        return key;
    }

    private String resolveProvidedKey(HttpServletRequest request) {
        String direct = request.getHeader("X-Metrics-Datasource-Key");
        if (StringUtils.hasText(direct)) {
            return direct;
        }
        String authorization = request.getHeader("Authorization");
        if (StringUtils.hasText(authorization) && authorization.startsWith(BEARER_PREFIX)) {
            return authorization.substring(BEARER_PREFIX.length());
        }
        return null;
    }

    private Set<String> permissions(DataSourceApiKey key) {
        try {
            return MAPPER.readValue(key.getPermissionsJson(), new TypeReference<Set<String>>() {})
                    .stream()
                    .filter(StringUtils::hasText)
                    .map(value -> value.trim().toUpperCase())
                    .collect(Collectors.toSet());
        } catch (Exception e) {
            throw new IllegalStateException("Invalid datasource key permissions for key id=" + key.getId());
        }
    }

    public static String hash(String rawKey) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = digest.digest(rawKey.getBytes(StandardCharsets.UTF_8));
            StringBuilder out = new StringBuilder(bytes.length * 2);
            for (byte b : bytes) {
                out.append(String.format("%02x", b));
            }
            return out.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
    }
}
