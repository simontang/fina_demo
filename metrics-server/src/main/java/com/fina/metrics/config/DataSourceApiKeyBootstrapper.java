package com.fina.metrics.config;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.metrics.entity.DataSourceApiKey;
import com.fina.metrics.entity.DataSourceConfig;
import com.fina.metrics.mapper.DataSourceApiKeyMapper;
import com.fina.metrics.mapper.DataSourceConfigMapper;
import com.fina.metrics.service.DataSourceApiKeyPermission;
import com.fina.metrics.service.impl.DataSourceApiKeyAuthServiceImpl;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.List;

@Slf4j
@Component
@DependsOn("masterSchemaInitializer")
@RequiredArgsConstructor
public class DataSourceApiKeyBootstrapper {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final DataSourceConfigMapper datasourceMapper;
    private final DataSourceApiKeyMapper keyMapper;

    @Value("${metrics.datasource-api-key.bootstrap.enabled:true}")
    private boolean enabled;

    @Value("${metrics.datasource-api-key.bootstrap.default-key-name:default}")
    private String defaultKeyName;

    @Value("${metrics.datasource-api-key.bootstrap.default-key-prefix:metrics-datasource-default-}")
    private String defaultKeyPrefix;

    @PostConstruct
    public void bootstrap() {
        if (!enabled) {
            log.info("Datasource API key bootstrap disabled");
            return;
        }
        List<DataSourceConfig> datasources = datasourceMapper.selectList(
                new LambdaQueryWrapper<DataSourceConfig>().eq(DataSourceConfig::getDeleted, 0));
        for (DataSourceConfig datasource : datasources) {
            ensureDefaultKey(datasource.getId());
        }
    }

    private void ensureDefaultKey(Long datasourceId) {
        Long existing = keyMapper.selectCount(new LambdaQueryWrapper<DataSourceApiKey>()
                .eq(DataSourceApiKey::getDatasourceId, datasourceId)
                .eq(DataSourceApiKey::getKeyName, defaultKeyName)
                .eq(DataSourceApiKey::getDeleted, 0));
        if (existing != null && existing > 0) {
            return;
        }
        DataSourceApiKey key = new DataSourceApiKey();
        key.setDatasourceId(datasourceId);
        key.setKeyName(defaultKeyName);
        key.setKeyHash(DataSourceApiKeyAuthServiceImpl.hash(defaultKeyPrefix + datasourceId));
        key.setPermissionsJson(defaultPermissionsJson());
        key.setStatus(1);
        key.setDeleted(0);
        keyMapper.insert(key);
        log.info("Created default datasource API key datasourceId={} keyName={}", datasourceId, defaultKeyName);
    }

    private String defaultPermissionsJson() {
        try {
            return MAPPER.writeValueAsString(Arrays.stream(DataSourceApiKeyPermission.values())
                    .map(Enum::name)
                    .toList());
        } catch (Exception e) {
            throw new IllegalStateException("Failed to serialize datasource key permissions", e);
        }
    }
}
