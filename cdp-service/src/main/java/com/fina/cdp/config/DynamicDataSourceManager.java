package com.fina.cdp.config;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fina.cdp.entity.DataSourceConfig;
import com.fina.cdp.mapper.DataSourceConfigMapper;
import com.fina.cdp.util.EncryptUtil;
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.DependsOn;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import java.sql.Connection;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
@Component
@DependsOn("masterSchemaInitializer")
@RequiredArgsConstructor
public class DynamicDataSourceManager {

    private final DataSourceConfigMapper configMapper;

    @Value("${cdp.encryption.key}")
    private String encryptKey;

    @Value("${cdp.skip-init:false}")
    private boolean skipInit;

    private final ConcurrentHashMap<Long, HikariDataSource> poolMap = new ConcurrentHashMap<>();

    @PostConstruct
    public void init() {
        if (skipInit) {
            log.warn("CDP DynamicDataSourceManager: skip-init=true, no dynamic datasources loaded");
            return;
        }
        try {
            List<DataSourceConfig> configs = configMapper.selectList(
                    new LambdaQueryWrapper<DataSourceConfig>()
                            .eq(DataSourceConfig::getStatus, 1)
                            .eq(DataSourceConfig::getDeleted, 0)
                            .eq(DataSourceConfig::getSourceType, DataSourceType.CDP_POSTGRES.getCode())
            );
            configs.forEach(cfg -> {
                try {
                    registerDataSource(cfg);
                } catch (Exception e) {
                    log.warn("Failed to init CDP datasource id={} name={}: {}",
                            cfg.getId(), cfg.getName(), e.getMessage());
                }
            });
            log.info("Loaded {} CDP datasource(s)", poolMap.size());
        } catch (Exception e) {
            log.warn("Failed to load CDP datasources from master DB: {}", e.getMessage());
        }
    }

    public NamedParameterJdbcTemplate getNamedJdbcTemplate(Long datasourceId) {
        HikariDataSource ds = poolMap.get(datasourceId);
        if (ds == null || ds.isClosed()) {
            DataSourceConfig config = configMapper.selectById(datasourceId);
            if (config == null || Integer.valueOf(1).equals(config.getDeleted())
                    || !Integer.valueOf(1).equals(config.getStatus())) {
                throw new IllegalArgumentException("No active datasource found for id=" + datasourceId);
            }
            registerDataSource(config);
            ds = poolMap.get(datasourceId);
        }
        return new NamedParameterJdbcTemplate(ds);
    }

    public void registerDataSource(DataSourceConfig config) {
        closeExisting(config.getId());
        DataSourceType type = DataSourceType.resolve(config.getSourceType(), config.getUrl());

        HikariConfig hc = new HikariConfig();
        hc.setPoolName(type.getPoolNamePrefix() + "-" + config.getId());
        hc.setDriverClassName(type.getDriverClassName());
        hc.setJdbcUrl(config.getUrl());
        hc.setUsername(config.getUsername());
        hc.setPassword(decryptPassword(config.getPassword()));
        hc.setMaximumPoolSize(5);
        hc.setMinimumIdle(1);
        hc.setConnectionTimeout(30_000);
        hc.setIdleTimeout(600_000);
        hc.setMaxLifetime(1_800_000);
        hc.setInitializationFailTimeout(-1);

        String initSql = type.buildConnectionInitSql(config.getSchemaName());
        if (StringUtils.hasText(initSql)) {
            hc.setConnectionInitSql(initSql);
        }

        HikariDataSource ds = new HikariDataSource(hc);
        poolMap.put(config.getId(), ds);
        log.info("Registered CDP datasource id={} name={}", config.getId(), config.getName());
    }

    public boolean testConnection(DataSourceConfig config) {
        DataSourceType type = DataSourceType.resolve(config.getSourceType(), config.getUrl());
        HikariConfig hc = new HikariConfig();
        hc.setPoolName("cdp-postgres-test-" + System.currentTimeMillis());
        hc.setDriverClassName(type.getDriverClassName());
        hc.setJdbcUrl(config.getUrl());
        hc.setUsername(config.getUsername());
        hc.setPassword(decryptPassword(config.getPassword()));
        hc.setMaximumPoolSize(1);
        hc.setMinimumIdle(0);
        hc.setConnectionTimeout(10_000);
        hc.setInitializationFailTimeout(10_000);
        String initSql = type.buildConnectionInitSql(config.getSchemaName());
        if (StringUtils.hasText(initSql)) {
            hc.setConnectionInitSql(initSql);
        }
        try (HikariDataSource testDs = new HikariDataSource(hc);
             Connection conn = testDs.getConnection()) {
            return conn.isValid(5);
        } catch (Exception e) {
            log.warn("CDP connection test failed for datasource id={}: {}", config.getId(), e.getMessage());
            return false;
        }
    }

    @PreDestroy
    public void destroy() {
        poolMap.forEach((id, ds) -> {
            try {
                ds.close();
            } catch (Exception e) {
                log.warn("Error closing CDP datasource id={}", id, e);
            }
        });
        poolMap.clear();
    }

    private void closeExisting(Long datasourceId) {
        HikariDataSource existing = poolMap.remove(datasourceId);
        if (existing != null && !existing.isClosed()) {
            existing.close();
        }
    }

    private String decryptPassword(String encrypted) {
        if (encrypted == null || encrypted.isBlank()) {
            return "";
        }
        try {
            return EncryptUtil.decrypt(encrypted, encryptKey);
        } catch (Exception e) {
            log.warn("Datasource password decryption failed, using value as plain text");
            return encrypted;
        }
    }
}
