package com.fina.metrics.config;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fina.metrics.entity.DataSourceConfig;
import com.fina.metrics.mapper.DataSourceConfigMapper;
import com.fina.metrics.util.EncryptUtil;
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import java.sql.Connection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manages a runtime pool of datasources loaded from t_datasource_config
 * (stored in the master PostgreSQL database).
 *
 * Each datasource gets its own HikariCP connection pool, keyed by datasource ID.
 * Use getNamedJdbcTemplate(id) to execute queries against a specific datasource.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class DynamicDataSourceManager {

    private final DataSourceConfigMapper configMapper;

    @Value("${metrics.encryption.key}")
    private String encryptKey;

    @Value("${metrics.skip-init:false}")
    private boolean skipInit;

    /** id → live HikariCP pool for each active datasource */
    private final ConcurrentHashMap<Long, HikariDataSource> poolMap = new ConcurrentHashMap<>();

    @PostConstruct
    public void init() {
        if (skipInit) {
            log.warn("DynamicDataSourceManager: skip-init=true, no dynamic datasources loaded");
            return;
        }
        log.info("Loading active dynamic datasources from master PostgreSQL DB...");
        try {
            List<DataSourceConfig> configs = configMapper.selectList(
                    new LambdaQueryWrapper<DataSourceConfig>()
                            .eq(DataSourceConfig::getStatus, 1)
                            .eq(DataSourceConfig::getDeleted, 0)
            );
            configs.forEach(cfg -> {
                try {
                    registerDataSource(cfg);
                } catch (Exception e) {
                    log.error("Failed to init datasource id={} name={}: {}", cfg.getId(), cfg.getName(), e.getMessage());
                }
            });
            log.info("Loaded {} dynamic datasource(s)", poolMap.size());
        } catch (Exception e) {
            log.error("Failed to load datasources from master DB: {}", e.getMessage());
        }
    }

    /**
     * Register (or replace) a datasource by its config.
     * Called on startup and whenever a datasource is created/updated via API.
     */
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
        hc.setInitializationFailTimeout(-1); // don't fail fast on startup
        applyVendorDefaults(hc, type);

        String initSql = type.buildConnectionInitSql(config.getSchemaName());
        if (StringUtils.hasText(initSql)) {
            hc.setConnectionInitSql(initSql);
        }

        HikariDataSource ds = new HikariDataSource(hc);
        poolMap.put(config.getId(), ds);
        log.info("Registered datasource id={} name={} sourceType={}",
                config.getId(), config.getName(), type.getCode());
    }

    /**
     * Remove and close the pool for the given datasource ID.
     */
    public void removeDataSource(Long datasourceId) {
        closeExisting(datasourceId);
        log.info("Removed datasource id={}", datasourceId);
    }

    /**
     * Test connectivity for a datasource config (without persisting it).
     * Returns true if a connection can be obtained within timeout.
     */
    public boolean testConnection(DataSourceConfig config) {
        DataSourceType type = DataSourceType.resolve(config.getSourceType(), config.getUrl());
        HikariConfig hc = new HikariConfig();
        hc.setPoolName(type.getTestPoolNamePrefix() + "-" + System.currentTimeMillis());
        hc.setDriverClassName(type.getDriverClassName());
        hc.setJdbcUrl(config.getUrl());
        hc.setUsername(config.getUsername());
        hc.setPassword(decryptPassword(config.getPassword()));
        hc.setMaximumPoolSize(1);
        hc.setMinimumIdle(0);
        hc.setConnectionTimeout(10_000);
        hc.setInitializationFailTimeout(10_000);
        applyVendorDefaults(hc, type);
        String initSql = type.buildConnectionInitSql(config.getSchemaName());
        if (StringUtils.hasText(initSql)) {
            hc.setConnectionInitSql(initSql);
        }

        try (HikariDataSource testDs = new HikariDataSource(hc);
             Connection conn = testDs.getConnection()) {
            return conn.isValid(5);
        } catch (Exception e) {
            log.warn("Connection test failed for url={}: {}", config.getUrl(), e.getMessage());
            return false;
        }
    }

    /**
     * Returns a NamedParameterJdbcTemplate for a specific datasource.
     * Supports :paramName syntax in SQL.
     */
    public NamedParameterJdbcTemplate getNamedJdbcTemplate(Long datasourceId) {
        HikariDataSource ds = poolMap.get(datasourceId);
        if (ds == null) {
            throw new IllegalArgumentException("No active datasource found for id=" + datasourceId
                    + ". Make sure the datasource is registered and status=1.");
        }
        return new NamedParameterJdbcTemplate(ds);
    }

    /**
     * Check whether a datasource pool is currently registered.
     */
    public boolean isRegistered(Long datasourceId) {
        return poolMap.containsKey(datasourceId);
    }

    /**
     * Return HikariCP pool metrics for a registered datasource.
     * Returns a map with {@code registered=false} when the pool is absent.
     */
    public Map<String, Object> getPoolStatus(Long datasourceId) {
        HikariDataSource ds = poolMap.get(datasourceId);
        Map<String, Object> result = new HashMap<>();
        if (ds == null || ds.isClosed()) {
            result.put("registered", false);
            return result;
        }
        var pool = ds.getHikariPoolMXBean();
        result.put("registered", true);
        result.put("poolName", ds.getPoolName());
        result.put("totalConnections", pool.getTotalConnections());
        result.put("activeConnections", pool.getActiveConnections());
        result.put("idleConnections", pool.getIdleConnections());
        result.put("pendingThreads", pool.getThreadsAwaitingConnection());
        return result;
    }

    /**
     * List all currently registered datasource IDs.
     */
    public Map<Long, HikariDataSource> getPoolMap() {
        return poolMap;
    }

    @PreDestroy
    public void destroy() {
        log.info("Closing all dynamic datasource pools...");
        poolMap.forEach((id, ds) -> {
            try {
                ds.close();
            } catch (Exception e) {
                log.warn("Error closing pool for datasource id={}", id, e);
            }
        });
        poolMap.clear();
    }

    private void closeExisting(Long datasourceId) {
        HikariDataSource existing = poolMap.remove(datasourceId);
        if (existing != null && !existing.isClosed()) {
            try {
                existing.close();
            } catch (Exception e) {
                log.warn("Error closing old pool for datasource id={}", datasourceId, e);
            }
        }
    }

    private String decryptPassword(String encrypted) {
        if (encrypted == null || encrypted.isBlank()) return "";
        try {
            return EncryptUtil.decrypt(encrypted, encryptKey);
        } catch (Exception e) {
            // Fallback: assume plain-text (migration scenario)
            log.warn("Password decryption failed, using as-is (plain text fallback)");
            return encrypted;
        }
    }

    private void applyVendorDefaults(HikariConfig hc, DataSourceType type) {
        String transactionIsolation = type.getTransactionIsolationName();
        if (StringUtils.hasText(transactionIsolation)) {
            hc.setTransactionIsolation(transactionIsolation);
        }
    }
}
