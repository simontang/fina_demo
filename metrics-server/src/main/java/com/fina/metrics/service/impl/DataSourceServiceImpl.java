package com.fina.metrics.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fina.metrics.config.DynamicDataSourceManager;
import com.fina.metrics.config.DataSourceType;
import com.fina.metrics.dto.DataSourceRequest;
import com.fina.metrics.dto.DataSourceUpdateRequest;
import com.fina.metrics.dto.DataSourceVO;
import com.fina.metrics.entity.DataSourceConfig;
import com.fina.metrics.mapper.DataSourceConfigMapper;
import com.fina.metrics.service.DataSourceService;
import com.fina.metrics.util.EncryptUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class DataSourceServiceImpl implements DataSourceService {

    private final DataSourceConfigMapper configMapper;
    private final DynamicDataSourceManager dsManager;

    @Value("${metrics.encryption.key}")
    private String encryptKey;

    // ─── Query ────────────────────────────────────────────────────────────────

    @Override
    public List<DataSourceVO> listAll() {
        List<DataSourceConfig> list = configMapper.selectList(
                new LambdaQueryWrapper<DataSourceConfig>()
                        .eq(DataSourceConfig::getDeleted, 0)
                        .orderByAsc(DataSourceConfig::getId)
        );
        return list.stream().map(this::toVO).collect(Collectors.toList());
    }

    @Override
    public List<DataSourceVO> listActive() {
        List<DataSourceConfig> list = configMapper.selectList(
                new LambdaQueryWrapper<DataSourceConfig>()
                        .eq(DataSourceConfig::getStatus, 1)
                        .eq(DataSourceConfig::getDeleted, 0)
                        .orderByAsc(DataSourceConfig::getId)
        );
        return list.stream().map(this::toVO).collect(Collectors.toList());
    }

    @Override
    public DataSourceVO getById(Long id) {
        return toVO(requireConfig(id));
    }

    // ─── Create / Update / Delete ─────────────────────────────────────────────

    @Override
    @Transactional
    public DataSourceVO create(DataSourceRequest request) {
        DataSourceConfig config = new DataSourceConfig();
        BeanUtils.copyProperties(request, config);
        config.setSourceType(DataSourceType.resolve(request.getSourceType(), request.getUrl()).getCode());
        config.setPassword(EncryptUtil.encrypt(request.getPassword(), encryptKey));
        config.setDeleted(0);

        configMapper.insert(config);

        if (config.getStatus() == 1) {
            dsManager.registerDataSource(config);
        }

        log.info("Created datasource id={} name={}", config.getId(), config.getName());
        return toVO(config);
    }

    @Override
    @Transactional
    public DataSourceVO update(Long id, DataSourceUpdateRequest request) {
        DataSourceConfig config = requireConfig(id);

        config.setName(request.getName());
        config.setUrl(request.getUrl());
        config.setUsername(request.getUsername());
        config.setSchemaName(request.getSchemaName());
        config.setSourceType(DataSourceType.resolve(request.getSourceType(), request.getUrl()).getCode());
        config.setDescription(request.getDescription());
        config.setStatus(request.getStatus());

        // Only re-encrypt if the caller actually provided a new password
        if (StringUtils.hasText(request.getPassword())) {
            config.setPassword(EncryptUtil.encrypt(request.getPassword(), encryptKey));
        }

        configMapper.updateById(config);

        if (config.getStatus() == 1) {
            dsManager.registerDataSource(config);
        } else {
            dsManager.removeDataSource(id);
        }

        log.info("Updated datasource id={} name={}", id, config.getName());
        return toVO(config);
    }

    @Override
    @Transactional
    public void delete(Long id) {
        requireConfig(id);
        configMapper.deleteById(id);
        dsManager.removeDataSource(id);
        log.info("Deleted datasource id={}", id);
    }

    // ─── Status toggle ────────────────────────────────────────────────────────

    @Override
    @Transactional
    public DataSourceVO enable(Long id) {
        return setStatus(id, 1);
    }

    @Override
    @Transactional
    public DataSourceVO disable(Long id) {
        return setStatus(id, 0);
    }

    @Override
    @Transactional
    public DataSourceVO setStatus(Long id, Integer status) {
        DataSourceConfig config = requireConfig(id);
        config.setStatus(status);
        configMapper.updateById(config);

        if (status == 1) {
            dsManager.registerDataSource(config);
            log.info("Enabled datasource id={}", id);
        } else {
            dsManager.removeDataSource(id);
            log.info("Disabled datasource id={}", id);
        }

        return toVO(config);
    }

    // ─── Connection test ──────────────────────────────────────────────────────

    @Override
    public boolean testConnection(DataSourceRequest request) {
        DataSourceConfig config = new DataSourceConfig();
        BeanUtils.copyProperties(request, config);
        config.setSourceType(DataSourceType.resolve(request.getSourceType(), request.getUrl()).getCode());
        // Caller supplies plain-text password; encrypt so DynamicDataSourceManager
        // can decrypt it consistently (same path as persisted configs).
        config.setPassword(EncryptUtil.encrypt(request.getPassword(), encryptKey));
        return dsManager.testConnection(config);
    }

    @Override
    public Map<String, Object> testConnectionById(Long id) {
        DataSourceConfig config = requireConfig(id);
        // Config password is already encrypted in the DB — dsManager decrypts internally
        boolean ok = dsManager.testConnection(config);
        String message = ok
                ? "Connection successful"
                : "Connection failed — check host, port, credentials and network access";
        log.info("Connection test for id={} result={}", id, ok);
        return Map.of("connected", ok, "message", message, "datasourceId", id);
    }

    // ─── Reload & pool status ─────────────────────────────────────────────────

    @Override
    public void reload(Long id) {
        DataSourceConfig config = requireConfig(id);
        dsManager.registerDataSource(config);
        log.info("Reloaded datasource id={}", id);
    }

    @Override
    public Map<String, Object> getPoolStatus(Long id) {
        requireConfig(id);  // ensures the id exists
        return dsManager.getPoolStatus(id);
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    private DataSourceConfig requireConfig(Long id) {
        DataSourceConfig config = configMapper.selectOne(
                new LambdaQueryWrapper<DataSourceConfig>()
                        .eq(DataSourceConfig::getId, id)
                        .eq(DataSourceConfig::getDeleted, 0)
        );
        if (config == null) {
            throw new IllegalArgumentException("DataSource not found: id=" + id);
        }
        return config;
    }

    private DataSourceVO toVO(DataSourceConfig config) {
        DataSourceVO vo = new DataSourceVO();
        BeanUtils.copyProperties(config, vo);
        vo.setStatusLabel(config.getStatus() == 1 ? "active" : "inactive");
        vo.setConnected(dsManager.isRegistered(config.getId()));
        return vo;
    }
}
