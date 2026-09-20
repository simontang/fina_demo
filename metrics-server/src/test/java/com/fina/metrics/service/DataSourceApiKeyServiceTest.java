package com.fina.metrics.service;

import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.fina.metrics.dto.DataSourceApiKeyRequest;
import com.fina.metrics.dto.DataSourceApiKeyVO;
import com.fina.metrics.entity.DataSourceApiKey;
import com.fina.metrics.entity.DataSourceConfig;
import com.fina.metrics.mapper.DataSourceApiKeyMapper;
import com.fina.metrics.mapper.DataSourceConfigMapper;
import com.fina.metrics.service.impl.DataSourceApiKeyServiceImpl;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class DataSourceApiKeyServiceTest {

    private DataSourceApiKeyMapper keyMapper;
    private DataSourceConfigMapper datasourceMapper;
    private DataSourceApiKeyServiceImpl service;

    @BeforeEach
    void setUp() {
        keyMapper = mock(DataSourceApiKeyMapper.class);
        datasourceMapper = mock(DataSourceConfigMapper.class);
        service = new DataSourceApiKeyServiceImpl(keyMapper, datasourceMapper);
        when(datasourceMapper.selectOne(any(Wrapper.class))).thenReturn(datasource());
    }

    @Test
    void createsDatasourceBoundKeyWithGeneratedSecretReturnedOnce() {
        DataSourceApiKeyRequest request = new DataSourceApiKeyRequest();
        request.setKeyName("builder");
        request.setPermissions(List.of("meta_read", "query"));
        when(keyMapper.insert(any())).thenAnswer(invocation -> {
            DataSourceApiKey key = invocation.getArgument(0);
            key.setId(10L);
            return 1;
        });

        DataSourceApiKeyVO created = service.create(15L, request);

        assertThat(created.getDatasourceId()).isEqualTo(15L);
        assertThat(created.getKeyName()).isEqualTo("builder");
        assertThat(created.getPermissions()).containsExactly("META_READ", "QUERY");
        assertThat(created.getRawKey()).startsWith("mds_");
        verify(keyMapper).insert(argThat(key ->
                key.getDatasourceId().equals(15L)
                        && key.getKeyName().equals("builder")
                        && key.getPermissionsJson().equals("[\"META_READ\",\"QUERY\"]")
                        && key.getKeyHash() != null
                        && !key.getKeyHash().isBlank()));
    }

    @Test
    void callerProvidedSecretIsNotEchoedBack() {
        DataSourceApiKeyRequest request = new DataSourceApiKeyRequest();
        request.setKeyName("runtime");
        request.setRawKey("known-secret");

        DataSourceApiKeyVO created = service.create(15L, request);

        assertThat(created.getRawKey()).isNull();
        assertThat(created.getPermissions()).containsExactly(
                "META_READ", "META_WRITE", "QUERY", "RUNTIME_QUERY");
    }

    @Test
    void rejectsUnsupportedPermission() {
        DataSourceApiKeyRequest request = new DataSourceApiKeyRequest();
        request.setKeyName("bad");
        request.setPermissions(List.of("ADMIN"));

        assertThatThrownBy(() -> service.create(15L, request))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("ADMIN");
    }

    @Test
    void disablesOnlyKeyUnderTheRequestedDatasource() {
        DataSourceApiKey existing = key(22L, 15L, "runtime", "[\"RUNTIME_QUERY\"]", 1);
        when(keyMapper.selectOne(any(Wrapper.class))).thenReturn(existing);

        DataSourceApiKeyVO disabled = service.disable(15L, 22L);

        assertThat(disabled.getStatus()).isZero();
        verify(keyMapper).updateById(argThat(key -> key.getId().equals(22L) && key.getStatus() == 0));
    }

    private DataSourceConfig datasource() {
        DataSourceConfig datasource = new DataSourceConfig();
        datasource.setId(15L);
        datasource.setName("Hankel");
        return datasource;
    }

    private DataSourceApiKey key(Long id, Long datasourceId, String name, String permissions, Integer status) {
        DataSourceApiKey key = new DataSourceApiKey();
        key.setId(id);
        key.setDatasourceId(datasourceId);
        key.setKeyName(name);
        key.setPermissionsJson(permissions);
        key.setStatus(status);
        return key;
    }
}
