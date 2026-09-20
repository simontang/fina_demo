package com.fina.metrics.service;

import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.fina.metrics.entity.DataSourceApiKey;
import com.fina.metrics.exception.ForbiddenException;
import com.fina.metrics.exception.UnauthorizedException;
import com.fina.metrics.mapper.DataSourceApiKeyMapper;
import com.fina.metrics.service.impl.DataSourceApiKeyAuthServiceImpl;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockHttpServletRequest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class DataSourceApiKeyAuthServiceTest {

    private DataSourceApiKeyMapper mapper;
    private DataSourceApiKeyAuthServiceImpl service;

    @BeforeEach
    void setUp() {
        mapper = mock(DataSourceApiKeyMapper.class);
        service = new DataSourceApiKeyAuthServiceImpl(mapper);
    }

    @Test
    void resolvesDatasourceKeyFromHeader() {
        when(mapper.selectOne(any(Wrapper.class))).thenReturn(key(15L, "default",
                "[\"META_READ\",\"META_WRITE\",\"QUERY\",\"RUNTIME_QUERY\"]"));
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("X-Metrics-Datasource-Key", "metrics-datasource-default-15");

        DataSourceApiKeyAuthService.AuthContext context =
                service.requirePermission(request, 15L, DataSourceApiKeyPermission.META_READ);

        assertThat(context.datasourceId()).isEqualTo(15L);
        assertThat(context.keyName()).isEqualTo("default");
    }

    @Test
    void acceptsBearerCompatibilityForDatasourceKey() {
        when(mapper.selectOne(any(Wrapper.class))).thenReturn(key(15L, "default", "[\"QUERY\"]"));
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("Authorization", "Bearer secret");

        assertThatCode(() -> service.requirePermission(request, 15L, DataSourceApiKeyPermission.QUERY))
                .doesNotThrowAnyException();
    }

    @Test
    void rejectsAccessToAnotherDatasource() {
        when(mapper.selectOne(any(Wrapper.class))).thenReturn(key(15L, "default", "[\"META_READ\"]"));
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("X-Metrics-Datasource-Key", "secret");

        assertThatThrownBy(() -> service.requirePermission(request, 16L, DataSourceApiKeyPermission.META_READ))
                .isInstanceOf(ForbiddenException.class)
                .hasMessageContaining("datasourceId=16");
    }

    @Test
    void rejectsMissingPermission() {
        when(mapper.selectOne(any(Wrapper.class))).thenReturn(key(15L, "default", "[\"META_READ\"]"));
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("X-Metrics-Datasource-Key", "secret");

        assertThatThrownBy(() -> service.requirePermission(request, 15L, DataSourceApiKeyPermission.META_WRITE))
                .isInstanceOf(ForbiddenException.class)
                .hasMessageContaining("META_WRITE");
    }

    @Test
    void rejectsMissingOrInvalidKey() {
        MockHttpServletRequest missing = new MockHttpServletRequest();
        assertThatThrownBy(() -> service.requireAnyDatasourceKey(missing))
                .isInstanceOf(UnauthorizedException.class);

        when(mapper.selectOne(any(Wrapper.class))).thenReturn(null);
        MockHttpServletRequest invalid = new MockHttpServletRequest();
        invalid.addHeader("X-Metrics-Datasource-Key", "wrong");
        assertThatThrownBy(() -> service.requireAnyDatasourceKey(invalid))
                .isInstanceOf(UnauthorizedException.class);
    }

    private DataSourceApiKey key(Long datasourceId, String name, String permissions) {
        DataSourceApiKey key = new DataSourceApiKey();
        key.setId(1L);
        key.setDatasourceId(datasourceId);
        key.setKeyName(name);
        key.setPermissionsJson(permissions);
        key.setStatus(1);
        return key;
    }
}
