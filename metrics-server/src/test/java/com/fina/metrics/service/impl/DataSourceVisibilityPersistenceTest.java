package com.fina.metrics.service.impl;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.fina.metrics.config.DynamicDataSourceManager;
import com.fina.metrics.dto.DataSourceRequest;
import com.fina.metrics.dto.DataSourceUpdateRequest;
import com.fina.metrics.dto.DataSourceVO;
import com.fina.metrics.entity.DataSourceConfig;
import com.fina.metrics.mapper.DataSourceConfigMapper;
import com.fina.metrics.service.RuntimeMetaCache;
import com.fina.metrics.util.EncryptUtil;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.validation.Validation;
import jakarta.validation.ValidatorFactory;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class DataSourceVisibilityPersistenceTest {

    private static final long ID = 42L;
    private static final String KEY = "fina-metrics-2024!";
    private DataSourceConfigMapper mapper;
    private DynamicDataSourceManager manager;
    private RuntimeMetaCache cache;
    private DataSourceServiceImpl service;

    @BeforeAll
    static void initializeMybatisMetadata() {
        TableInfoHelper.initTableInfo(
                new MapperBuilderAssistant(new MybatisConfiguration(), "visibility-test"),
                DataSourceConfig.class);
    }

    @BeforeEach
    void setUp() {
        mapper = mock(DataSourceConfigMapper.class);
        manager = mock(DynamicDataSourceManager.class);
        cache = mock(RuntimeMetaCache.class);
        service = new DataSourceServiceImpl(mapper, manager, cache);
        ReflectionTestUtils.setField(service, "encryptKey", KEY);
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"ALL", "RESTRICTED"})
    void createsPersistAndReturnMode(String requestedMode) {
        DataSourceRequest request = createRequest();
        request.setVisibleScopeMode(requestedMode);
        when(mapper.insert(any(DataSourceConfig.class))).thenAnswer(call -> {
            DataSourceConfig config = call.getArgument(0);
            config.setId(ID);
            assertThat(config.getVisibleScopeMode()).isEqualTo(requestedMode == null ? "RESTRICTED" : requestedMode);
            assertThat(EncryptUtil.decrypt(config.getPassword(), KEY)).isEqualTo("secret");
            return 1;
        });

        DataSourceVO result = service.create(request);

        assertThat(result.getVisibleScopeMode()).isEqualTo(requestedMode == null ? "RESTRICTED" : requestedMode);
        assertThat(result.getSourceType()).isEqualTo("cdp_postgres");
        verify(cache).invalidateDatasourceAfterCommit(ID, "datasource created");
    }

    @Test
    void omittedJsonModeDefaultsOnlyOnCreate() throws Exception {
        ObjectMapper json = new ObjectMapper();
        assertThat(json.readValue("{}", DataSourceRequest.class).getVisibleScopeMode()).isEqualTo("RESTRICTED");
        assertThat(json.readValue("{}", DataSourceUpdateRequest.class).getVisibleScopeMode()).isNull();
    }

    @ParameterizedTest
    @ValueSource(strings = {"ALL", "RESTRICTED"})
    void omittedUpdatePreservesExistingModeAndPassword(String existingMode) {
        DataSourceConfig existing = existingConfig(existingMode);
        when(mapper.selectOne(any())).thenReturn(existing);
        DataSourceUpdateRequest request = updateRequest();

        DataSourceVO result = service.update(ID, request);

        assertThat(result.getVisibleScopeMode()).isEqualTo(existingMode);
        assertThat(existing.getPassword()).isEqualTo("existing-encrypted-password");
        verify(mapper).updateById(existing);
        verify(manager).registerDataSource(existing);
        verify(cache).invalidateDatasourceAfterCommit(ID, "datasource updated");
    }

    @ParameterizedTest
    @ValueSource(strings = {"ALL", "RESTRICTED"})
    void explicitUpdateReplacesMode(String requestedMode) {
        DataSourceConfig existing = existingConfig("ALL".equals(requestedMode) ? "RESTRICTED" : "ALL");
        when(mapper.selectOne(any())).thenReturn(existing);
        DataSourceUpdateRequest request = updateRequest();
        request.setVisibleScopeMode(requestedMode);

        assertThat(service.update(ID, request).getVisibleScopeMode()).isEqualTo(requestedMode);

        assertThat(existing.getVisibleScopeMode()).isEqualTo(requestedMode);
        verify(mapper).updateById(existing);
        verify(cache).invalidateDatasourceAfterCommit(ID, "datasource updated");
    }

    @ParameterizedTest
    @ValueSource(strings = {"", " ", "all", "restricted", "ALL ", "NONE"})
    void invalidModesAreRejectedByServiceAndBeanValidation(String invalidMode) {
        DataSourceRequest create = createRequest();
        create.setVisibleScopeMode(invalidMode);
        DataSourceUpdateRequest update = updateRequest();
        update.setVisibleScopeMode(invalidMode);

        assertThatThrownBy(() -> service.create(create)).isInstanceOf(IllegalArgumentException.class)
                .hasMessage("visibleScopeMode must be ALL or RESTRICTED");
        assertThatThrownBy(() -> service.update(ID, update)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> service.testConnection(create)).isInstanceOf(IllegalArgumentException.class);
        verifyNoInteractions(mapper, manager, cache);
        try (ValidatorFactory factory = Validation.buildDefaultValidatorFactory()) {
            assertThat(factory.getValidator().validate(create))
                    .anyMatch(violation -> violation.getPropertyPath().toString().equals("visibleScopeMode"));
            assertThat(factory.getValidator().validate(update))
                    .anyMatch(violation -> violation.getPropertyPath().toString().equals("visibleScopeMode"));
        }
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"ALL", "RESTRICTED"})
    void validAndOmittedModesPassBeanValidation(String mode) {
        DataSourceRequest create = createRequest();
        create.setVisibleScopeMode(mode);
        DataSourceUpdateRequest update = updateRequest();
        update.setVisibleScopeMode(mode);
        try (ValidatorFactory factory = Validation.buildDefaultValidatorFactory()) {
            assertThat(factory.getValidator().validate(create)).isEmpty();
            assertThat(factory.getValidator().validate(update)).isEmpty();
        }
    }

    @Test
    void readsExposePersistedMode() {
        DataSourceConfig existing = existingConfig("ALL");
        when(mapper.selectOne(any())).thenReturn(existing);
        when(mapper.selectList(any())).thenReturn(List.of(existing));

        assertThat(service.getById(ID).getVisibleScopeMode()).isEqualTo("ALL");
        assertThat(service.listAll()).extracting(DataSourceVO::getVisibleScopeMode).containsExactly("ALL");
        assertThat(service.listActive()).extracting(DataSourceVO::getVisibleScopeMode).containsExactly("ALL");
        verifyNoInteractions(cache);
    }

    @Test
    void deletionAndReloadInvalidateDatasource() {
        DataSourceConfig existing = existingConfig("ALL");
        when(mapper.selectOne(any())).thenReturn(existing);

        service.reload(ID);
        verify(manager).registerDataSource(existing);
        verify(cache).invalidateDatasourceAfterCommit(ID, "datasource reloaded");

        service.delete(ID);
        verify(mapper).deleteById(ID);
        verify(manager).removeDataSource(ID);
        verify(cache).invalidateDatasourceAfterCommit(ID, "datasource deleted");
    }

    @Test
    void statusChangesPreserveModeAndInvalidateDatasource() {
        DataSourceConfig existing = existingConfig("ALL");
        when(mapper.selectOne(any())).thenReturn(existing);

        assertThat(service.disable(ID).getVisibleScopeMode()).isEqualTo("ALL");
        assertThat(existing.getStatus()).isZero();
        verify(manager).removeDataSource(ID);
        assertThat(service.enable(ID).getVisibleScopeMode()).isEqualTo("ALL");
        assertThat(existing.getStatus()).isOne();
        verify(manager).registerDataSource(existing);
        verify(cache, times(2)).invalidateDatasourceAfterCommit(ID, "datasource status changed");
    }

    @Test
    void failedPersistenceDoesNotInvalidateCache() {
        when(mapper.insert(any(DataSourceConfig.class))).thenThrow(new IllegalStateException("write failed"));

        assertThatThrownBy(() -> service.create(createRequest())).isInstanceOf(IllegalStateException.class);

        verifyNoInteractions(cache, manager);
    }

    private DataSourceRequest createRequest() {
        DataSourceRequest request = new DataSourceRequest();
        request.setName("Example");
        request.setUrl("jdbc:postgresql://localhost:5432/example");
        request.setUsername("example");
        request.setPassword("secret");
        request.setStatus(1);
        return request;
    }

    private DataSourceUpdateRequest updateRequest() {
        DataSourceUpdateRequest request = new DataSourceUpdateRequest();
        request.setName("Updated");
        request.setUrl("jdbc:postgresql://localhost:5432/example");
        request.setUsername("example");
        request.setStatus(1);
        return request;
    }

    private DataSourceConfig existingConfig(String mode) {
        DataSourceConfig config = new DataSourceConfig();
        config.setId(ID);
        config.setStatus(1);
        config.setVisibleScopeMode(mode);
        config.setPassword("existing-encrypted-password");
        return config;
    }
}
