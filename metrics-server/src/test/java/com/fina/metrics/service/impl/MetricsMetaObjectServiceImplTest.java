package com.fina.metrics.service.impl;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.metrics.dto.MetricsMetaObjectRequest;
import com.fina.metrics.dto.MetricsMetaObjectVO;
import com.fina.metrics.dto.PageResult;
import com.fina.metrics.entity.MetricsMetaObject;
import com.fina.metrics.mapper.MetricsMetaObjectMapper;
import com.fina.metrics.service.MetricsMetaObjectTypes;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class MetricsMetaObjectServiceImplTest {

    private final ObjectMapper objectMapper = new ObjectMapper();
    private MetricsMetaObjectMapper mapper;
    private MetricsMetaObjectServiceImpl service;

    @BeforeEach
    void setUp() {
        mapper = mock(MetricsMetaObjectMapper.class);
        service = new MetricsMetaObjectServiceImpl(mapper);
    }

    @Test
    void createStoresPayloadAsJsonText() throws Exception {
        MetricsMetaObjectRequest request = request("metric_index", "new_metric", """
                {"metric_name":"new_metric","display_name":"New Metric"}
                """);

        service.create(request);

        ArgumentCaptor<MetricsMetaObject> captor = ArgumentCaptor.forClass(MetricsMetaObject.class);
        verify(mapper).insert(captor.capture());
        MetricsMetaObject inserted = captor.getValue();
        assertThat(inserted.getDatasourceId()).isEqualTo(15L);
        assertThat(inserted.getObjectType()).isEqualTo("metric_index");
        assertThat(inserted.getObjectKey()).isEqualTo("new_metric");
        assertThat(inserted.getDeleted()).isZero();
        assertThat(objectMapper.readTree(inserted.getPayloadJson()).path("display_name").asText())
                .isEqualTo("New Metric");
    }

    @Test
    void rejectsUnsupportedTypeAndScalarPayload() throws Exception {
        MetricsMetaObjectRequest badType = request("bad_type", "x", "{}");
        assertThatThrownBy(() -> service.create(badType))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Unsupported");

        MetricsMetaObjectRequest scalarPayload = request(MetricsMetaObjectTypes.METRIC_INDEX, "x", "\"text\"");
        assertThatThrownBy(() -> service.create(scalarPayload))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("object or array");
    }

    @Test
    void listAppliesInMemoryPagination() {
        when(mapper.selectList(any())).thenReturn(List.of(
                entity(1L, "metric_index", "a", "{}"),
                entity(2L, "metric_index", "b", "{}"),
                entity(3L, "metric_index", "c", "{}")));

        PageResult<MetricsMetaObjectVO> page = service.list(null, "metric_index", null, 2, 2);

        assertThat(page.getTotal()).isEqualTo(3);
        assertThat(page.getItems()).extracting(MetricsMetaObjectVO::getObjectKey)
                .containsExactly("c");
    }

    @Test
    void overlayReturnsGlobalThenDatasourceScopedObjects() {
        when(mapper.selectList(any()))
                .thenReturn(List.of(entity(1L, "metric_index", "global", "{}")))
                .thenReturn(List.of(entity(2L, "metric_index", "scoped", "{}")));

        List<MetricsMetaObjectVO> objects = service.listActiveForOverlay("metric_index", 15L);

        assertThat(objects).extracting(MetricsMetaObjectVO::getObjectKey)
                .containsExactly("global", "scoped");
    }

    private MetricsMetaObjectRequest request(String type, String key, String payloadJson) throws Exception {
        MetricsMetaObjectRequest request = new MetricsMetaObjectRequest();
        request.setDatasourceId(15L);
        request.setObjectType(type);
        request.setObjectKey(key);
        request.setPayload(objectMapper.readTree(payloadJson));
        request.setStatus(1);
        return request;
    }

    private MetricsMetaObject entity(Long id, String type, String key, String payloadJson) {
        MetricsMetaObject entity = new MetricsMetaObject();
        entity.setId(id);
        entity.setDatasourceId(15L);
        entity.setObjectType(type);
        entity.setObjectKey(key);
        entity.setPayloadJson(payloadJson);
        entity.setStatus(1);
        entity.setDeleted(0);
        return entity;
    }
}
