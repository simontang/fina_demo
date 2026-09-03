package com.fina.metrics.service.impl;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.metrics.dto.MetricsMetaObjectVO;
import com.fina.metrics.service.MetricsMetaObjectService;
import com.fina.metrics.service.MetricsMetaObjectTypes;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class MetaCatalogServiceImplTest {

    private static final long DATASOURCE_ID = 15L;

    private final ObjectMapper mapper = new ObjectMapper();
    private MetricsMetaObjectService metaObjectService;
    private MetaCatalogServiceImpl service;

    @BeforeEach
    void setUp() {
        metaObjectService = mock(MetricsMetaObjectService.class);
        when(metaObjectService.listActiveForOverlay(anyString(), any())).thenReturn(List.of());
        service = new MetaCatalogServiceImpl(metaObjectService);
        service.init();
    }

    @Test
    void overlaysMetricIndexDetailAndCatalogConfigFromDatabase() throws Exception {
        when(metaObjectService.listActiveForOverlay(MetricsMetaObjectTypes.CATALOG_CONFIG, DATASOURCE_ID))
                .thenReturn(List.of(metaObject(1L, "catalog_config", "default", """
                        {"metric_catalog_version":"db-v1","domain_categories":["db_domain"]}
                        """)));
        when(metaObjectService.listActiveForOverlay(MetricsMetaObjectTypes.METRIC_INDEX, DATASOURCE_ID))
                .thenReturn(List.of(metaObject(2L, "metric_index", "db_sales", """
                        {"metric_name":"db_sales","display_name":"DB Sales","domain":"db_domain","short_desc":"From DB"}
                        """)));
        when(metaObjectService.listActiveForOverlay(MetricsMetaObjectTypes.METRIC_DETAIL, DATASOURCE_ID))
                .thenReturn(List.of(metaObject(3L, "metric_detail", "db_sales", """
                        {
                          "metric_name":"db_sales",
                          "display_name":"DB Sales",
                          "domain":"db_domain",
                          "description":"Database supplied metric detail",
                          "calculation":{"sql_expression":"SUM(amount)"},
                          "source":{"table_view":"db_sales_view"},
                          "supported_dimensions":[]
                        }
                        """)));

        assertThat(service.getCatalogVersion(DATASOURCE_ID)).isEqualTo("db-v1");
        assertThat(service.getDomainCategories(DATASOURCE_ID)).containsExactly("db_domain");
        assertThat(service.getIndexItems(DATASOURCE_ID))
                .anySatisfy(item -> assertThat(item.path("metric_name").asText()).isEqualTo("db_sales"));
        assertThat(service.findDetailItem("db_sales", DATASOURCE_ID))
                .get()
                .extracting(item -> item.path("description").asText())
                .isEqualTo("Database supplied metric detail");
    }

    @Test
    void usesObjectKeyWhenMetricPayloadOmitsMetricName() throws Exception {
        when(metaObjectService.listActiveForOverlay(MetricsMetaObjectTypes.METRIC_INDEX, DATASOURCE_ID))
                .thenReturn(List.of(metaObject(4L, "metric_index", "key_metric", """
                        {"display_name":"Key Metric"}
                        """)));

        assertThat(service.getIndexItems(DATASOURCE_ID))
                .anySatisfy(item -> assertThat(item.path("metric_name").asText()).isEqualTo("key_metric"));
    }

    @Test
    void skipsNonObjectMetricOverlayPayloads() throws Exception {
        when(metaObjectService.listActiveForOverlay(MetricsMetaObjectTypes.METRIC_INDEX, DATASOURCE_ID))
                .thenReturn(List.of(metaObject(5L, "metric_index", "bad_metric", "[]")));

        assertThat(service.getIndexItems(DATASOURCE_ID))
                .noneSatisfy(item -> assertThat(item.path("metric_name").asText()).isEqualTo("bad_metric"));
    }

    private MetricsMetaObjectVO metaObject(Long id, String type, String key, String payloadJson) throws Exception {
        MetricsMetaObjectVO object = new MetricsMetaObjectVO();
        object.setId(id);
        object.setObjectType(type);
        object.setObjectKey(key);
        object.setPayload(mapper.readTree(payloadJson));
        object.setStatus(1);
        return object;
    }
}
