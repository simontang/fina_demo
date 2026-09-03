package com.fina.metrics.service.impl;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.metrics.dto.MetricsMetaObjectVO;
import com.fina.metrics.dto.TableViewDetailResponse;
import com.fina.metrics.dto.TableViewIndexItem;
import com.fina.metrics.service.MetricsMetaObjectService;
import com.fina.metrics.service.MetricsMetaObjectTypes;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class TableViewMetaServiceImplTest {

    private static final long DATASOURCE_ID = 15L;

    private final ObjectMapper mapper = new ObjectMapper();
    private MetricsMetaObjectService metaObjectService;
    private TableViewMetaServiceImpl service;

    @BeforeEach
    void setUp() {
        metaObjectService = mock(MetricsMetaObjectService.class);
        when(metaObjectService.listActiveForOverlay(anyString(), any())).thenReturn(List.of());
        service = new TableViewMetaServiceImpl(
                new PathMatchingResourcePatternResolver(),
                metaObjectService);
        service.init();
    }

    @Test
    void overlaysTableCatalogAndViewDetailFromDatabase() throws Exception {
        when(metaObjectService.listActiveForOverlay(MetricsMetaObjectTypes.TABLE_CATALOG, DATASOURCE_ID))
                .thenReturn(List.of(metaObject(1L, "table_catalog", "db_sales_view", """
                        {
                          "tableName":"db_sales_view",
                          "docType":"销售汇总",
                          "docTypeEn":"Sales Summary",
                          "shortDesc":"DB supplied table catalog"
                        }
                        """)));
        when(metaObjectService.listActiveForOverlay(MetricsMetaObjectTypes.TABLE_VIEW_DETAIL, DATASOURCE_ID))
                .thenReturn(List.of(metaObject(2L, "table_view_detail", "db_sales_view", """
                        {
                          "viewName":"db_sales_view",
                          "mainTable":"ORDR",
                          "lineTable":"RDR1",
                          "selectSql":"SELECT * FROM db_sales_view",
                          "columns":[
                            {"name":"amount","label":"Amount","type":"numeric","example":"100.00"}
                          ]
                        }
                        """)));

        assertThat(service.getTableViewsIndex(DATASOURCE_ID))
                .anySatisfy(item -> {
                    assertThat(item.getTableName()).isEqualTo("db_sales_view");
                    assertThat(item.getDisplayName()).isEqualTo("Sales Summary");
                    assertThat(item.getColumnCount()).isEqualTo(1);
                });

        TableViewDetailResponse detail = service.getTableViewsDetails(DATASOURCE_ID).stream()
                .filter(item -> "db_sales_view".equals(item.getTableName()))
                .findFirst()
                .orElseThrow();
        assertThat(detail.getSelectSql()).isEqualTo("SELECT * FROM db_sales_view");
        assertThat(detail.getColumns()).extracting(TableViewDetailResponse.ColumnMeta::getName)
                .containsExactly("amount");
    }

    @Test
    void keepsStaticTablesWhenNoDatabaseOverlayExists() {
        List<TableViewIndexItem> tables = service.getTableViewsIndex(DATASOURCE_ID);

        assertThat(tables).isNotEmpty();
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
