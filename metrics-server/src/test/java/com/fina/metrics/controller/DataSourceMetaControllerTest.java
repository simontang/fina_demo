package com.fina.metrics.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.metrics.dto.*;
import com.fina.metrics.service.DataSourceTableAccessService;
import com.fina.metrics.service.MetricsMetaObjectService;
import com.fina.metrics.service.MetricsMetaObjectTypes;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

class DataSourceMetaControllerTest {

    private static final long DATASOURCE_ID = 15L;

    private final ObjectMapper mapper = new ObjectMapper();
    private MetricsMetaObjectService metaObjectService;
    private DataSourceTableAccessService tableAccessService;
    private DataSourceMetaController controller;

    @BeforeEach
    void setUp() {
        metaObjectService = mock(MetricsMetaObjectService.class);
        tableAccessService = mock(DataSourceTableAccessService.class);
        controller = new DataSourceMetaController(metaObjectService, tableAccessService);
    }

    @Test
    void createTableMetaDefaultsToTableViewDetailAndCreatesExactGrant() throws Exception {
        when(metaObjectService.create(any())).thenReturn(metaObject("table_view_detail", "hankel_sales"));
        when(tableAccessService.listGrants("hankel", DATASOURCE_ID)).thenReturn(List.of());
        when(tableAccessService.createGrant(eq("hankel"), eq(DATASOURCE_ID), any()))
                .thenReturn(tableGrant("hankel_sales"));
        DataSourcePublishedMetaRequest request = new DataSourcePublishedMetaRequest();
        request.setPayload(mapper.readTree("""
                {"schemaName":"public","tableName":"hankel_sales","displayName":"Hankel Sales"}
                """));

        ApiResponse<DataSourcePublishedMetaVO> response =
                controller.createTableMeta(DATASOURCE_ID, "hankel", request);

        assertThat(response.getCode()).isEqualTo(200);
        assertThat(response.getData().getMetaObject().getObjectKey()).isEqualTo("hankel_sales");
        assertThat(response.getData().getTableGrant().getTablePattern()).isEqualTo("hankel_sales");

        ArgumentCaptor<MetricsMetaObjectRequest> metaCaptor =
                ArgumentCaptor.forClass(MetricsMetaObjectRequest.class);
        verify(metaObjectService).create(metaCaptor.capture());
        assertThat(metaCaptor.getValue().getDatasourceId()).isEqualTo(DATASOURCE_ID);
        assertThat(metaCaptor.getValue().getObjectType()).isEqualTo(MetricsMetaObjectTypes.TABLE_VIEW_DETAIL);
        assertThat(metaCaptor.getValue().getObjectKey()).isEqualTo("hankel_sales");
        assertThat(metaCaptor.getValue().getStatus()).isEqualTo(1);

        ArgumentCaptor<DataSourceTableGrantRequest> grantCaptor =
                ArgumentCaptor.forClass(DataSourceTableGrantRequest.class);
        verify(tableAccessService).createGrant(eq("hankel"), eq(DATASOURCE_ID), grantCaptor.capture());
        assertThat(grantCaptor.getValue().getSchemaName()).isEqualTo("public");
        assertThat(grantCaptor.getValue().getTablePattern()).isEqualTo("hankel_sales");
        assertThat(grantCaptor.getValue().getPatternType()).isEqualTo("EXACT");
    }

    @Test
    void createMetricMetaDoesNotTouchTableGrants() throws Exception {
        when(metaObjectService.create(any())).thenReturn(metaObject("metric_detail", "hankel_sales_amount"));
        DataSourcePublishedMetaRequest request = new DataSourcePublishedMetaRequest();
        request.setPayload(mapper.readTree("""
                {"metric_name":"hankel_sales_amount","source":{"table_view":"hankel_sales"}}
                """));

        ApiResponse<MetricsMetaObjectVO> response = controller.createMetricMeta(DATASOURCE_ID, request);

        assertThat(response.getCode()).isEqualTo(200);
        assertThat(response.getData().getObjectKey()).isEqualTo("hankel_sales_amount");
        verify(metaObjectService).create(any(MetricsMetaObjectRequest.class));
        verifyNoInteractions(tableAccessService);
    }

    private MetricsMetaObjectVO metaObject(String type, String key) {
        MetricsMetaObjectVO object = new MetricsMetaObjectVO();
        object.setId(1L);
        object.setDatasourceId(DATASOURCE_ID);
        object.setObjectType(type);
        object.setObjectKey(key);
        object.setStatus(1);
        return object;
    }

    private DataSourceTableGrantVO tableGrant(String pattern) {
        DataSourceTableGrantVO grant = new DataSourceTableGrantVO();
        grant.setId(1L);
        grant.setTenantId("hankel");
        grant.setDatasourceId(DATASOURCE_ID);
        grant.setSchemaName("public");
        grant.setTablePattern(pattern);
        grant.setPatternType("EXACT");
        grant.setCaseSensitive(false);
        grant.setStatus(1);
        return grant;
    }
}
