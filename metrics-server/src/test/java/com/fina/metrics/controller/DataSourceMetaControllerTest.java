package com.fina.metrics.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fina.metrics.dto.*;
import com.fina.metrics.exception.ForbiddenException;
import com.fina.metrics.exception.GlobalExceptionHandler;
import com.fina.metrics.service.DataSourceApiKeyAuthService;
import com.fina.metrics.service.DataSourceTableAccessService;
import com.fina.metrics.service.MetricsMetaObjectService;
import com.fina.metrics.service.MetricsMetaObjectTypes;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;
import org.mockito.InOrder;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class DataSourceMetaControllerTest {

    private static final long DATASOURCE_ID = 15L;
    private static final String TABLE_NAME = "hankel_sales";

    private final ObjectMapper mapper = new ObjectMapper();
    private MetricsMetaObjectService metaObjectService;
    private DataSourceTableAccessService tableAccessService;
    private DataSourceApiKeyAuthService datasourceAuth;
    private DataSourceMetaController controller;

    @BeforeEach
    void setUp() {
        metaObjectService = mock(MetricsMetaObjectService.class);
        tableAccessService = mock(DataSourceTableAccessService.class);
        datasourceAuth = mock(DataSourceApiKeyAuthService.class);
        controller = new DataSourceMetaController(metaObjectService, tableAccessService, datasourceAuth);
    }

    @AfterEach
    void metadataNeverMutatesScope() {
        verify(tableAccessService, never()).createGrant(any(), any(), any());
        verify(tableAccessService, never()).updateGrant(any(), any(), any(), any());
        verify(tableAccessService, never()).deleteGrant(any(), any(), any());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void publishesAuthorizedTableMetaWithoutChangingScope(boolean update) throws Exception {
        MetricsMetaObjectVO stored = metaObject(MetricsMetaObjectTypes.TABLE_VIEW_DETAIL, TABLE_NAME);
        when(metaObjectService.create(any())).thenReturn(stored);
        when(metaObjectService.updateByDatasourceTypeKey(any(), any(), any(), any())).thenReturn(stored);
        authorize("public", TABLE_NAME);
        DataSourcePublishedMetaRequest request = tableRequest();

        ApiResponse<DataSourcePublishedMetaVO> response = publish(update, "hankel", request);

        assertThat(response.getCode()).isEqualTo(200);
        assertThat(response.getData().getMetaObject()).isSameAs(stored);
        assertThat(response.getData().getTableGrant()).isNull();
        ArgumentCaptor<MetricsMetaObjectRequest> metaCaptor =
                ArgumentCaptor.forClass(MetricsMetaObjectRequest.class);
        InOrder order = inOrder(tableAccessService, metaObjectService);
        order.verify(tableAccessService).isTableAuthorized(null, DATASOURCE_ID, "public", TABLE_NAME);
        if (update) {
            order.verify(metaObjectService).updateByDatasourceTypeKey(
                    eq(DATASOURCE_ID), eq(MetricsMetaObjectTypes.TABLE_VIEW_DETAIL), eq(TABLE_NAME), metaCaptor.capture());
        } else {
            order.verify(metaObjectService).create(metaCaptor.capture());
        }
        assertThat(metaCaptor.getValue().getDatasourceId()).isEqualTo(DATASOURCE_ID);
        assertThat(metaCaptor.getValue().getObjectType()).isEqualTo(MetricsMetaObjectTypes.TABLE_VIEW_DETAIL);
        assertThat(metaCaptor.getValue().getObjectKey()).isEqualTo(TABLE_NAME);
        assertThat(metaCaptor.getValue().getPayload()).isEqualTo(request.getPayload());
        assertThat(metaCaptor.getValue().getStatus()).isEqualTo(1);
        verifyNoMoreInteractions(tableAccessService);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void missingScopeFailsBeforeAnyMetaWrite(boolean update) throws Exception {
        DataSourcePublishedMetaRequest request = tableRequest();

        assertThatThrownBy(() -> publish(update, "hankel", request))
                .isInstanceOf(ForbiddenException.class).hasMessageContaining(TABLE_NAME);

        verify(tableAccessService).isTableAuthorized(null, DATASOURCE_ID, "public", TABLE_NAME);
        verifyNoInteractions(metaObjectService);
        verifyNoMoreInteractions(tableAccessService);
    }

    @ParameterizedTest
    @NullAndEmptySource
    @ValueSource(strings = {"hankel", "another-tenant", "  "})
    void tenantHeaderIsIgnoredForCreateAndUpdate(String tenantId) throws Exception {
        authorize("public", TABLE_NAME);

        controller.createTableMeta(DATASOURCE_ID, tenantId, tableRequest(), null);
        controller.updateTableMeta(DATASOURCE_ID, TABLE_NAME, tenantId, tableRequest(), null);

        verify(tableAccessService, times(2)).isTableAuthorized(null, DATASOURCE_ID, "public", TABLE_NAME);
        verifyNoMoreInteractions(tableAccessService);
    }

    @Test
    void tableCatalogRequiresAuthorizationToo() throws Exception {
        DataSourcePublishedMetaRequest request = tableRequest();
        request.setObjectType(MetricsMetaObjectTypes.TABLE_CATALOG);

        assertThatThrownBy(() -> controller.createTableMeta(DATASOURCE_ID, null, request, null))
                .isInstanceOf(ForbiddenException.class);

        verifyNoInteractions(metaObjectService);
    }

    @Test
    void objectKeyIsOnlyATableFallbackWhenPayloadHasNoTarget() throws Exception {
        DataSourcePublishedMetaRequest request = tableRequest();
        request.setObjectKey("stable-meta-key");
        authorize("public", TABLE_NAME);

        controller.createTableMeta(DATASOURCE_ID, null, request, null);

        verify(tableAccessService).isTableAuthorized(null, DATASOURCE_ID, "public", TABLE_NAME);
        verifyNoMoreInteractions(tableAccessService);
        ArgumentCaptor<MetricsMetaObjectRequest> captor = ArgumentCaptor.forClass(MetricsMetaObjectRequest.class);
        verify(metaObjectService).create(captor.capture());
        assertThat(captor.getValue().getObjectKey()).isEqualTo("stable-meta-key");
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void objectKeyFallbackStillRequiresAuthorization(boolean update) throws Exception {
        DataSourcePublishedMetaRequest request = new DataSourcePublishedMetaRequest();
        request.setObjectKey(TABLE_NAME);
        request.setPayload(mapper.readTree("{\"schemaName\":\"public\"}"));

        assertThatThrownBy(() -> publish(update, null, request)).isInstanceOf(ForbiddenException.class);

        verify(tableAccessService).isTableAuthorized(null, DATASOURCE_ID, "public", TABLE_NAME);
        verifyNoInteractions(metaObjectService);
    }

    @Test
    void viewNameCanBeTheAuthorizedTarget() throws Exception {
        DataSourcePublishedMetaRequest request = new DataSourcePublishedMetaRequest();
        request.setPayload(mapper.readTree("{\"schemaName\":\"public\",\"viewName\":\"sales_view\"}"));
        authorize("public", "sales_view");

        controller.createTableMeta(DATASOURCE_ID, null, request, null);

        verify(tableAccessService).isTableAuthorized(null, DATASOURCE_ID, "public", "sales_view");
        ArgumentCaptor<MetricsMetaObjectRequest> captor = ArgumentCaptor.forClass(MetricsMetaObjectRequest.class);
        verify(metaObjectService).create(captor.capture());
        assertThat(captor.getValue().getObjectKey()).isEqualTo("sales_view");
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void objectKeyFallbackRejectsSchemaConflictBeforeWriting(boolean update) throws Exception {
        DataSourcePublishedMetaRequest request = new DataSourcePublishedMetaRequest();
        request.setObjectKey("private.orders");
        request.setPayload(mapper.readTree("{\"schemaName\":\"public\"}"));

        assertThatThrownBy(() -> {
            if (update) {
                controller.updateTableMeta(DATASOURCE_ID, "private.orders", null, request, null);
            } else {
                controller.createTableMeta(DATASOURCE_ID, null, request, null);
            }
        }).isInstanceOf(IllegalArgumentException.class).hasMessageContaining("Conflicting table schema");

        verifyNoInteractions(tableAccessService, metaObjectService);
    }

    @ParameterizedTest
    @ValueSource(strings = {"viewName", "mainTable", "lineTable"})
    void everyExplicitTableReferenceMustBeAuthorizedBeforeCreateOrUpdate(String field) throws Exception {
        authorize("public", TABLE_NAME);
        DataSourcePublishedMetaRequest request = tableRequest();
        ((ObjectNode) request.getPayload()).put(field, "outside_scope");

        assertThatThrownBy(() -> publish(false, "hankel", request)).isInstanceOf(ForbiddenException.class);
        assertThatThrownBy(() -> publish(true, "other", request)).isInstanceOf(ForbiddenException.class);

        verify(tableAccessService, times(2)).isTableAuthorized(null, DATASOURCE_ID, "public", "outside_scope");
        verifyNoInteractions(metaObjectService);
    }

    @Test
    void qualifiedTableCannotHideOutsideSchemaBehindPayloadSchema() throws Exception {
        DataSourcePublishedMetaRequest request = tableRequest();
        ((ObjectNode) request.getPayload()).put("tableName", "\"private\".\"hankel_sales\"");
        authorize("public", TABLE_NAME);

        assertThatThrownBy(() -> controller.createTableMeta(DATASOURCE_ID, null, request, null))
                .isInstanceOf(IllegalArgumentException.class).hasMessageContaining("Conflicting table schema");

        verify(tableAccessService, never()).isTableAuthorized(any(), any(), any(), any());
        verifyNoInteractions(metaObjectService);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void viewSqlReferencesAreAuthorizedBeforeMetaWrites(boolean update) throws Exception {
        authorize("public", TABLE_NAME);
        DataSourcePublishedMetaRequest request = tableRequest();
        ((ObjectNode) request.getPayload()).put("selectSql", """
                WITH sales AS (SELECT * FROM public.hankel_sales)
                SELECT * FROM sales s JOIN private.payroll p ON p.id = s.id
                """);

        assertThatThrownBy(() -> publish(update, "hankel", request))
                .isInstanceOf(ForbiddenException.class).hasMessageContaining("payroll");

        verify(tableAccessService).isTableAuthorized(null, DATASOURCE_ID, "private", "payroll");
        verifyNoInteractions(metaObjectService);
    }

    @Test
    void authorizedViewChecksAllSourcesBeforeWriting() throws Exception {
        DataSourcePublishedMetaRequest request = tableRequest();
        ((ObjectNode) request.getPayload()).put("viewName", "sales_view")
                .put("mainTable", "hankel_orders").put("lineTable", "sales.hankel_items")
                .put("selectSql", "SELECT * FROM hankel_sales JOIN sales.hankel_items i ON i.id = hankel_sales.id");
        authorize("public", TABLE_NAME);
        authorize("public", "sales_view");
        authorize("public", "hankel_orders");
        authorize("sales", "hankel_items");
        authorize(null, TABLE_NAME);

        controller.createTableMeta(DATASOURCE_ID, null, request, null);

        InOrder order = inOrder(tableAccessService, metaObjectService);
        order.verify(tableAccessService).isTableAuthorized(null, DATASOURCE_ID, "public", TABLE_NAME);
        order.verify(tableAccessService).isTableAuthorized(null, DATASOURCE_ID, "public", "sales_view");
        order.verify(tableAccessService).isTableAuthorized(null, DATASOURCE_ID, "public", "hankel_orders");
        order.verify(tableAccessService).isTableAuthorized(null, DATASOURCE_ID, "sales", "hankel_items");
        order.verify(tableAccessService).isTableAuthorized(null, DATASOURCE_ID, null, TABLE_NAME);
        order.verify(metaObjectService).create(any());
        verifyNoMoreInteractions(tableAccessService);
    }

    @Test
    void tableAuthorizationDoesNotAuthorizeDdlInSelectSql() throws Exception {
        DataSourcePublishedMetaRequest request = tableRequest();
        ((ObjectNode) request.getPayload()).put("selectSql", "CREATE VIEW hankel_sales AS SELECT * FROM private.payroll");
        authorize("public", TABLE_NAME);

        assertThatThrownBy(() -> controller.createTableMeta(DATASOURCE_ID, null, request, null))
                .isInstanceOf(IllegalArgumentException.class);

        verifyNoInteractions(metaObjectService);
    }

    @Test
    void unqualifiedSqlSourcesUseDatasourceSchemaInsteadOfPayloadSchema() throws Exception {
        DataSourcePublishedMetaRequest request = tableRequest();
        ((ObjectNode) request.getPayload()).put("selectSql", "SELECT * FROM hankel_sales");
        authorize("public", TABLE_NAME);

        assertThatThrownBy(() -> controller.createTableMeta(DATASOURCE_ID, null, request, null))
                .isInstanceOf(ForbiddenException.class);

        verify(tableAccessService).isTableAuthorized(null, DATASOURCE_ID, null, TABLE_NAME);
        verifyNoInteractions(metaObjectService);
    }

    @Test
    void postTableMetaPreservesLegacyResponseShapeWithoutScopeMutation() throws Exception {
        MetricsMetaObjectVO stored = metaObject(MetricsMetaObjectTypes.TABLE_VIEW_DETAIL, TABLE_NAME);
        when(metaObjectService.create(any())).thenReturn(stored);
        authorize("public", TABLE_NAME);
        String response = MockMvcBuilders.standaloneSetup(controller).build()
                .perform(post("/api/v1/datasources/{dsId}/meta/tables", DATASOURCE_ID)
                        .header("X-Tenant-Id", "different-tenant")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(mapper.writeValueAsString(tableRequest())))
                .andExpect(status().isOk())
                .andReturn().getResponse().getContentAsString();

        assertThat(mapper.readTree(response).path("data").properties())
                .extracting(java.util.Map.Entry::getKey).containsExactlyInAnyOrder("metaObject", "tableGrant");
        assertThat(mapper.readTree(response).path("data").path("tableGrant").isNull()).isTrue();
        verify(tableAccessService).isTableAuthorized(null, DATASOURCE_ID, "public", TABLE_NAME);
        verifyNoMoreInteractions(tableAccessService);
    }

    @Test
    void postMetricCannotWidenScopeThroughLegacyGrant() throws Exception {
        when(tableAccessService.listActiveGrants(null, DATASOURCE_ID)).thenReturn(List.of(tableGrant(TABLE_NAME)));
        DataSourcePublishedMetaRequest request = metricRequest();
        request.setAccessGrant(grantRequest("hankel_", "PREFIX"));

        MockMvcBuilders.standaloneSetup(controller).setControllerAdvice(new GlobalExceptionHandler()).build()
                .perform(post("/api/v1/datasources/{dsId}/meta/metrics", DATASOURCE_ID)
                        .header("X-Tenant-Id", "different-tenant")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(mapper.writeValueAsString(request)))
                .andExpect(status().isForbidden())
                .andExpect(content().json("{\"code\":403}"));

        verify(tableAccessService).listActiveGrants(null, DATASOURCE_ID);
        verifyNoInteractions(metaObjectService);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void metricPublicationAcceptsIdenticalLegacyRuleWithoutScopeMutation(boolean update) throws Exception {
        when(tableAccessService.listActiveGrants(null, DATASOURCE_ID)).thenReturn(List.of(tableGrant(TABLE_NAME)));
        DataSourcePublishedMetaRequest request = metricRequest();
        request.setAccessGrant(grantRequest(TABLE_NAME, "EXACT"));

        if (update) {
            controller.updateMetricMeta(DATASOURCE_ID, "hankel_sales_amount", request, null);
            verify(metaObjectService).updateByDatasourceTypeKey(
                    eq(DATASOURCE_ID), eq(MetricsMetaObjectTypes.METRIC_DETAIL), eq("hankel_sales_amount"), any());
        } else {
            controller.createMetricMeta(DATASOURCE_ID, request, null);
            verify(metaObjectService).create(any());
        }

        verify(tableAccessService).listActiveGrants(null, DATASOURCE_ID);
        verifyNoMoreInteractions(tableAccessService);
    }

    @Test
    void metricUpdateCannotWidenScopeThroughLegacyGrant() throws Exception {
        when(tableAccessService.listActiveGrants(null, DATASOURCE_ID)).thenReturn(List.of(tableGrant(TABLE_NAME)));
        DataSourcePublishedMetaRequest request = metricRequest();
        request.setAccessGrant(grantRequest("hankel_", "PREFIX"));

        assertThatThrownBy(() -> controller.updateMetricMeta(DATASOURCE_ID, "hankel_sales_amount", request, null))
                .isInstanceOf(ForbiddenException.class);

        verifyNoInteractions(metaObjectService);
    }

    @ParameterizedTest
    @ValueSource(strings = {"EXACT", "PREFIX"})
    void legacyGrantMatchingAnActiveRuleIsValidationOnly(String patternType) throws Exception {
        authorize("public", TABLE_NAME);
        String pattern = patternType.equals("EXACT") ? TABLE_NAME : "hankel_";
        DataSourceTableGrantVO existing = tableGrant(pattern);
        existing.setPatternType(patternType);
        when(tableAccessService.listActiveGrants(null, DATASOURCE_ID)).thenReturn(List.of(existing));
        DataSourcePublishedMetaRequest request = tableRequest();
        DataSourceTableGrantRequest legacyGrant = grantRequest(pattern, patternType);
        request.setAccessGrant(legacyGrant);

        ApiResponse<DataSourcePublishedMetaVO> response = controller.createTableMeta(DATASOURCE_ID, "other", request, null);

        assertThat(response.getData().getTableGrant()).isNull();
        InOrder order = inOrder(tableAccessService, metaObjectService);
        order.verify(tableAccessService).isTableAuthorized(null, DATASOURCE_ID, "public", TABLE_NAME);
        order.verify(tableAccessService).listActiveGrants(null, DATASOURCE_ID);
        order.verify(metaObjectService).create(any());
        verifyNoMoreInteractions(tableAccessService);
        assertThat(request.getAccessGrant()).isSameAs(legacyGrant);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void legacyGrantCannotWidenScopeBeforeCreateOrUpdate(boolean update) throws Exception {
        authorize("public", TABLE_NAME);
        when(tableAccessService.listActiveGrants(null, DATASOURCE_ID)).thenReturn(List.of(tableGrant(TABLE_NAME)));
        DataSourcePublishedMetaRequest request = tableRequest();
        request.setAccessGrant(grantRequest("hankel_", "PREFIX"));

        assertThatThrownBy(() -> publish(update, "hankel", request))
                .isInstanceOf(ForbiddenException.class).hasMessageContaining("accessGrant");

        verifyNoInteractions(metaObjectService);
    }

    @ParameterizedTest
    @ValueSource(strings = {"schema", "case", "pattern", "status", "type"})
    void legacyGrantMustMatchAllActiveRuleSemantics(String difference) throws Exception {
        authorize("public", TABLE_NAME);
        when(tableAccessService.listActiveGrants(null, DATASOURCE_ID)).thenReturn(List.of(tableGrant(TABLE_NAME)));
        DataSourcePublishedMetaRequest request = tableRequest();
        DataSourceTableGrantRequest grant = grantRequest(TABLE_NAME, "EXACT");
        switch (difference) {
            case "schema" -> grant.setSchemaName(null);
            case "case" -> grant.setCaseSensitive(true);
            case "pattern" -> grant.setTablePattern("outside_scope");
            case "status" -> grant.setStatus(0);
            case "type" -> grant.setPatternType("PREFIX");
            default -> throw new AssertionError(difference);
        }
        request.setAccessGrant(grant);

        assertThatThrownBy(() -> controller.createTableMeta(DATASOURCE_ID, null, request, null))
                .isInstanceOf(ForbiddenException.class);

        verifyNoInteractions(metaObjectService);
    }

    @Test
    void inactiveGrantCannotBeReactivatedThroughMetaPublication() throws Exception {
        authorize("public", TABLE_NAME);
        DataSourceTableGrantVO inactive = tableGrant(TABLE_NAME);
        inactive.setStatus(0);
        when(tableAccessService.listGrants(null, DATASOURCE_ID)).thenReturn(List.of(inactive));
        when(tableAccessService.listActiveGrants(null, DATASOURCE_ID)).thenReturn(List.of());
        DataSourcePublishedMetaRequest request = tableRequest();
        request.setAccessGrant(grantRequest(TABLE_NAME, "EXACT"));

        assertThatThrownBy(() -> controller.createTableMeta(DATASOURCE_ID, null, request, null))
                .isInstanceOf(ForbiddenException.class);

        verify(tableAccessService).listActiveGrants(null, DATASOURCE_ID);
        verify(tableAccessService, never()).listGrants(any(), any());
        verifyNoInteractions(metaObjectService);
    }

    @Test
    void coveredLegacyGrantCannotAuthorizeOutsidePayload() throws Exception {
        DataSourcePublishedMetaRequest request = tableRequest();
        request.setAccessGrant(grantRequest(TABLE_NAME, "EXACT"));
        when(tableAccessService.listActiveGrants(null, DATASOURCE_ID)).thenReturn(List.of(tableGrant(TABLE_NAME)));

        assertThatThrownBy(() -> controller.createTableMeta(DATASOURCE_ID, null, request, null))
                .isInstanceOf(ForbiddenException.class);

        verifyNoInteractions(metaObjectService);
    }

    @Test
    void deleteTableMetaLeavesAllScopeRulesUntouched() throws Exception {
        MetricsMetaObjectVO catalog = metaObject(MetricsMetaObjectTypes.TABLE_CATALOG, TABLE_NAME);
        MetricsMetaObjectVO detail = metaObject(MetricsMetaObjectTypes.TABLE_VIEW_DETAIL, TABLE_NAME);
        detail.setId(2L);
        detail.setPayload(tableRequest().getPayload());
        when(metaObjectService.listByDatasourceAndTypes(
                DATASOURCE_ID, List.of(MetricsMetaObjectTypes.TABLE_CATALOG, MetricsMetaObjectTypes.TABLE_VIEW_DETAIL),
                TABLE_NAME, 1, 2))
                .thenReturn(PageResult.<MetricsMetaObjectVO>builder().items(List.of(catalog, detail)).build());

        ApiResponse<Void> response = controller.deleteTableMeta(DATASOURCE_ID, TABLE_NAME, "other", null, null);

        assertThat(response.getCode()).isEqualTo(200);
        verify(metaObjectService).delete(1L);
        verify(metaObjectService).delete(2L);
        verifyNoInteractions(tableAccessService);
    }

    @Test
    void createMetricMetaDoesNotTouchTableGrants() throws Exception {
        when(metaObjectService.create(any())).thenReturn(metaObject("metric_detail", "hankel_sales_amount"));
        DataSourcePublishedMetaRequest request = new DataSourcePublishedMetaRequest();
        request.setPayload(mapper.readTree("""
                {"metric_name":"hankel_sales_amount","source":{"table_view":"hankel_sales"}}
                """));

        ApiResponse<MetricsMetaObjectVO> response = controller.createMetricMeta(DATASOURCE_ID, request, null);

        assertThat(response.getCode()).isEqualTo(200);
        assertThat(response.getData().getObjectKey()).isEqualTo("hankel_sales_amount");
        verify(metaObjectService).create(any(MetricsMetaObjectRequest.class));
        verifyNoInteractions(tableAccessService);
    }

    private ApiResponse<DataSourcePublishedMetaVO> publish(
            boolean update, String tenantId, DataSourcePublishedMetaRequest request) {
        return update
                ? controller.updateTableMeta(DATASOURCE_ID, TABLE_NAME, tenantId, request, null)
                : controller.createTableMeta(DATASOURCE_ID, tenantId, request, null);
    }

    private DataSourcePublishedMetaRequest tableRequest() throws Exception {
        DataSourcePublishedMetaRequest request = new DataSourcePublishedMetaRequest();
        request.setPayload(mapper.readTree("""
                {"schemaName":"public","tableName":"hankel_sales","displayName":"Hankel Sales"}
                """));
        return request;
    }

    private DataSourcePublishedMetaRequest metricRequest() throws Exception {
        DataSourcePublishedMetaRequest request = new DataSourcePublishedMetaRequest();
        request.setPayload(mapper.readTree("""
                {"metric_name":"hankel_sales_amount","source":{"table_view":"hankel_sales"}}
                """));
        return request;
    }

    private void authorize(String schema, String table) {
        when(tableAccessService.isTableAuthorized(null, DATASOURCE_ID, schema, table)).thenReturn(true);
    }

    private DataSourceTableGrantRequest grantRequest(String pattern, String type) {
        DataSourceTableGrantRequest grant = new DataSourceTableGrantRequest();
        grant.setSchemaName("public");
        grant.setTablePattern(pattern);
        grant.setPatternType(type);
        return grant;
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
