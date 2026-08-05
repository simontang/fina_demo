package com.fina.metrics.service.impl;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.metrics.config.DynamicDataSourceManager;
import com.fina.metrics.dto.MetricsIndexResponse;
import com.fina.metrics.dto.SemanticQueryRequest;
import com.fina.metrics.dto.TableViewIndexItem;
import com.fina.metrics.entity.DataSourceConfig;
import com.fina.metrics.entity.MetricsMeta;
import com.fina.metrics.mapper.DataSourceConfigMapper;
import com.fina.metrics.mapper.MetricsMetaMapper;
import com.fina.metrics.service.MetaCatalogService;
import com.fina.metrics.service.SemanticQueryBuilder;
import com.fina.metrics.service.TableViewMetaService;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class MetricsServiceImplTest {

    private static final long DATASOURCE_ID = 42L;

    private final ObjectMapper mapper = new ObjectMapper();
    private MetricsMetaMapper metaMapper;
    private DataSourceConfigMapper datasourceMapper;
    private MetaCatalogService catalog;
    private SemanticQueryBuilder queryBuilder;
    private TableViewMetaService tableViewMetaService;
    private MetricsServiceImpl service;

    @BeforeAll
    static void initializeMybatisMetadata() {
        TableInfoHelper.initTableInfo(
                new MapperBuilderAssistant(new MybatisConfiguration(), "test"),
                MetricsMeta.class);
    }

    @BeforeEach
    void setUp() {
        metaMapper = mock(MetricsMetaMapper.class);
        datasourceMapper = mock(DataSourceConfigMapper.class);
        catalog = mock(MetaCatalogService.class);
        queryBuilder = mock(SemanticQueryBuilder.class);
        tableViewMetaService = mock(TableViewMetaService.class);
        service = new MetricsServiceImpl(
                metaMapper,
                datasourceMapper,
                mock(DynamicDataSourceManager.class),
                catalog,
                queryBuilder,
                tableViewMetaService);
        when(catalog.getCatalogVersion()).thenReturn("1.0");
        when(catalog.getDomainCategories()).thenReturn(List.of());
        when(metaMapper.selectList(any())).thenReturn(List.of());
        when(tableViewMetaService.getTableViewsIndex()).thenReturn(List.of());
    }

    @Test
    void caterpillarIndexOnlyContainsCaterpillarMetadata() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Caterpillar PostgreSQL"));
        when(catalog.getIndexItems()).thenReturn(List.of(
                indexMetric("caterpillar_leads_received", null),
                indexMetric("retailcdp_total_revenue", "cdp_postgres"),
                indexMetric("order_amt_tax_inc", null)));
        when(catalog.findDetailItem("caterpillar_leads_received")).thenReturn(Optional.of(
                detailMetric("caterpillar_leads_received", "caterpillar_lead")));
        when(metaMapper.selectList(any())).thenReturn(List.of(registration("caterpillar_leads_received")));
        when(tableViewMetaService.getTableViewsIndex()).thenReturn(List.of(
                table("caterpillar_lead"),
                table("retailcdp_transactions"),
                table("MTC_VW_AI_ORDR")));

        MetricsIndexResponse response = service.getMetricsIndex(DATASOURCE_ID);

        assertThat(response.getMetrics()).extracting(MetricsIndexResponse.MetricIndexItem::getMetricName)
                .containsExactly("caterpillar_leads_received");
        assertThat(response.getTables()).extracting(TableViewIndexItem::getTableName)
                .containsExactly("caterpillar_lead");
    }

    @Test
    void caterpillarIndexHidesUnregisteredMetrics() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Caterpillar PostgreSQL"));
        when(catalog.getIndexItems()).thenReturn(List.of(
                indexMetric("caterpillar_leads_received", "cdp_postgres")));
        when(catalog.findDetailItem("caterpillar_leads_received")).thenReturn(Optional.of(
                detailMetric("caterpillar_leads_received", "caterpillar_lead")));
        when(tableViewMetaService.getTableViewsIndex()).thenReturn(List.of(table("caterpillar_lead")));

        MetricsIndexResponse response = service.getMetricsIndex(DATASOURCE_ID);

        assertThat(response.getMetrics()).isEmpty();
        assertThat(response.getTables()).extracting(TableViewIndexItem::getTableName)
                .containsExactly("caterpillar_lead");
    }

    @Test
    void retailScopeKeepsExistingBehaviorAndHidesCaterpillarMetadata() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Retail CDP PostgreSQL"));
        when(catalog.getIndexItems()).thenReturn(List.of(
                indexMetric("caterpillar_leads_received", null),
                indexMetric("retailcdp_total_revenue", "cdp_postgres"),
                indexMetric("order_amt_tax_inc", null)));
        when(tableViewMetaService.getTableViewsIndex()).thenReturn(List.of(
                table("caterpillar_lead"),
                table("retailcdp_transactions"),
                table("MTC_VW_AI_ORDR")));

        MetricsIndexResponse response = service.getMetricsIndex(DATASOURCE_ID);

        assertThat(response.getMetrics()).extracting(MetricsIndexResponse.MetricIndexItem::getMetricName)
                .containsExactly("retailcdp_total_revenue");
        assertThat(response.getTables()).extracting(TableViewIndexItem::getTableName)
                .containsExactly("retailcdp_transactions");
    }

    @Test
    void caterpillarSemanticQueryRejectsRetailMetric() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Caterpillar PostgreSQL"));
        when(catalog.findDetailItem("retailcdp_total_revenue")).thenReturn(Optional.of(
                detailMetric("retailcdp_total_revenue", "retailcdp_transactions")));

        SemanticQueryRequest request = request("retailcdp_total_revenue");

        assertThatThrownBy(() -> service.query(request))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("not available for Caterpillar datasource");
    }

    @Test
    void caterpillarSemanticQueryRequiresActiveRegistration() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Caterpillar PostgreSQL"));
        when(catalog.findDetailItem("caterpillar_leads_received")).thenReturn(Optional.of(
                detailMetric("caterpillar_leads_received", "caterpillar_lead")));
        when(tableViewMetaService.getTableViewsIndex()).thenReturn(List.of(table("caterpillar_lead")));
        when(metaMapper.selectOne(any())).thenReturn(null);

        assertThatThrownBy(() -> service.query(request("caterpillar_leads_received")))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("MetricsMeta not found");
    }

    @Test
    void caterpillarSemanticQueryRejectsMetricsFromDifferentTables() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Caterpillar PostgreSQL"));
        when(catalog.findDetailItem("caterpillar_leads_received")).thenReturn(Optional.of(
                detailMetric("caterpillar_leads_received", "caterpillar_lead")));
        when(catalog.findDetailItem("caterpillar_call_answer_rate")).thenReturn(Optional.of(
                detailMetric("caterpillar_call_answer_rate", "caterpillar_call_record")));
        when(tableViewMetaService.getTableViewsIndex()).thenReturn(List.of(
                table("caterpillar_lead"), table("caterpillar_call_record")));
        when(metaMapper.selectOne(any())).thenReturn(new MetricsMeta());

        SemanticQueryRequest request = new SemanticQueryRequest();
        request.setDatasourceId(DATASOURCE_ID);
        request.setMetrics(List.of("caterpillar_leads_received", "caterpillar_call_answer_rate"));

        assertThatThrownBy(() -> service.query(request))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("must share the same source table/view");
    }

    @Test
    void caterpillarSemanticQueryRejectsUnknownStaticTable() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Caterpillar PostgreSQL"));
        when(catalog.findDetailItem("caterpillar_leads_received")).thenReturn(Optional.of(
                detailMetric("caterpillar_leads_received", "caterpillar_missing")));
        when(metaMapper.selectOne(any())).thenReturn(new MetricsMeta());

        assertThatThrownBy(() -> service.query(request("caterpillar_leads_received")))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("not available for Caterpillar datasource");
    }

    @Test
    void caterpillarNameWithoutCdpSourceTypeDoesNotEnableSpecialBranch() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(
                datasource("Caterpillar PostgreSQL", "sap_b1_hana"));
        when(catalog.findDetailItem("caterpillar_leads_received")).thenReturn(Optional.of(
                detailMetric("caterpillar_leads_received", "caterpillar_lead")));

        assertThatThrownBy(() -> service.query(request("caterpillar_leads_received")))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("only available for Caterpillar datasource");
    }

    @Test
    void otherCdpDatasourcesKeepSemanticQueryDisabled() {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Retail CDP PostgreSQL"));

        assertThatThrownBy(() -> service.query(request("retailcdp_total_revenue")))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Semantic metrics are not enabled for cdp_postgres yet");
    }

    private DataSourceConfig datasource(String name) {
        return datasource(name, "cdp_postgres");
    }

    private DataSourceConfig datasource(String name, String sourceType) {
        DataSourceConfig datasource = new DataSourceConfig();
        datasource.setId(DATASOURCE_ID);
        datasource.setName(name);
        datasource.setSourceType(sourceType);
        datasource.setUrl("jdbc:postgresql://localhost/postgres");
        return datasource;
    }

    private MetricsMeta registration(String metricCode) {
        MetricsMeta meta = new MetricsMeta();
        meta.setMetricCode(metricCode);
        return meta;
    }

    private JsonNode indexMetric(String metricName, String sourceType) throws Exception {
        String sourceTypeJson = sourceType == null ? "" : ",\"source_type\":\"" + sourceType + "\"";
        return mapper.readTree("{\"metric_name\":\"" + metricName + "\"" + sourceTypeJson + "}");
    }

    private JsonNode detailMetric(String metricName, String tableView) throws Exception {
        return mapper.readTree("""
                {
                  "metric_name": "%s",
                  "calculation": {"sql_expression": "COUNT(*)"},
                  "source": {"table_view": "%s", "base_filters": []},
                  "supported_dimensions": []
                }
                """.formatted(metricName, tableView));
    }

    private TableViewIndexItem table(String tableName) {
        return TableViewIndexItem.builder().tableName(tableName).build();
    }

    private SemanticQueryRequest request(String metricName) {
        SemanticQueryRequest request = new SemanticQueryRequest();
        request.setDatasourceId(DATASOURCE_ID);
        request.setMetrics(List.of(metricName));
        return request;
    }
}
