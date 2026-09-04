package com.fina.metrics.service.impl;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.metrics.config.DynamicDataSourceManager;
import com.fina.metrics.dto.MetricsIndexResponse;
import com.fina.metrics.dto.MetricsQueryData;
import com.fina.metrics.dto.SemanticQueryRequest;
import com.fina.metrics.dto.DataSourceTableGrantVO;
import com.fina.metrics.dto.TableViewIndexItem;
import com.fina.metrics.entity.DataSourceConfig;
import com.fina.metrics.entity.MetricsMeta;
import com.fina.metrics.mapper.DataSourceConfigMapper;
import com.fina.metrics.mapper.MetricsMetaMapper;
import com.fina.metrics.service.DataSourceTableAccessService;
import com.fina.metrics.service.MetaCatalogService;
import com.fina.metrics.service.SemanticQueryBuilder;
import com.fina.metrics.service.TableViewMetaService;
import org.mockito.ArgumentCaptor;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.jdbc.core.ResultSetExtractor;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class MetricsServiceImplTest {

    private static final long DATASOURCE_ID = 42L;

    private final ObjectMapper mapper = new ObjectMapper();
    private MetricsMetaMapper metaMapper;
    private DataSourceConfigMapper datasourceMapper;
    private DynamicDataSourceManager dsManager;
    private MetaCatalogService catalog;
    private SemanticQueryBuilder queryBuilder;
    private TableViewMetaService tableViewMetaService;
    private DataSourceTableAccessService tableAccessService;
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
        dsManager = mock(DynamicDataSourceManager.class);
        catalog = mock(MetaCatalogService.class);
        queryBuilder = mock(SemanticQueryBuilder.class);
        tableViewMetaService = mock(TableViewMetaService.class);
        tableAccessService = mock(DataSourceTableAccessService.class);
        service = new MetricsServiceImpl(
                metaMapper,
                datasourceMapper,
                dsManager,
                catalog,
                queryBuilder,
                tableViewMetaService,
                tableAccessService);
        when(catalog.getCatalogVersion(DATASOURCE_ID)).thenReturn("1.0");
        when(catalog.getDomainCategories(DATASOURCE_ID)).thenReturn(List.of());
        when(catalog.getDetailItems(DATASOURCE_ID)).thenReturn(List.of());
        when(metaMapper.selectList(any())).thenReturn(List.of());
        when(tableViewMetaService.getTableViewsIndex(DATASOURCE_ID)).thenReturn(List.of());
        when(tableAccessService.hasActiveGrants(any(), any())).thenReturn(false);
        when(tableAccessService.listActiveGrants(any(), any())).thenReturn(List.of());
    }

    @Test
    void caterpillarIndexOnlyContainsCaterpillarMetadata() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Caterpillar PostgreSQL"));
        when(catalog.getIndexItems(DATASOURCE_ID)).thenReturn(List.of(
                indexMetric("caterpillar_leads_received", "cdp_postgres"),
                indexMetric("retailcdp_total_revenue", "cdp_postgres"),
                indexMetric("order_amt_tax_inc", null)));
        when(catalog.findDetailItem("caterpillar_leads_received", DATASOURCE_ID)).thenReturn(Optional.of(
                detailMetric("caterpillar_leads_received", "caterpillar_lead")));
        when(metaMapper.selectList(any())).thenReturn(List.of(registration("caterpillar_leads_received")));
        when(tableViewMetaService.getTableViewsIndex(DATASOURCE_ID)).thenReturn(List.of(
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
        when(catalog.getIndexItems(DATASOURCE_ID)).thenReturn(List.of(
                indexMetric("caterpillar_leads_received", "cdp_postgres")));
        when(catalog.findDetailItem("caterpillar_leads_received", DATASOURCE_ID)).thenReturn(Optional.of(
                detailMetric("caterpillar_leads_received", "caterpillar_lead")));
        when(tableViewMetaService.getTableViewsIndex(DATASOURCE_ID)).thenReturn(List.of(table("caterpillar_lead")));

        MetricsIndexResponse response = service.getMetricsIndex(DATASOURCE_ID);

        assertThat(response.getMetrics()).isEmpty();
        assertThat(response.getTables()).extracting(TableViewIndexItem::getTableName)
                .containsExactly("caterpillar_lead");
    }

    @Test
    void retailScopeKeepsExistingBehaviorAndHidesCaterpillarMetadata() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Retail CDP PostgreSQL"));
        when(catalog.getIndexItems(DATASOURCE_ID)).thenReturn(List.of(
                indexMetric("caterpillar_leads_received", null),
                indexMetric("retailcdp_total_revenue", "cdp_postgres"),
                indexMetric("order_amt_tax_inc", null)));
        when(tableViewMetaService.getTableViewsIndex(DATASOURCE_ID)).thenReturn(List.of(
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
    void tableGrantsFilterIndexTablesAndMetricSources() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Hankel PostgreSQL"));
        when(tableAccessService.listActiveGrants("hankel", DATASOURCE_ID))
                .thenReturn(List.of(tableGrant("hankel_")));
        when(catalog.getIndexItems(DATASOURCE_ID)).thenReturn(List.of(
                indexMetricWithSource("hankel_sales_amount", "hankel_sales"),
                indexMetricWithSource("retailcdp_total_revenue", "retailcdp_transactions")));
        when(tableViewMetaService.getTableViewsIndex(DATASOURCE_ID)).thenReturn(List.of(
                table("hankel_sales"),
                table("retailcdp_transactions")));

        MetricsIndexResponse response = service.getMetricsIndex(DATASOURCE_ID, "hankel");

        assertThat(response.getMetrics()).extracting(MetricsIndexResponse.MetricIndexItem::getMetricName)
                .containsExactly("hankel_sales_amount");
        assertThat(response.getMetrics()).extracting(MetricsIndexResponse.MetricIndexItem::isRegistered)
                .containsExactly(true);
        assertThat(response.getTables()).extracting(TableViewIndexItem::getTableName)
                .containsExactly("hankel_sales");
    }

    @Test
    void tableGrantedMetricIsHiddenUntilSourceTableMetaIsPublished() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Hankel PostgreSQL"));
        when(tableAccessService.listActiveGrants("hankel", DATASOURCE_ID))
                .thenReturn(List.of(tableGrant("hankel_")));
        when(catalog.getIndexItems(DATASOURCE_ID)).thenReturn(List.of(
                indexMetricWithSource("hankel_sales_amount", "hankel_sales")));
        when(tableViewMetaService.getTableViewsIndex(DATASOURCE_ID)).thenReturn(List.of());

        MetricsIndexResponse response = service.getMetricsIndex(DATASOURCE_ID, "hankel");

        assertThat(response.getMetrics()).isEmpty();
    }

    @Test
    void metricDetailExposesRuntimeQueryConstraints() throws Exception {
        JsonNode detail = mapper.readTree("""
                {
                  "metric_name": "hankel_final_score",
                  "display_name": "Hankel Final Score",
                  "source_type": "cdp_postgres",
                  "source": {"table_view": "public.hankel_leaderboard"},
                  "calculation": {"type": "aggregate", "aggregation": "avg", "measure": "score"},
                  "supported_dimensions": [
                    {"dim_id": "canonical_sales_name", "field_name": "canonical_sales_name"}
                  ],
                  "query_constraints": {
                    "required_group_by": ["canonical_sales_name"],
                    "non_additive": true
                  }
                }
                """);
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Hankel PostgreSQL"));
        when(tableAccessService.listActiveGrants("hankel", DATASOURCE_ID))
                .thenReturn(List.of(tableGrant("hankel_")));
        when(tableViewMetaService.getTableViewsIndex(DATASOURCE_ID))
                .thenReturn(List.of(table("hankel_leaderboard")));
        when(catalog.findDetailItem("hankel_final_score", DATASOURCE_ID))
                .thenReturn(Optional.of(detail));

        var response = service.getMetricDetail(DATASOURCE_ID, "hankel_final_score", "hankel");

        assertThat(response.getQueryConstraints())
                .containsEntry("non_additive", true)
                .containsEntry("required_group_by", List.of("canonical_sales_name"));
    }

    @Test
    void caterpillarSemanticQueryRejectsRetailMetric() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Caterpillar PostgreSQL"));
        when(catalog.findDetailItem("retailcdp_total_revenue", DATASOURCE_ID)).thenReturn(Optional.of(
                detailMetric("retailcdp_total_revenue", "retailcdp_transactions")));

        SemanticQueryRequest request = request("retailcdp_total_revenue");

        assertThatThrownBy(() -> service.query(request))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("not available for datasource type cdp_postgres");
    }

    @Test
    void caterpillarSemanticQueryRequiresActiveRegistration() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Caterpillar PostgreSQL"));
        when(catalog.findDetailItem("caterpillar_leads_received", DATASOURCE_ID)).thenReturn(Optional.of(
                detailMetric("caterpillar_leads_received", "caterpillar_lead")));
        when(tableViewMetaService.getTableViewsIndex(DATASOURCE_ID)).thenReturn(List.of(table("caterpillar_lead")));
        when(metaMapper.selectOne(any())).thenReturn(null);

        assertThatThrownBy(() -> service.query(request("caterpillar_leads_received")))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("MetricsMeta not found");
    }

    @Test
    void caterpillarSemanticQueryRejectsMetricsFromDifferentTables() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Caterpillar PostgreSQL"));
        when(catalog.findDetailItem("caterpillar_leads_received", DATASOURCE_ID)).thenReturn(Optional.of(
                detailMetric("caterpillar_leads_received", "caterpillar_lead")));
        when(catalog.findDetailItem("caterpillar_call_answer_rate", DATASOURCE_ID)).thenReturn(Optional.of(
                detailMetric("caterpillar_call_answer_rate", "caterpillar_call_record")));
        when(tableViewMetaService.getTableViewsIndex(DATASOURCE_ID)).thenReturn(List.of(
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
        when(catalog.findDetailItem("caterpillar_leads_received", DATASOURCE_ID)).thenReturn(Optional.of(
                detailMetric("caterpillar_leads_received", "caterpillar_missing")));
        when(metaMapper.selectOne(any())).thenReturn(new MetricsMeta());

        assertThatThrownBy(() -> service.query(request("caterpillar_leads_received")))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("references an unpublished table");
    }

    @Test
    void caterpillarNameWithoutCdpSourceTypeDoesNotEnableSpecialBranch() throws Exception {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(
                datasource("Caterpillar PostgreSQL", "sap_b1_hana"));
        when(catalog.findDetailItem("caterpillar_leads_received", DATASOURCE_ID)).thenReturn(Optional.of(
                detailMetric("caterpillar_leads_received", "caterpillar_lead")));

        assertThatThrownBy(() -> service.query(request("caterpillar_leads_received")))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("not available for datasource type sap_b1_hana");
    }

    @Test
    void otherCdpDatasourcesKeepSemanticQueryDisabled() {
        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Retail CDP PostgreSQL"));

        assertThatThrownBy(() -> service.query(request("retailcdp_total_revenue")))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Semantic metrics are not enabled for cdp_postgres yet");
    }

    @Test
    void cdpSemanticQueryWithTableGrantAllowsAuthorizedMetric() throws Exception {
        JsonNode detail = detailMetric("hankel_sales_row_count", "public.hankel_distr_sell_out");
        NamedParameterJdbcTemplate jdbc = mock(NamedParameterJdbcTemplate.class);
        String sql = "SELECT COUNT(*) AS \"hankel_sales_row_count\" FROM \"public\".\"hankel_distr_sell_out\" LIMIT 1";

        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Hankel PostgreSQL"));
        when(tableAccessService.listActiveGrants("hankel", DATASOURCE_ID))
                .thenReturn(List.of(tableGrant("hankel_")));
        when(tableViewMetaService.getTableViewsIndex(DATASOURCE_ID))
                .thenReturn(List.of(table("hankel_distr_sell_out")));
        when(catalog.findDetailItem("hankel_sales_row_count", DATASOURCE_ID)).thenReturn(Optional.of(detail));
        when(queryBuilder.buildMulti(
                eq(List.of("hankel_sales_row_count")),
                any(),
                eq(List.of(detail)),
                eq("cdp_postgres")))
                .thenReturn(new SemanticQueryBuilder.BuildResult(sql, Map.of(), List.of("hankel_sales_row_count")));
        when(dsManager.getNamedJdbcTemplate(DATASOURCE_ID)).thenReturn(jdbc);
        when(jdbc.query(
                eq(sql),
                any(MapSqlParameterSource.class),
                any(ResultSetExtractor.class)))
                .thenReturn(List.of(List.of(10L)));

        MetricsQueryData result = service.query(request("hankel_sales_row_count"), "hankel");

        assertThat(result.getSemanticModel()).isEqualTo("public.hankel_distr_sell_out");
        assertThat(result.getRows()).containsExactly(List.of(10L));
        verify(queryBuilder).buildMulti(
                eq(List.of("hankel_sales_row_count")),
                any(),
                eq(List.of(detail)),
                eq("cdp_postgres"));
    }

    @Test
    void cdpSemanticQueryCollectsSqlFreeDerivedMetricDependencies() throws Exception {
        JsonNode ratio = ratioMetric(
                "hankel_gross_margin_rate",
                "public.hankel_distr_sell_in",
                "hankel_gross_margin",
                "hankel_sell_in_nes");
        JsonNode grossMargin = aggregateMetric(
                "hankel_gross_margin",
                "public.hankel_distr_sell_in",
                "sum",
                "gross_margin");
        JsonNode nes = aggregateMetric(
                "hankel_sell_in_nes",
                "public.hankel_distr_sell_in",
                "sum",
                "nes");
        NamedParameterJdbcTemplate jdbc = mock(NamedParameterJdbcTemplate.class);
        String sql = "SELECT 1 AS \"hankel_gross_margin_rate\"";

        when(datasourceMapper.selectById(DATASOURCE_ID)).thenReturn(datasource("Hankel PostgreSQL"));
        when(tableAccessService.listActiveGrants("hankel", DATASOURCE_ID))
                .thenReturn(List.of(tableGrant("hankel_")));
        when(tableViewMetaService.getTableViewsIndex(DATASOURCE_ID))
                .thenReturn(List.of(table("hankel_distr_sell_in")));
        when(catalog.findDetailItem("hankel_gross_margin_rate", DATASOURCE_ID)).thenReturn(Optional.of(ratio));
        when(catalog.findDetailItem("hankel_gross_margin", DATASOURCE_ID)).thenReturn(Optional.of(grossMargin));
        when(catalog.findDetailItem("hankel_sell_in_nes", DATASOURCE_ID)).thenReturn(Optional.of(nes));
        when(queryBuilder.buildMulti(
                eq(List.of("hankel_gross_margin_rate")),
                any(),
                any(),
                eq("cdp_postgres")))
                .thenReturn(new SemanticQueryBuilder.BuildResult(sql, Map.of(), List.of("hankel_gross_margin_rate")));
        when(dsManager.getNamedJdbcTemplate(DATASOURCE_ID)).thenReturn(jdbc);
        when(jdbc.query(
                eq(sql),
                any(MapSqlParameterSource.class),
                any(ResultSetExtractor.class)))
                .thenReturn(List.of(List.of(1)));

        service.query(request("hankel_gross_margin_rate"), "hankel");

        @SuppressWarnings("unchecked")
        ArgumentCaptor<List<JsonNode>> detailsCaptor = ArgumentCaptor.forClass(List.class);
        verify(queryBuilder).buildMulti(
                eq(List.of("hankel_gross_margin_rate")),
                any(),
                detailsCaptor.capture(),
                eq("cdp_postgres"));
        assertThat(detailsCaptor.getValue())
                .extracting(node -> node.path("metric_name").asText())
                .containsExactly(
                        "hankel_gross_margin_rate",
                        "hankel_gross_margin",
                        "hankel_sell_in_nes");
    }

    @Test
    void customSqlRejectsWriteSqlBeforeExecution() {
        SemanticQueryRequest request = new SemanticQueryRequest();
        request.setDatasourceId(DATASOURCE_ID);
        request.setCustomSql("DELETE FROM OCRD");

        assertThatThrownBy(() -> service.query(request))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Only SELECT or WITH");
    }

    @Test
    void customSqlWithoutGrantsIsRejected() {
        SemanticQueryRequest request = new SemanticQueryRequest();
        request.setDatasourceId(DATASOURCE_ID);
        request.setCustomSql("SELECT * FROM OCRD");
        request.setLimit(25);

        assertThatThrownBy(() -> service.query(request, "tenant_5"))
                .isInstanceOf(com.fina.metrics.exception.ForbiddenException.class)
                .hasMessageContaining("requires active table grants");
    }

    @Test
    void customSqlWithGrantsUsesAuthorizedProbeAndKeepsAdhocSemanticModel() {
        MetricsQueryData data = MetricsQueryData.builder()
                .semanticModel("probe")
                .columns(List.of())
                .rows(List.of())
                .rowCount(0)
                .build();
        when(tableAccessService.hasActiveGrants("hankel", DATASOURCE_ID)).thenReturn(true);
        when(tableViewMetaService.getTableViewsIndex(DATASOURCE_ID))
                .thenReturn(List.of(table("hankel_sales")));
        when(tableAccessService.probeSql(eq("hankel"), eq(DATASOURCE_ID), any())).thenReturn(data);
        SemanticQueryRequest request = new SemanticQueryRequest();
        request.setDatasourceId(DATASOURCE_ID);
        request.setCustomSql("SELECT * FROM hankel_sales");
        request.setLimit(25);

        MetricsQueryData result = service.query(request, "hankel");

        assertThat(result.getSemanticModel()).isEqualTo("adhoc");
        verify(tableAccessService).probeSql(eq("hankel"), eq(DATASOURCE_ID), any());
    }

    @Test
    void customSqlWithGrantsRejectsUnpublishedTable() {
        when(tableAccessService.hasActiveGrants("hankel", DATASOURCE_ID)).thenReturn(true);
        when(tableViewMetaService.getTableViewsIndex(DATASOURCE_ID))
                .thenReturn(List.of(table("hankel_published")));
        SemanticQueryRequest request = new SemanticQueryRequest();
        request.setDatasourceId(DATASOURCE_ID);
        request.setCustomSql("SELECT * FROM hankel_raw");

        assertThatThrownBy(() -> service.query(request, "hankel"))
                .isInstanceOf(com.fina.metrics.exception.ForbiddenException.class)
                .hasMessageContaining("unpublished table: hankel_raw");
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

    private JsonNode indexMetricWithSource(String metricName, String tableView) throws Exception {
        return mapper.readTree("""
                {
                  "metric_name": "%s",
                  "source_type": "cdp_postgres",
                  "source": {"table_view": "%s"}
                }
                """.formatted(metricName, tableView));
    }

    private JsonNode detailMetric(String metricName, String tableView) throws Exception {
        return mapper.readTree("""
                {
                  "metric_name": "%s",
                  "source_type": "cdp_postgres",
                  "calculation": {"sql_expression": "COUNT(*)"},
                  "source": {"table_view": "%s", "base_filters": []},
                  "supported_dimensions": []
                }
                """.formatted(metricName, tableView));
    }

    private JsonNode aggregateMetric(
            String metricName,
            String tableView,
            String aggregation,
            String measure) throws Exception {
        return mapper.readTree("""
                {
                  "metric_name": "%s",
                  "source_type": "cdp_postgres",
                  "calculation": {
                    "type": "aggregate",
                    "aggregation": "%s",
                    "measure": "%s"
                  },
                  "source": {"table_view": "%s", "base_filters": []},
                  "supported_dimensions": []
                }
                """.formatted(metricName, aggregation, measure, tableView));
    }

    private JsonNode ratioMetric(
            String metricName,
            String tableView,
            String numerator,
            String denominator) throws Exception {
        return mapper.readTree("""
                {
                  "metric_name": "%s",
                  "source_type": "cdp_postgres",
                  "calculation": {
                    "type": "derived",
                    "operator": "ratio",
                    "numerator": "%s",
                    "denominator": "%s"
                  },
                  "source": {"table_view": "%s", "base_filters": []},
                  "supported_dimensions": []
                }
                """.formatted(metricName, numerator, denominator, tableView));
    }

    private TableViewIndexItem table(String tableName) {
        return TableViewIndexItem.builder().tableName(tableName).build();
    }

    private DataSourceTableGrantVO tableGrant(String prefix) {
        DataSourceTableGrantVO grant = new DataSourceTableGrantVO();
        grant.setTenantId("hankel");
        grant.setDatasourceId(DATASOURCE_ID);
        grant.setSchemaName("public");
        grant.setTablePattern(prefix);
        grant.setPatternType("PREFIX");
        grant.setCaseSensitive(false);
        grant.setStatus(1);
        return grant;
    }

    private SemanticQueryRequest request(String metricName) {
        SemanticQueryRequest request = new SemanticQueryRequest();
        request.setDatasourceId(DATASOURCE_ID);
        request.setMetrics(List.of(metricName));
        return request;
    }
}
