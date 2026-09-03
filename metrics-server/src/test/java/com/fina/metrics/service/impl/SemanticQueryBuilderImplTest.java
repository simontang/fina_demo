package com.fina.metrics.service.impl;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.metrics.dto.SemanticQueryRequest;
import com.fina.metrics.service.SemanticQueryBuilder;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class SemanticQueryBuilderImplTest {

    private final ObjectMapper mapper = new ObjectMapper();
    private final SemanticQueryBuilderImpl builder = new SemanticQueryBuilderImpl();

    @Test
    void rendersPostgresTimeGrainForCdpDatasource() throws Exception {
        SemanticQueryRequest request = new SemanticQueryRequest();
        request.setGroupBy(List.of("event_time__day"));
        request.setLimit(50);

        JsonNode detail = mapper.readTree("""
                {
                  "metric_name": "cdp_page_event_count",
                  "default_time_context": {
                    "time_dimension": "event_time",
                    "supported_grains": ["day", "month", "year"]
                  },
                  "calculation": {"sql_expression": "COUNT(*)"},
                  "source": {"table_view": "demo_v_session_event_wide", "base_filters": []},
                  "supported_dimensions": []
                }
                """);

        SemanticQueryBuilder.BuildResult result = builder.buildMulti(
                List.of("cdp_page_event_count"),
                request,
                List.of(detail),
                "cdp_postgres");

        assertThat(result.sql()).contains("to_char(\"event_time\", 'YYYY-MM-DD') AS \"event_time__day\"");
        assertThat(result.sql()).contains("FROM \"demo_v_session_event_wide\"");
        assertThat(result.sql()).contains("GROUP BY to_char(\"event_time\", 'YYYY-MM-DD')");
        assertThat(result.sql()).doesNotContain("TO_NVARCHAR");
    }

    @Test
    void keepsHanaTimeGrainRenderingForSapDatasource() throws Exception {
        SemanticQueryRequest request = new SemanticQueryRequest();
        request.setGroupBy(List.of("DocDate__month"));

        JsonNode detail = mapper.readTree("""
                {
                  "metric_name": "order_amt_tax_inc",
                  "default_time_context": {
                    "time_dimension": "DocDate",
                    "supported_grains": ["day", "month", "year"]
                  },
                  "calculation": {"sql_expression": "SUM(\\"GTotal\\")"},
                  "source": {"table_view": "MTC_VW_AI_ORDR", "base_filters": []},
                  "supported_dimensions": []
                }
                """);

        SemanticQueryBuilder.BuildResult result = builder.buildMulti(
                List.of("order_amt_tax_inc"),
                request,
                List.of(detail),
                "sap_b1_hana");

        assertThat(result.sql()).contains("TO_NVARCHAR(\"DocDate\", 'YYYY-MM') AS \"DocDate__month\"");
    }

    @Test
    void quotesSchemaQualifiedTableViewByIdentifierPart() throws Exception {
        SemanticQueryRequest request = new SemanticQueryRequest();

        JsonNode detail = mapper.readTree("""
                {
                  "metric_name": "hankel_row_count",
                  "calculation": {"sql_expression": "COUNT(*)"},
                  "source": {"table_view": "public.hankel_distr_sell_out", "base_filters": []},
                  "supported_dimensions": []
                }
                """);

        SemanticQueryBuilder.BuildResult result = builder.buildMulti(
                List.of("hankel_row_count"),
                request,
                List.of(detail),
                "cdp_postgres");

        assertThat(result.sql()).contains("FROM \"public\".\"hankel_distr_sell_out\"");
        assertThat(result.sql()).doesNotContain("FROM \"public.hankel_distr_sell_out\"");
    }

    @Test
    void buildsAggregateMetricFromSqlFreeMeta() throws Exception {
        SemanticQueryRequest request = new SemanticQueryRequest();
        request.setGroupBy(List.of("sales_team", "year_month_time"));
        SemanticQueryRequest.FilterItem filter = new SemanticQueryRequest.FilterItem();
        filter.setDimension("posting_year");
        filter.setOperator("EQ");
        filter.setValues(List.of(2026));
        request.setFilters(List.of(filter));
        SemanticQueryRequest.OrderByItem orderBy = new SemanticQueryRequest.OrderByItem();
        orderBy.setField("hankel_sell_in_nes");
        orderBy.setDirection("DESC");
        request.setOrderBy(List.of(orderBy));
        request.setLimit(20);

        JsonNode detail = mapper.readTree("""
                {
                  "metric_name": "hankel_sell_in_nes",
                  "source_type": "cdp_postgres",
                  "source": {"table_view": "public.hankel_distr_sell_in", "base_filters": []},
                  "calculation": {
                    "type": "aggregate",
                    "aggregation": "sum",
                    "measure": "nes"
                  },
                  "supported_dimensions": [
                    {"dim_id": "sales_team", "field_name": "sales_team"},
                    {"dim_id": "year_month_time", "field_name": "year_month_time"},
                    {"dim_id": "posting_year", "field_name": "posting_year"}
                  ]
                }
                """);

        SemanticQueryBuilder.BuildResult result = builder.buildMulti(
                List.of("hankel_sell_in_nes"),
                request,
                List.of(detail),
                "cdp_postgres");

        assertThat(result.sql()).contains("SUM(\"nes\") AS \"hankel_sell_in_nes\"");
        assertThat(result.sql()).contains("\"posting_year\" = :f0_v0");
        assertThat(result.sql()).contains("GROUP BY \"sales_team\", \"year_month_time\"");
        assertThat(result.sql()).contains("ORDER BY \"hankel_sell_in_nes\" DESC");
        assertThat(result.sql()).contains("LIMIT 20");
        assertThat(result.params()).containsEntry("f0_v0", 2026);
    }

    @Test
    void buildsDerivedRatioMetricFromSqlFreeMeta() throws Exception {
        SemanticQueryRequest request = new SemanticQueryRequest();
        request.setGroupBy(List.of("sales_team"));

        JsonNode nes = aggregateMetric("hankel_sell_in_nes", "sum", "nes");
        JsonNode grossMargin = aggregateMetric("hankel_gross_margin", "sum", "gross_margin");
        JsonNode marginRate = mapper.readTree("""
                {
                  "metric_name": "hankel_gross_margin_rate",
                  "source_type": "cdp_postgres",
                  "source": {"table_view": "public.hankel_distr_sell_in", "base_filters": []},
                  "calculation": {
                    "type": "derived",
                    "operator": "ratio",
                    "numerator": "hankel_gross_margin",
                    "denominator": "hankel_sell_in_nes"
                  },
                  "supported_dimensions": [
                    {"dim_id": "sales_team", "field_name": "sales_team"}
                  ]
                }
                """);

        SemanticQueryBuilder.BuildResult result = builder.buildMulti(
                List.of("hankel_sell_in_nes", "hankel_gross_margin", "hankel_gross_margin_rate"),
                request,
                List.of(nes, grossMargin, marginRate),
                "cdp_postgres");

        assertThat(result.sql()).contains("SUM(\"nes\") AS \"hankel_sell_in_nes\"");
        assertThat(result.sql()).contains("SUM(\"gross_margin\") AS \"hankel_gross_margin\"");
        assertThat(result.sql()).contains(
                "(SUM(\"gross_margin\")) / NULLIF((SUM(\"nes\")), 0) AS \"hankel_gross_margin_rate\"");
    }

    @Test
    void buildsDerivedFormulaMetricFromSqlFreeMeta() throws Exception {
        SemanticQueryRequest request = new SemanticQueryRequest();

        JsonNode quantity = aggregateMetric("hankel_sell_in_quantity", "sum", "sell_in_quantity");
        JsonNode nes = aggregateMetric("hankel_sell_in_nes", "sum", "nes");
        JsonNode averagePrice = mapper.readTree("""
                {
                  "metric_name": "hankel_avg_sell_in_price",
                  "source_type": "cdp_postgres",
                  "source": {"table_view": "public.hankel_distr_sell_in", "base_filters": []},
                  "calculation": {
                    "type": "derived",
                    "formula": "hankel_sell_in_nes / hankel_sell_in_quantity"
                  },
                  "supported_dimensions": []
                }
                """);

        SemanticQueryBuilder.BuildResult result = builder.buildMulti(
                List.of("hankel_avg_sell_in_price"),
                request,
                List.of(averagePrice, nes, quantity),
                "cdp_postgres");

        assertThat(result.sql()).contains(
                "(SUM(\"nes\"))/(SUM(\"sell_in_quantity\")) AS \"hankel_avg_sell_in_price\"");
    }

    private JsonNode aggregateMetric(String metricName, String aggregation, String measure) throws Exception {
        return mapper.readTree("""
                {
                  "metric_name": "%s",
                  "source_type": "cdp_postgres",
                  "source": {"table_view": "public.hankel_distr_sell_in", "base_filters": []},
                  "calculation": {
                    "type": "aggregate",
                    "aggregation": "%s",
                    "measure": "%s"
                  },
                  "supported_dimensions": [
                    {"dim_id": "sales_team", "field_name": "sales_team"}
                  ]
                }
                """.formatted(metricName, aggregation, measure));
    }
}
