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
}
