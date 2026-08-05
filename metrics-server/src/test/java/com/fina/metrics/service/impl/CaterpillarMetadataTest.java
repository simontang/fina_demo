package com.fina.metrics.service.impl;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashSet;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

class CaterpillarMetadataTest {

    private static final String PREFIX = "caterpillar_";

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void staticMetadataContainsCompleteCaterpillarScope() throws IOException {
        JsonNode tableCatalog = readJson("meta/table-catalog.json");
        Set<String> tableNames = new LinkedHashSet<>();
        tableCatalog.forEach(item -> {
            String tableName = item.path("tableName").asText("");
            if (tableName.startsWith(PREFIX)) {
                tableNames.add(tableName);
            }
        });

        assertThat(tableNames).hasSize(26);

        int columnCount = 0;
        for (String tableName : tableNames) {
            String resourceName = "meta/view-" + tableName.replace('_', '-') + ".json";
            JsonNode detail = readJson(resourceName);
            assertThat(detail.path("viewName").asText()).isEqualTo(tableName);
            assertThat(detail.path("selectSql").asText()).isEqualTo("SELECT * FROM " + tableName);
            assertThat(detail.path("columns").isArray()).isTrue();
            assertThat(detail.path("columns").size()).isPositive();
            detail.path("columns").forEach(column -> {
                assertThat(column.path("name").asText()).isNotBlank();
                assertThat(column.path("type").asText()).isNotBlank();
                assertThat(column.path("example").asText()).isNotBlank();
            });
            columnCount += detail.path("columns").size();
        }
        assertThat(columnCount).isEqualTo(339);

        JsonNode index = readJson("meta/metrics-index-meta.json");
        Set<String> indexMetrics = prefixedMetricNames(index.path("metrics_index"));
        JsonNode details = readJson("meta/metrics-detail-meta.json");
        Set<String> detailMetrics = prefixedMetricNames(details);

        assertThat(indexMetrics).hasSize(8);
        assertThat(detailMetrics).containsExactlyElementsOf(indexMetrics);

        details.forEach(detail -> {
            String metricName = detail.path("metric_name").asText("");
            if (!metricName.startsWith(PREFIX)) {
                return;
            }
            assertThat(detail.path("source_type").asText()).isEqualTo("cdp_postgres");
            assertThat(detail.path("source").path("table_view").asText()).isIn(tableNames);
        });

        assertThat(metricExpression(details, "caterpillar_qualified_lead_rate"))
                .contains("NULLIF(COUNT(*), 0)");
        assertThat(metricExpression(details, "caterpillar_call_answer_rate"))
                .contains("NULLIF(COUNT(*), 0)");
        assertThat(metricExpression(details, "caterpillar_assignment_acceptance_rate"))
                .contains("NULLIF(COUNT(*), 0)");
        assertThat(metricExpression(details, "caterpillar_assignment_order_rate"))
                .contains("NULLIF(COUNT(*), 0)");
        assertThat(metricExpression(details, "caterpillar_survey_completion_rate"))
                .contains("NULLIF(COUNT(*), 0)");
        assertThat(metricExpression(details, "caterpillar_paid_order_revenue"))
                .startsWith("COALESCE(");
    }

    private Set<String> prefixedMetricNames(JsonNode items) {
        Set<String> names = new LinkedHashSet<>();
        items.forEach(item -> {
            String name = item.path("metric_name").asText("");
            if (name.startsWith(PREFIX)) {
                names.add(name);
            }
        });
        return names;
    }

    private String metricExpression(JsonNode details, String metricName) {
        for (JsonNode detail : details) {
            if (metricName.equals(detail.path("metric_name").asText())) {
                return detail.path("calculation").path("sql_expression").asText();
            }
        }
        throw new AssertionError("Missing metric detail: " + metricName);
    }

    private JsonNode readJson(String resourceName) throws IOException {
        try (InputStream input = getClass().getClassLoader().getResourceAsStream(resourceName)) {
            assertThat(input).as(resourceName).isNotNull();
            return mapper.readTree(input);
        }
    }
}
