package com.fina.metrics.dto;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class SemanticQueryRequestJsonTest {

    private final ObjectMapper mapper = new ObjectMapper();
    private final Validator validator = Validation.buildDefaultValidatorFactory().getValidator();

    @Test
    void acceptsSnakeCaseQueryFieldsFromExternalAgentCalls() throws Exception {
        SemanticQueryRequest request = mapper.readValue("""
                {
                  "datasource_id": 15,
                  "metric_names": ["hankel_sell_out_value"],
                  "group_by": ["period_date__month"],
                  "order_by": [
                    {"metric_name": "hankel_sell_out_value", "direction": "DESC"}
                  ],
                  "max_rows": 20,
                  "custom_sql": null
                }
                """, SemanticQueryRequest.class);

        assertThat(request.getDatasourceId()).isEqualTo(15L);
        assertThat(request.getMetrics()).containsExactly("hankel_sell_out_value");
        assertThat(request.getGroupBy()).containsExactly("period_date__month");
        assertThat(request.getOrderBy()).hasSize(1);
        assertThat(request.getOrderBy().get(0).getField()).isEqualTo("hankel_sell_out_value");
        assertThat(request.getOrderBy().get(0).getDirection()).isEqualTo("DESC");
        assertThat(request.getLimit()).isEqualTo(20);
        assertThat(validator.validate(request)).isEmpty();
    }

    @Test
    void acceptsSingleValueAliasesForCommonAiGeneratedPayloads() throws Exception {
        SemanticQueryRequest request = mapper.readValue("""
                {
                  "data_source_id": 15,
                  "metric": "hankel_sell_out_value",
                  "dimensions": "period_date__month",
                  "sort": {"metric": "hankel_sell_out_value"}
                }
                """, SemanticQueryRequest.class);

        assertThat(request.getDatasourceId()).isEqualTo(15L);
        assertThat(request.getMetrics()).isEqualTo(List.of("hankel_sell_out_value"));
        assertThat(request.getGroupBy()).isEqualTo(List.of("period_date__month"));
        assertThat(request.getOrderBy()).hasSize(1);
        assertThat(request.getOrderBy().get(0).getField()).isEqualTo("hankel_sell_out_value");
        assertThat(request.getOrderBy().get(0).getDirection()).isEqualTo("ASC");
        assertThat(validator.validate(request)).isEmpty();
    }
}
