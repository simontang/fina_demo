package com.fina.metrics.hankel;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;

class HankelDistributorSellOutViewContractTest {

    private static final String VALUE_QUALITY_CONDITION =
            "q.is_quantity_outlier OR q.is_amount_outlier OR q.has_invalid_allocation_value";

    @Test
    void defaultSellOutAmountExcludesRowsWithAnyQualityIssue() throws Exception {
        String sql = normalize(Files.readString(Path.of("scripts/hankel/distributor-sell-out-view.sql")));

        assertThat(sql)
                .contains("WHEN " + VALUE_QUALITY_CONDITION
                        + " THEN NULL ELSE q.parsed_territory_sell_out END AS sell_out_value")
                .contains("WHEN " + VALUE_QUALITY_CONDITION
                        + " THEN COALESCE(q.parsed_territory_sell_out, 0) ELSE 0::numeric END AS excluded_sell_out_value")
                .contains(VALUE_QUALITY_CONDITION + " AS is_value_quality_excluded");
    }

    @Test
    void publishedMetricDescriptionMatchesTheViewGuardrail() throws Exception {
        String publishScript = Files.readString(Path.of("scripts/hankel/publish-distributor-sell-out-meta.sh"));

        assertThat(publishScript)
                .doesNotContain("不受数量异常影响")
                .contains("排除数量极端、金额极端或 Territory 金额不可解析");
    }

    private String normalize(String value) {
        return value.replaceAll("\\s+", " ").trim();
    }
}
