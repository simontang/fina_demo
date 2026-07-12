package com.fina.cdp.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.cdp.dto.SegmentProcessRequest;
import com.fina.cdp.entity.SegmentData;
import com.fina.cdp.entity.SegmentDefinition;
import com.fina.cdp.mapper.SegmentDataMapper;
import com.fina.cdp.mapper.SegmentDefinitionMapper;
import com.fina.cdp.service.impl.SegmentProcessingServiceImpl;
import com.fina.cdp.util.SqlSafetyValidator;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.*;

class SegmentProcessingServiceImplTest {

    private final SegmentDefinitionMapper definitionMapper = mock(SegmentDefinitionMapper.class);
    private final SegmentDataMapper dataMapper = mock(SegmentDataMapper.class);
    private final SegmentQueryExecutor queryExecutor = mock(SegmentQueryExecutor.class);
    private final SegmentProcessingService service = new SegmentProcessingServiceImpl(
            definitionMapper,
            dataMapper,
            queryExecutor,
            new SqlSafetyValidator(),
            new ObjectMapper());

    @Test
    void processCreatesSnapshotWithJsonArrayForArbitraryColumns() {
        SegmentDefinition definition = new SegmentDefinition();
        definition.setId(9L);
        definition.setTenantId("tenant_a");
        definition.setDatasourceId(7L);
        definition.setName("Dormant customers");
        definition.setQuerySql("select customer_id, total_spend_365 from retailcdp_customers where total_spend_365 > :minSpend");
        definition.setStatus(1);

        SegmentProcessRequest request = new SegmentProcessRequest();
        request.setParams(Map.of("minSpend", 1000));

        Map<String, Object> row1 = new LinkedHashMap<>();
        row1.put("customer_id", "C001");
        row1.put("total_spend_365", 1588.5);
        Map<String, Object> row2 = new LinkedHashMap<>();
        row2.put("customer_id", "C002");
        row2.put("preferred_channel", "wechat");

        when(definitionMapper.selectByTenantAndId("tenant_a", 9L)).thenReturn(definition);
        when(queryExecutor.query(7L, definition.getQuerySql(), request.getParams())).thenReturn(List.of(row1, row2));
        when(dataMapper.insert(any(SegmentData.class))).thenAnswer(invocation -> {
            SegmentData inserted = invocation.getArgument(0);
            inserted.setId(41L);
            return 1;
        });

        var result = service.process("tenant_a", 9L, request);

        ArgumentCaptor<SegmentData> captor = ArgumentCaptor.forClass(SegmentData.class);
        verify(dataMapper).insert(captor.capture());
        assertThat(captor.getValue().getTenantId()).isEqualTo("tenant_a");
        assertThat(captor.getValue().getDefinitionId()).isEqualTo(9L);
        assertThat(captor.getValue().getRunId()).isNotBlank();
        assertThat(captor.getValue().getRowCount()).isEqualTo(2);
        assertThat(captor.getValue().getDataJson()).contains("\"customer_id\":\"C001\"");
        assertThat(captor.getValue().getDataJson()).contains("\"preferred_channel\":\"wechat\"");
        assertThat(result.getRowCount()).isEqualTo(2);
    }

    @Test
    void tenantCannotProcessAnotherTenantDefinition() {
        SegmentProcessRequest request = new SegmentProcessRequest();
        when(definitionMapper.selectByTenantAndId("tenant_b", 9L)).thenReturn(null);

        assertThatThrownBy(() -> service.process("tenant_b", 9L, request))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("not found");

        verifyNoInteractions(queryExecutor);
        verify(dataMapper, never()).insert(any());
    }
}
