package com.fina.cdp.service;

import com.fina.cdp.dto.SegmentDefinitionRequest;
import com.fina.cdp.entity.SegmentDefinition;
import com.fina.cdp.mapper.SegmentDefinitionMapper;
import com.fina.cdp.service.impl.SegmentDefinitionServiceImpl;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.*;

class SegmentDefinitionServiceImplTest {

    private final SegmentDefinitionMapper mapper = mock(SegmentDefinitionMapper.class);
    private final SegmentDefinitionService service = new SegmentDefinitionServiceImpl(mapper);

    @Test
    void createAssignsTenantAndDefaultStatusFromServiceContext() {
        SegmentDefinitionRequest request = new SegmentDefinitionRequest();
        request.setName("Dormant customers");
        request.setDatasourceId(7L);
        request.setQuerySql("select customer_id from retailcdp_customers");

        when(mapper.insert(any(SegmentDefinition.class))).thenAnswer(invocation -> {
            SegmentDefinition inserted = invocation.getArgument(0);
            inserted.setId(11L);
            return 1;
        });

        service.create("tenant_a", request);

        ArgumentCaptor<SegmentDefinition> captor = ArgumentCaptor.forClass(SegmentDefinition.class);
        verify(mapper).insert(captor.capture());
        assertThat(captor.getValue().getTenantId()).isEqualTo("tenant_a");
        assertThat(captor.getValue().getStatus()).isEqualTo(1);
        assertThat(captor.getValue().getDatasourceId()).isEqualTo(7L);
    }

    @Test
    void readsAndDeletesDefinitionsOnlyWithinTenant() {
        SegmentDefinition definition = new SegmentDefinition();
        definition.setId(21L);
        definition.setTenantId("tenant_a");
        definition.setName("VIP");
        definition.setDatasourceId(7L);
        definition.setQuerySql("select customer_id from retailcdp_customers");
        definition.setStatus(1);

        when(mapper.selectByTenantAndId("tenant_a", 21L)).thenReturn(definition);
        when(mapper.selectByTenant("tenant_a")).thenReturn(List.of(definition));
        when(mapper.softDeleteByTenant("tenant_a", 21L)).thenReturn(1);

        assertThat(service.getById("tenant_a", 21L).getId()).isEqualTo(21L);
        assertThat(service.list("tenant_a")).hasSize(1);
        service.delete("tenant_a", 21L);

        verify(mapper).selectByTenantAndId("tenant_a", 21L);
        verify(mapper).selectByTenant("tenant_a");
        verify(mapper).softDeleteByTenant("tenant_a", 21L);
    }

    @Test
    void tenantCannotReadAnotherTenantDefinition() {
        when(mapper.selectByTenantAndId("tenant_b", 21L)).thenReturn(null);

        assertThatThrownBy(() -> service.getById("tenant_b", 21L))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("not found");
    }
}
