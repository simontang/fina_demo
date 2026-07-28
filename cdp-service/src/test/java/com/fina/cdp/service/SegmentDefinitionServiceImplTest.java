package com.fina.cdp.service;

import com.fina.cdp.dto.SegmentDefinitionRequest;
import com.fina.cdp.dto.PageResponse;
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
    void pagesDefinitionsWithinTenantUsingRequestedOffset() {
        SegmentDefinition definition = definition(21L, "tenant_a");
        when(mapper.countByTenant("tenant_a", "vip")).thenReturn(25L);
        when(mapper.selectPageByTenant("tenant_a", "vip", 10, 10)).thenReturn(List.of(definition));

        PageResponse<?> page = service.page("tenant_a", 2, 10, "  vip  ");

        assertThat(page.getTotal()).isEqualTo(25);
        assertThat(page.getPage()).isEqualTo(2);
        assertThat(page.getPageSize()).isEqualTo(10);
        assertThat(page.getItems()).hasSize(1);
        verify(mapper).countByTenant("tenant_a", "vip");
        verify(mapper).selectPageByTenant("tenant_a", "vip", 10, 10);
    }

    @Test
    void pageNormalizesInvalidValuesAndClampsPageSize() {
        when(mapper.countByTenant("tenant_a", null)).thenReturn(0L);
        when(mapper.selectPageByTenant("tenant_a", null, 20, 0)).thenReturn(List.of());
        when(mapper.selectPageByTenant("tenant_a", null, 200, 200)).thenReturn(List.of());

        PageResponse<?> defaults = service.page("tenant_a", 0, -1, "   ");
        PageResponse<?> clamped = service.page("tenant_a", 2, 500, null);

        assertThat(defaults.getPage()).isEqualTo(1);
        assertThat(defaults.getPageSize()).isEqualTo(20);
        assertThat(defaults.getItems()).isEmpty();
        assertThat(clamped.getPage()).isEqualTo(2);
        assertThat(clamped.getPageSize()).isEqualTo(200);
        verify(mapper, times(2)).countByTenant("tenant_a", null);
        verify(mapper).selectPageByTenant("tenant_a", null, 20, 0);
        verify(mapper).selectPageByTenant("tenant_a", null, 200, 200);
    }

    @Test
    void pageEscapesLikeWildcardsAndKeepsLargeOffsetPositive() {
        String escapedKeyword = "50\\%\\_off\\\\vip";
        long offset = 429_496_729_200L;
        when(mapper.countByTenant("tenant_a", escapedKeyword)).thenReturn(0L);
        when(mapper.selectPageByTenant(
                "tenant_a", escapedKeyword, 200, offset)).thenReturn(List.of());

        PageResponse<?> page = service.page(
                "tenant_a", Integer.MAX_VALUE, 200, "  50%_off\\vip  ");

        assertThat(page.getPage()).isEqualTo(Integer.MAX_VALUE);
        assertThat(page.getItems()).isEmpty();
        verify(mapper).countByTenant("tenant_a", escapedKeyword);
        verify(mapper).selectPageByTenant("tenant_a", escapedKeyword, 200, offset);
    }

    @Test
    void tenantCannotReadAnotherTenantDefinition() {
        when(mapper.selectByTenantAndId("tenant_b", 21L)).thenReturn(null);

        assertThatThrownBy(() -> service.getById("tenant_b", 21L))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("not found");
    }

    private SegmentDefinition definition(Long id, String tenantId) {
        SegmentDefinition definition = new SegmentDefinition();
        definition.setId(id);
        definition.setTenantId(tenantId);
        definition.setName("VIP");
        definition.setDatasourceId(7L);
        definition.setQuerySql("select customer_id from retailcdp_customers");
        definition.setStatus(1);
        return definition;
    }
}
