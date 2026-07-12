package com.fina.cdp.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.cdp.dto.PageResponse;
import com.fina.cdp.dto.SegmentDataRequest;
import com.fina.cdp.entity.SegmentData;
import com.fina.cdp.mapper.SegmentDataMapper;
import com.fina.cdp.mapper.SegmentDefinitionMapper;
import com.fina.cdp.service.impl.SegmentDataServiceImpl;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.*;

class SegmentDataServiceImplTest {

    private final SegmentDataMapper dataMapper = mock(SegmentDataMapper.class);
    private final SegmentDefinitionMapper definitionMapper = mock(SegmentDefinitionMapper.class);
    private final SegmentDataService service = new SegmentDataServiceImpl(dataMapper, definitionMapper, new ObjectMapper());

    @Test
    void createValidatesJsonArrayAndAssignsTenant() {
        SegmentDataRequest request = new SegmentDataRequest();
        request.setDefinitionId(9L);
        request.setRunId("run-1");
        request.setDataJson("[{\"customer_id\":\"C001\"},{\"customer_id\":\"C002\"}]");

        when(definitionMapper.existsByTenantAndId("tenant_a", 9L)).thenReturn(1);
        when(dataMapper.insert(any(SegmentData.class))).thenAnswer(invocation -> {
            SegmentData inserted = invocation.getArgument(0);
            inserted.setId(31L);
            return 1;
        });

        service.create("tenant_a", request);

        ArgumentCaptor<SegmentData> captor = ArgumentCaptor.forClass(SegmentData.class);
        verify(dataMapper).insert(captor.capture());
        assertThat(captor.getValue().getTenantId()).isEqualTo("tenant_a");
        assertThat(captor.getValue().getRowCount()).isEqualTo(2);
    }

    @Test
    void rejectsInvalidJsonData() {
        SegmentDataRequest request = new SegmentDataRequest();
        request.setDefinitionId(9L);
        request.setDataJson("{\"customer_id\":\"C001\"}");

        when(definitionMapper.existsByTenantAndId("tenant_a", 9L)).thenReturn(1);

        assertThatThrownBy(() -> service.create("tenant_a", request))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("JSON array");
    }

    @Test
    void listsAndDeletesDataOnlyWithinTenant() {
        SegmentData row = new SegmentData();
        row.setId(31L);
        row.setTenantId("tenant_a");
        row.setDefinitionId(9L);
        row.setRunId("run-1");
        row.setDataJson("[]");
        row.setRowCount(0);

        when(dataMapper.countByTenant("tenant_a", 9L)).thenReturn(1L);
        when(dataMapper.selectPageByTenant("tenant_a", 9L, 20, 0)).thenReturn(List.of(row));
        when(dataMapper.softDeleteByTenant("tenant_a", 31L)).thenReturn(1);

        PageResponse<?> page = service.list("tenant_a", 9L, 1, 20);
        service.delete("tenant_a", 31L);

        assertThat(page.getTotal()).isEqualTo(1);
        verify(dataMapper).selectPageByTenant("tenant_a", 9L, 20, 0);
        verify(dataMapper).softDeleteByTenant("tenant_a", 31L);
    }
}
