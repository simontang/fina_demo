package com.fina.cdp.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fina.cdp.dto.MarketingCampaignRequest;
import com.fina.cdp.dto.MarketingCampaignScheduleRequest;
import com.fina.cdp.dto.PageResponse;
import com.fina.cdp.entity.MarketingCampaign;
import com.fina.cdp.entity.SegmentData;
import com.fina.cdp.mapper.MarketingCampaignMapper;
import com.fina.cdp.mapper.SegmentDataMapper;
import com.fina.cdp.service.impl.MarketingCampaignServiceImpl;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.*;

class MarketingCampaignServiceImplTest {

    private final MarketingCampaignMapper mapper = mock(MarketingCampaignMapper.class);
    private final SegmentDataMapper segmentDataMapper = mock(SegmentDataMapper.class);
    private final MarketingCampaignService service = new MarketingCampaignServiceImpl(
            mapper,
            segmentDataMapper,
            new ObjectMapper());

    @Test
    void createAssignsTenantDefaultStatusAndDefaultStrategies() {
        MarketingCampaignRequest request = validRequest();
        request.setMainSegmentDataId(31L);
        SegmentData segmentData = new SegmentData();
        segmentData.setId(31L);
        segmentData.setTenantId("tenant_a");

        when(segmentDataMapper.selectByTenantAndId("tenant_a", 31L)).thenReturn(segmentData);
        when(mapper.insert(any(MarketingCampaign.class))).thenAnswer(invocation -> {
            MarketingCampaign inserted = invocation.getArgument(0);
            inserted.setId(11L);
            return 1;
        });

        var result = service.create("tenant_a", request);

        ArgumentCaptor<MarketingCampaign> captor = ArgumentCaptor.forClass(MarketingCampaign.class);
        verify(mapper).insert(captor.capture());
        MarketingCampaign inserted = captor.getValue();
        assertThat(inserted.getTenantId()).isEqualTo("tenant_a");
        assertThat(inserted.getStatus()).isEqualTo("draft");
        assertThat(inserted.getSegmentationStrategyJson()).isEqualTo("{}");
        assertThat(inserted.getControlGroupStrategyJson()).isEqualTo("{}");
        assertThat(inserted.getMainSegmentDataId()).isEqualTo(31L);
        assertThat(result.getStatus()).isEqualTo("draft");
    }

    @Test
    void createStoresObjectAndArrayStrategyJson() {
        MarketingCampaignRequest request = validRequest();
        ObjectNode segmentation = JsonNodeFactory.instance.objectNode();
        segmentation.put("rule", "vip");
        request.setSegmentationStrategy(segmentation);
        request.setWaveStrategy(JsonNodeFactory.instance.arrayNode().add("wave_1"));

        service.create("tenant_a", request);

        ArgumentCaptor<MarketingCampaign> captor = ArgumentCaptor.forClass(MarketingCampaign.class);
        verify(mapper).insert(captor.capture());
        assertThat(captor.getValue().getSegmentationStrategyJson()).contains("\"rule\":\"vip\"");
        assertThat(captor.getValue().getWaveStrategyJson()).isEqualTo("[\"wave_1\"]");
    }

    @Test
    void rejectsCrossTenantMainSegmentData() {
        MarketingCampaignRequest request = validRequest();
        request.setMainSegmentDataId(31L);
        when(segmentDataMapper.selectByTenantAndId("tenant_a", 31L)).thenReturn(null);

        assertThatThrownBy(() -> service.create("tenant_a", request))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Main segment data not found");

        verify(mapper, never()).insert(any());
    }

    @Test
    void listFiltersWithinTenantTypeAndStatus() {
        MarketingCampaign campaign = campaign(11L, "tenant_a", "scheduled");
        when(mapper.countByTenant("tenant_a", "reactivation", "scheduled")).thenReturn(1L);
        when(mapper.selectPageByTenant("tenant_a", "reactivation", "scheduled", 20, 0)).thenReturn(List.of(campaign));

        PageResponse<?> page = service.list("tenant_a", "reactivation", "scheduled", 1, 20);

        assertThat(page.getTotal()).isEqualTo(1);
        assertThat(page.getItems()).hasSize(1);
        verify(mapper).countByTenant("tenant_a", "reactivation", "scheduled");
        verify(mapper).selectPageByTenant("tenant_a", "reactivation", "scheduled", 20, 0);
    }

    @Test
    void tenantCannotReadOrDeleteAnotherTenantCampaign() {
        when(mapper.selectByTenantAndId("tenant_b", 11L)).thenReturn(null);
        when(mapper.softDeleteByTenant("tenant_b", 11L)).thenReturn(0);

        assertThatThrownBy(() -> service.getById("tenant_b", 11L))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("not found");
        assertThatThrownBy(() -> service.delete("tenant_b", 11L))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("not found");
    }

    @Test
    void startStopAndScheduleUpdateLifecycle() {
        MarketingCampaign campaign = campaign(11L, "tenant_a", "draft");
        when(mapper.selectByTenantAndId("tenant_a", 11L)).thenReturn(campaign);

        var started = service.start("tenant_a", 11L);
        assertThat(started.getStatus()).isEqualTo("running");
        assertThat(started.getActualStartedAt()).isNotNull();

        campaign.setStatus("running");
        var stopped = service.stop("tenant_a", 11L);
        assertThat(stopped.getStatus()).isEqualTo("stopped");
        assertThat(stopped.getActualStoppedAt()).isNotNull();

        campaign.setStatus("draft");
        MarketingCampaignScheduleRequest schedule = new MarketingCampaignScheduleRequest();
        schedule.setStartTime(LocalDateTime.now().plusHours(2));
        schedule.setEndTime(LocalDateTime.now().plusHours(3));
        var scheduled = service.schedule("tenant_a", 11L, schedule);
        assertThat(scheduled.getStatus()).isEqualTo("scheduled");
        assertThat(scheduled.getStartTime()).isEqualTo(schedule.getStartTime());
        assertThat(scheduled.getEndTime()).isEqualTo(schedule.getEndTime());
    }

    @Test
    void rejectsInvalidLifecycleTransitions() {
        MarketingCampaign completed = campaign(11L, "tenant_a", "completed");
        when(mapper.selectByTenantAndId("tenant_a", 11L)).thenReturn(completed);

        assertThatThrownBy(() -> service.start("tenant_a", 11L))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("cannot be started");
        assertThatThrownBy(() -> service.stop("tenant_a", 11L))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("cannot be stopped");
    }

    @Test
    void transitionDueCampaignsStartsScheduledAndCompletesExpiredRunning() {
        MarketingCampaign scheduled = campaign(11L, "tenant_a", "scheduled");
        MarketingCampaign running = campaign(12L, "tenant_a", "running");
        when(mapper.selectScheduledDue(any(LocalDateTime.class))).thenReturn(List.of(scheduled));
        when(mapper.selectRunningExpired(any(LocalDateTime.class))).thenReturn(List.of(running));

        int updated = service.transitionDueCampaigns();

        assertThat(updated).isEqualTo(2);
        assertThat(scheduled.getStatus()).isEqualTo("running");
        assertThat(scheduled.getActualStartedAt()).isNotNull();
        assertThat(running.getStatus()).isEqualTo("completed");
        verify(mapper, times(2)).updateById(any(MarketingCampaign.class));
    }

    private MarketingCampaignRequest validRequest() {
        MarketingCampaignRequest request = new MarketingCampaignRequest();
        request.setName("Dormant reactivation");
        request.setType("reactivation");
        request.setGoal("Reactivate dormant members");
        request.setStartTime(LocalDateTime.now().plusHours(1));
        request.setEndTime(LocalDateTime.now().plusDays(7));
        return request;
    }

    private MarketingCampaign campaign(Long id, String tenantId, String status) {
        MarketingCampaign campaign = new MarketingCampaign();
        campaign.setId(id);
        campaign.setTenantId(tenantId);
        campaign.setName("Dormant reactivation");
        campaign.setType("reactivation");
        campaign.setStatus(status);
        campaign.setGoal("Reactivate dormant members");
        campaign.setStartTime(LocalDateTime.now().minusHours(1));
        campaign.setEndTime(LocalDateTime.now().plusHours(1));
        campaign.setSegmentationStrategyJson("{}");
        campaign.setControlGroupStrategyJson("{}");
        campaign.setContentChannelStrategyJson("{}");
        campaign.setOfferStrategyJson("{}");
        campaign.setWaveStrategyJson("{}");
        campaign.setAbTestStrategyJson("{}");
        campaign.setStatisticsJson("{}");
        return campaign;
    }
}
