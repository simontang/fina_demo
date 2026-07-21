package com.fina.cdp.service.impl;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.cdp.config.TenantResolver;
import com.fina.cdp.dto.MarketingCampaignRequest;
import com.fina.cdp.dto.MarketingCampaignScheduleRequest;
import com.fina.cdp.dto.MarketingCampaignVO;
import com.fina.cdp.dto.PageResponse;
import com.fina.cdp.entity.MarketingCampaign;
import com.fina.cdp.mapper.MarketingCampaignMapper;
import com.fina.cdp.mapper.SegmentDataMapper;
import com.fina.cdp.service.MarketingCampaignService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Set;

@Slf4j
@Service
@RequiredArgsConstructor
public class MarketingCampaignServiceImpl implements MarketingCampaignService {

    private static final int DEFAULT_PAGE = 1;
    private static final int DEFAULT_PAGE_SIZE = 20;
    private static final int MAX_PAGE_SIZE = 200;
    private static final String EMPTY_JSON = "{}";
    private static final String STATUS_DRAFT = "draft";
    private static final String STATUS_SCHEDULED = "scheduled";
    private static final String STATUS_RUNNING = "running";
    private static final String STATUS_STOPPED = "stopped";
    private static final String STATUS_COMPLETED = "completed";
    private static final Set<String> STATUSES = Set.of(
            STATUS_DRAFT,
            STATUS_SCHEDULED,
            STATUS_RUNNING,
            STATUS_STOPPED,
            STATUS_COMPLETED);

    private final MarketingCampaignMapper mapper;
    private final SegmentDataMapper segmentDataMapper;
    private final ObjectMapper objectMapper;

    @Override
    public PageResponse<MarketingCampaignVO> list(String tenantId, String type, String status, Integer page, Integer pageSize) {
        String tenant = TenantResolver.normalize(tenantId);
        String normalizedType = normalizeOptional(type);
        String normalizedStatus = normalizeOptionalStatus(status);
        int safePage = page == null || page < 1 ? DEFAULT_PAGE : page;
        int safePageSize = pageSize == null || pageSize < 1
                ? DEFAULT_PAGE_SIZE
                : Math.min(pageSize, MAX_PAGE_SIZE);
        int offset = (safePage - 1) * safePageSize;
        long total = mapper.countByTenant(tenant, normalizedType, normalizedStatus);
        List<MarketingCampaignVO> items = mapper.selectPageByTenant(tenant, normalizedType, normalizedStatus, safePageSize, offset)
                .stream()
                .map(this::toVO)
                .toList();
        return new PageResponse<>(items, total, safePage, safePageSize);
    }

    @Override
    public MarketingCampaignVO getById(String tenantId, Long id) {
        return toVO(requireCampaign(TenantResolver.normalize(tenantId), id));
    }

    @Override
    @Transactional
    public MarketingCampaignVO create(String tenantId, MarketingCampaignRequest request) {
        String tenant = TenantResolver.normalize(tenantId);
        validateTimes(request.getStartTime(), request.getEndTime());
        validateSegmentData(tenant, request.getMainSegmentDataId());

        MarketingCampaign campaign = new MarketingCampaign();
        campaign.setTenantId(tenant);
        applyRequest(campaign, request, true);
        campaign.setDeleted(0);
        mapper.insert(campaign);
        log.info("Created marketing campaign tenant={} id={} name={}", tenant, campaign.getId(), campaign.getName());
        return toVO(campaign);
    }

    @Override
    @Transactional
    public MarketingCampaignVO update(String tenantId, Long id, MarketingCampaignRequest request) {
        String tenant = TenantResolver.normalize(tenantId);
        MarketingCampaign campaign = requireCampaign(tenant, id);
        validateTimes(request.getStartTime(), request.getEndTime());
        validateSegmentData(tenant, request.getMainSegmentDataId());

        applyRequest(campaign, request, false);
        mapper.updateById(campaign);
        log.info("Updated marketing campaign tenant={} id={} name={}", tenant, id, campaign.getName());
        return toVO(campaign);
    }

    @Override
    @Transactional
    public void delete(String tenantId, Long id) {
        String tenant = TenantResolver.normalize(tenantId);
        int updated = mapper.softDeleteByTenant(tenant, id);
        if (updated == 0) {
            throw new IllegalArgumentException("Marketing campaign not found: " + id);
        }
        log.info("Deleted marketing campaign tenant={} id={}", tenant, id);
    }

    @Override
    @Transactional
    public MarketingCampaignVO start(String tenantId, Long id) {
        String tenant = TenantResolver.normalize(tenantId);
        MarketingCampaign campaign = requireCampaign(tenant, id);
        if (!Set.of(STATUS_DRAFT, STATUS_SCHEDULED).contains(campaign.getStatus())) {
            throw new IllegalStateException("Marketing campaign cannot be started from status: " + campaign.getStatus());
        }
        LocalDateTime now = LocalDateTime.now();
        if (!campaign.getEndTime().isAfter(now)) {
            throw new IllegalStateException("Marketing campaign endTime is already past: " + id);
        }
        campaign.setStatus(STATUS_RUNNING);
        campaign.setActualStartedAt(now);
        mapper.updateById(campaign);
        log.info("Started marketing campaign tenant={} id={}", tenant, id);
        return toVO(campaign);
    }

    @Override
    @Transactional
    public MarketingCampaignVO stop(String tenantId, Long id) {
        String tenant = TenantResolver.normalize(tenantId);
        MarketingCampaign campaign = requireCampaign(tenant, id);
        if (!Set.of(STATUS_SCHEDULED, STATUS_RUNNING).contains(campaign.getStatus())) {
            throw new IllegalStateException("Marketing campaign cannot be stopped from status: " + campaign.getStatus());
        }
        campaign.setStatus(STATUS_STOPPED);
        campaign.setActualStoppedAt(LocalDateTime.now());
        mapper.updateById(campaign);
        log.info("Stopped marketing campaign tenant={} id={}", tenant, id);
        return toVO(campaign);
    }

    @Override
    @Transactional
    public MarketingCampaignVO schedule(String tenantId, Long id, MarketingCampaignScheduleRequest request) {
        String tenant = TenantResolver.normalize(tenantId);
        MarketingCampaign campaign = requireCampaign(tenant, id);
        LocalDateTime start = request != null && request.getStartTime() != null
                ? request.getStartTime()
                : campaign.getStartTime();
        LocalDateTime end = request != null && request.getEndTime() != null
                ? request.getEndTime()
                : campaign.getEndTime();
        validateTimes(start, end);
        campaign.setStartTime(start);
        campaign.setEndTime(end);
        campaign.setStatus(STATUS_SCHEDULED);
        mapper.updateById(campaign);
        log.info("Scheduled marketing campaign tenant={} id={} start={} end={}", tenant, id, start, end);
        return toVO(campaign);
    }

    @Override
    @Transactional
    public int transitionDueCampaigns() {
        LocalDateTime now = LocalDateTime.now();
        int updated = 0;
        for (MarketingCampaign campaign : mapper.selectScheduledDue(now)) {
            campaign.setStatus(STATUS_RUNNING);
            campaign.setActualStartedAt(now);
            mapper.updateById(campaign);
            updated++;
        }
        for (MarketingCampaign campaign : mapper.selectRunningExpired(now)) {
            campaign.setStatus(STATUS_COMPLETED);
            mapper.updateById(campaign);
            updated++;
        }
        return updated;
    }

    private void applyRequest(MarketingCampaign campaign, MarketingCampaignRequest request, boolean creating) {
        campaign.setThreadId(request.getThreadId());
        campaign.setName(request.getName());
        campaign.setDescription(request.getDescription());
        campaign.setType(request.getType());
        campaign.setGoal(request.getGoal());
        campaign.setStartTime(request.getStartTime());
        campaign.setEndTime(request.getEndTime());
        campaign.setMainSegmentDataId(request.getMainSegmentDataId());
        campaign.setStatus(creating
                ? normalizeStatusOrDefault(request.getStatus(), STATUS_DRAFT)
                : normalizeStatusOrDefault(request.getStatus(), campaign.getStatus()));
        campaign.setSegmentationStrategyJson(toJsonOrDefault(request.getSegmentationStrategy()));
        campaign.setControlGroupStrategyJson(toJsonOrDefault(request.getControlGroupStrategy()));
        campaign.setContentChannelStrategyJson(toJsonOrDefault(request.getContentChannelStrategy()));
        campaign.setOfferStrategyJson(toJsonOrDefault(request.getOfferStrategy()));
        campaign.setWaveStrategyJson(toJsonOrDefault(request.getWaveStrategy()));
        campaign.setAbTestStrategyJson(toJsonOrDefault(request.getAbTestStrategy()));
        campaign.setStatisticsJson(toJsonOrDefault(request.getStatistics()));
    }

    private MarketingCampaign requireCampaign(String tenantId, Long id) {
        MarketingCampaign campaign = mapper.selectByTenantAndId(tenantId, id);
        if (campaign == null) {
            throw new IllegalArgumentException("Marketing campaign not found: " + id);
        }
        return campaign;
    }

    private void validateSegmentData(String tenantId, Long segmentDataId) {
        if (segmentDataId != null && segmentDataMapper.selectByTenantAndId(tenantId, segmentDataId) == null) {
            throw new IllegalArgumentException("Main segment data not found: " + segmentDataId);
        }
    }

    private void validateTimes(LocalDateTime startTime, LocalDateTime endTime) {
        if (startTime == null) {
            throw new IllegalArgumentException("startTime is required");
        }
        if (endTime == null) {
            throw new IllegalArgumentException("endTime is required");
        }
        if (!endTime.isAfter(startTime)) {
            throw new IllegalArgumentException("endTime must be after startTime");
        }
    }

    private String normalizeOptionalStatus(String status) {
        String normalized = normalizeOptional(status);
        if (normalized != null) {
            requireValidStatus(normalized);
        }
        return normalized;
    }

    private String normalizeStatusOrDefault(String status, String fallback) {
        String normalized = normalizeOptional(status);
        if (normalized == null) {
            return fallback;
        }
        requireValidStatus(normalized);
        return normalized;
    }

    private void requireValidStatus(String status) {
        if (!STATUSES.contains(status)) {
            throw new IllegalArgumentException("Invalid marketing campaign status: " + status);
        }
    }

    private String normalizeOptional(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }

    private String toJsonOrDefault(JsonNode node) {
        if (node == null || node.isNull()) {
            return EMPTY_JSON;
        }
        if (!node.isObject() && !node.isArray()) {
            throw new IllegalArgumentException("Marketing campaign JSON strategy fields must be JSON object or array");
        }
        try {
            return objectMapper.writeValueAsString(node);
        } catch (JsonProcessingException e) {
            throw new IllegalArgumentException("Marketing campaign JSON strategy fields must be valid JSON", e);
        }
    }

    private JsonNode parseJson(String json) {
        try {
            return objectMapper.readTree(json == null || json.isBlank() ? EMPTY_JSON : json);
        } catch (Exception e) {
            throw new IllegalStateException("Stored marketing campaign JSON is invalid", e);
        }
    }

    private MarketingCampaignVO toVO(MarketingCampaign campaign) {
        MarketingCampaignVO vo = new MarketingCampaignVO();
        vo.setId(campaign.getId());
        vo.setTenantId(campaign.getTenantId());
        vo.setThreadId(campaign.getThreadId());
        vo.setName(campaign.getName());
        vo.setDescription(campaign.getDescription());
        vo.setType(campaign.getType());
        vo.setStatus(campaign.getStatus());
        vo.setGoal(campaign.getGoal());
        vo.setStartTime(campaign.getStartTime());
        vo.setEndTime(campaign.getEndTime());
        vo.setMainSegmentDataId(campaign.getMainSegmentDataId());
        vo.setSegmentationStrategy(parseJson(campaign.getSegmentationStrategyJson()));
        vo.setControlGroupStrategy(parseJson(campaign.getControlGroupStrategyJson()));
        vo.setContentChannelStrategy(parseJson(campaign.getContentChannelStrategyJson()));
        vo.setOfferStrategy(parseJson(campaign.getOfferStrategyJson()));
        vo.setWaveStrategy(parseJson(campaign.getWaveStrategyJson()));
        vo.setAbTestStrategy(parseJson(campaign.getAbTestStrategyJson()));
        vo.setStatistics(parseJson(campaign.getStatisticsJson()));
        vo.setActualStartedAt(campaign.getActualStartedAt());
        vo.setActualStoppedAt(campaign.getActualStoppedAt());
        vo.setCreatedAt(campaign.getCreatedAt());
        vo.setUpdatedAt(campaign.getUpdatedAt());
        return vo;
    }
}
