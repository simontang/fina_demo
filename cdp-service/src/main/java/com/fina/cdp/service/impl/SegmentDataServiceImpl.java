package com.fina.cdp.service.impl;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.cdp.config.TenantResolver;
import com.fina.cdp.dto.PageResponse;
import com.fina.cdp.dto.SegmentDataRequest;
import com.fina.cdp.dto.SegmentDataVO;
import com.fina.cdp.entity.SegmentData;
import com.fina.cdp.mapper.SegmentDataMapper;
import com.fina.cdp.mapper.SegmentDefinitionMapper;
import com.fina.cdp.service.SegmentDataService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class SegmentDataServiceImpl implements SegmentDataService {

    private static final int DEFAULT_PAGE = 1;
    private static final int DEFAULT_PAGE_SIZE = 20;
    private static final int MAX_PAGE_SIZE = 200;

    private final SegmentDataMapper dataMapper;
    private final SegmentDefinitionMapper definitionMapper;
    private final ObjectMapper objectMapper;

    @Override
    public PageResponse<SegmentDataVO> list(String tenantId, Long definitionId, Integer page, Integer pageSize) {
        String tenant = TenantResolver.normalize(tenantId);
        int safePage = page == null || page < 1 ? DEFAULT_PAGE : page;
        int safePageSize = pageSize == null || pageSize < 1
                ? DEFAULT_PAGE_SIZE
                : Math.min(pageSize, MAX_PAGE_SIZE);
        int offset = (safePage - 1) * safePageSize;
        long total = dataMapper.countByTenant(tenant, definitionId);
        List<SegmentDataVO> items = dataMapper.selectPageByTenant(tenant, definitionId, safePageSize, offset)
                .stream()
                .map(this::toVO)
                .toList();
        return new PageResponse<>(items, total, safePage, safePageSize);
    }

    @Override
    public SegmentDataVO getById(String tenantId, Long id) {
        return toVO(requireData(TenantResolver.normalize(tenantId), id));
    }

    @Override
    @Transactional
    public SegmentDataVO create(String tenantId, SegmentDataRequest request) {
        String tenant = TenantResolver.normalize(tenantId);
        requireDefinitionInTenant(tenant, request.getDefinitionId());
        JsonNode data = parseJsonArray(request.getDataJson());

        SegmentData segmentData = new SegmentData();
        segmentData.setTenantId(tenant);
        segmentData.setDefinitionId(request.getDefinitionId());
        segmentData.setRunId(hasText(request.getRunId()) ? request.getRunId().trim() : newRunId());
        segmentData.setDataJson(request.getDataJson());
        segmentData.setRowCount(data.size());
        segmentData.setDeleted(0);
        dataMapper.insert(segmentData);
        log.info("Created segment data tenant={} definition={} id={} rows={}",
                tenant, request.getDefinitionId(), segmentData.getId(), segmentData.getRowCount());
        return toVO(segmentData);
    }

    @Override
    @Transactional
    public SegmentDataVO update(String tenantId, Long id, SegmentDataRequest request) {
        String tenant = TenantResolver.normalize(tenantId);
        SegmentData existing = requireData(tenant, id);
        requireDefinitionInTenant(tenant, request.getDefinitionId());
        JsonNode data = parseJsonArray(request.getDataJson());

        existing.setDefinitionId(request.getDefinitionId());
        existing.setRunId(hasText(request.getRunId()) ? request.getRunId().trim() : existing.getRunId());
        existing.setDataJson(request.getDataJson());
        existing.setRowCount(data.size());
        dataMapper.updateById(existing);
        log.info("Updated segment data tenant={} id={} rows={}", tenant, id, existing.getRowCount());
        return toVO(existing);
    }

    @Override
    @Transactional
    public void delete(String tenantId, Long id) {
        String tenant = TenantResolver.normalize(tenantId);
        int updated = dataMapper.softDeleteByTenant(tenant, id);
        if (updated == 0) {
            throw new IllegalArgumentException("Segment data not found: " + id);
        }
        log.info("Deleted segment data tenant={} id={}", tenant, id);
    }

    private SegmentData requireData(String tenantId, Long id) {
        SegmentData data = dataMapper.selectByTenantAndId(tenantId, id);
        if (data == null) {
            throw new IllegalArgumentException("Segment data not found: " + id);
        }
        return data;
    }

    private void requireDefinitionInTenant(String tenantId, Long definitionId) {
        if (definitionMapper.existsByTenantAndId(tenantId, definitionId) == 0) {
            throw new IllegalArgumentException("Segment definition not found: " + definitionId);
        }
    }

    private JsonNode parseJsonArray(String dataJson) {
        try {
            JsonNode node = objectMapper.readTree(dataJson);
            if (!node.isArray()) {
                throw new IllegalArgumentException("segment data dataJson must be a JSON array");
            }
            return node;
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalArgumentException("segment data dataJson must be valid JSON array", e);
        }
    }

    private SegmentDataVO toVO(SegmentData data) {
        SegmentDataVO vo = new SegmentDataVO();
        vo.setId(data.getId());
        vo.setTenantId(data.getTenantId());
        vo.setDefinitionId(data.getDefinitionId());
        vo.setRunId(data.getRunId());
        vo.setDataJson(data.getDataJson());
        vo.setRowCount(data.getRowCount());
        vo.setCreatedAt(data.getCreatedAt());
        vo.setUpdatedAt(data.getUpdatedAt());
        return vo;
    }

    private static boolean hasText(String value) {
        return value != null && !value.isBlank();
    }

    private static String newRunId() {
        return "seg_run_" + UUID.randomUUID();
    }
}
