package com.fina.cdp.service.impl;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.cdp.config.TenantResolver;
import com.fina.cdp.dto.SegmentDataVO;
import com.fina.cdp.dto.SegmentProcessRequest;
import com.fina.cdp.entity.SegmentData;
import com.fina.cdp.entity.SegmentDefinition;
import com.fina.cdp.mapper.SegmentDataMapper;
import com.fina.cdp.mapper.SegmentDefinitionMapper;
import com.fina.cdp.service.SegmentProcessingService;
import com.fina.cdp.service.SegmentQueryExecutor;
import com.fina.cdp.util.SqlSafetyValidator;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class SegmentProcessingServiceImpl implements SegmentProcessingService {

    private final SegmentDefinitionMapper definitionMapper;
    private final SegmentDataMapper dataMapper;
    private final SegmentQueryExecutor queryExecutor;
    private final SqlSafetyValidator sqlSafetyValidator;
    private final ObjectMapper objectMapper;

    @Override
    @Transactional
    public SegmentDataVO process(String tenantId, Long definitionId, SegmentProcessRequest request) {
        String tenant = TenantResolver.normalize(tenantId);
        SegmentDefinition definition = definitionMapper.selectByTenantAndId(tenant, definitionId);
        if (definition == null) {
            throw new IllegalArgumentException("Segment definition not found: " + definitionId);
        }
        if (!Integer.valueOf(1).equals(definition.getStatus())) {
            throw new IllegalStateException("Segment definition is not active: " + definitionId);
        }

        sqlSafetyValidator.validateReadOnly(definition.getQuerySql());
        Map<String, Object> params = request != null && request.getParams() != null
                ? request.getParams()
                : Map.of();
        List<Map<String, Object>> rows = queryExecutor.query(
                definition.getDatasourceId(),
                definition.getQuerySql(),
                params);

        SegmentData data = new SegmentData();
        data.setTenantId(tenant);
        data.setDefinitionId(definitionId);
        data.setRunId(newRunId());
        data.setDataJson(toJson(rows));
        data.setRowCount(rows.size());
        data.setDeleted(0);
        dataMapper.insert(data);

        log.info("Processed segment definition tenant={} definition={} data={} rows={}",
                tenant, definitionId, data.getId(), data.getRowCount());
        return toVO(data);
    }

    private String toJson(List<Map<String, Object>> rows) {
        try {
            return objectMapper.writeValueAsString(rows);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Failed to serialize segment data rows", e);
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

    private static String newRunId() {
        return "seg_run_" + UUID.randomUUID();
    }
}
