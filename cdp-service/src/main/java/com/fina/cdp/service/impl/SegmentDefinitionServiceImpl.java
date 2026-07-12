package com.fina.cdp.service.impl;

import com.fina.cdp.config.TenantResolver;
import com.fina.cdp.dto.SegmentDefinitionRequest;
import com.fina.cdp.dto.SegmentDefinitionVO;
import com.fina.cdp.entity.SegmentDefinition;
import com.fina.cdp.mapper.SegmentDefinitionMapper;
import com.fina.cdp.service.SegmentDefinitionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class SegmentDefinitionServiceImpl implements SegmentDefinitionService {

    private final SegmentDefinitionMapper mapper;

    @Override
    public List<SegmentDefinitionVO> list(String tenantId) {
        String tenant = TenantResolver.normalize(tenantId);
        return mapper.selectByTenant(tenant).stream()
                .map(this::toVO)
                .toList();
    }

    @Override
    public SegmentDefinitionVO getById(String tenantId, Long id) {
        return toVO(requireDefinition(TenantResolver.normalize(tenantId), id));
    }

    @Override
    @Transactional
    public SegmentDefinitionVO create(String tenantId, SegmentDefinitionRequest request) {
        String tenant = TenantResolver.normalize(tenantId);
        SegmentDefinition definition = new SegmentDefinition();
        definition.setTenantId(tenant);
        definition.setName(request.getName());
        definition.setDescription(request.getDescription());
        definition.setDatasourceId(request.getDatasourceId());
        definition.setQuerySql(request.getQuerySql());
        definition.setStatus(request.getStatus() != null ? request.getStatus() : 1);
        definition.setDeleted(0);
        mapper.insert(definition);
        log.info("Created segment definition tenant={} id={} name={}", tenant, definition.getId(), definition.getName());
        return toVO(definition);
    }

    @Override
    @Transactional
    public SegmentDefinitionVO update(String tenantId, Long id, SegmentDefinitionRequest request) {
        String tenant = TenantResolver.normalize(tenantId);
        SegmentDefinition definition = requireDefinition(tenant, id);
        definition.setName(request.getName());
        definition.setDescription(request.getDescription());
        definition.setDatasourceId(request.getDatasourceId());
        definition.setQuerySql(request.getQuerySql());
        if (request.getStatus() != null) {
            definition.setStatus(request.getStatus());
        }
        mapper.updateById(definition);
        log.info("Updated segment definition tenant={} id={} name={}", tenant, id, definition.getName());
        return toVO(definition);
    }

    @Override
    @Transactional
    public void delete(String tenantId, Long id) {
        String tenant = TenantResolver.normalize(tenantId);
        int updated = mapper.softDeleteByTenant(tenant, id);
        if (updated == 0) {
            throw new IllegalArgumentException("Segment definition not found: " + id);
        }
        log.info("Deleted segment definition tenant={} id={}", tenant, id);
    }

    private SegmentDefinition requireDefinition(String tenantId, Long id) {
        SegmentDefinition definition = mapper.selectByTenantAndId(tenantId, id);
        if (definition == null) {
            throw new IllegalArgumentException("Segment definition not found: " + id);
        }
        return definition;
    }

    private SegmentDefinitionVO toVO(SegmentDefinition definition) {
        SegmentDefinitionVO vo = new SegmentDefinitionVO();
        vo.setId(definition.getId());
        vo.setTenantId(definition.getTenantId());
        vo.setName(definition.getName());
        vo.setDescription(definition.getDescription());
        vo.setDatasourceId(definition.getDatasourceId());
        vo.setQuerySql(definition.getQuerySql());
        vo.setStatus(definition.getStatus());
        vo.setCreatedAt(definition.getCreatedAt());
        vo.setUpdatedAt(definition.getUpdatedAt());
        return vo;
    }
}
