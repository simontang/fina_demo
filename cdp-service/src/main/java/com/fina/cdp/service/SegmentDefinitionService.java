package com.fina.cdp.service;

import com.fina.cdp.dto.SegmentDefinitionRequest;
import com.fina.cdp.dto.SegmentDefinitionVO;

import java.util.List;

public interface SegmentDefinitionService {

    List<SegmentDefinitionVO> list(String tenantId);

    SegmentDefinitionVO getById(String tenantId, Long id);

    SegmentDefinitionVO create(String tenantId, SegmentDefinitionRequest request);

    SegmentDefinitionVO update(String tenantId, Long id, SegmentDefinitionRequest request);

    void delete(String tenantId, Long id);
}
