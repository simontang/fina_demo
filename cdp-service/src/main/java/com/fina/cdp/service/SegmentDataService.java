package com.fina.cdp.service;

import com.fina.cdp.dto.PageResponse;
import com.fina.cdp.dto.SegmentDataRequest;
import com.fina.cdp.dto.SegmentDataVO;

public interface SegmentDataService {

    PageResponse<SegmentDataVO> list(String tenantId, Long definitionId, Integer page, Integer pageSize);

    SegmentDataVO getById(String tenantId, Long id);

    SegmentDataVO create(String tenantId, SegmentDataRequest request);

    SegmentDataVO update(String tenantId, Long id, SegmentDataRequest request);

    void delete(String tenantId, Long id);
}
