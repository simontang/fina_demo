package com.fina.cdp.service;

import com.fina.cdp.dto.SegmentDataVO;
import com.fina.cdp.dto.SegmentProcessRequest;

public interface SegmentProcessingService {

    SegmentDataVO process(String tenantId, Long definitionId, SegmentProcessRequest request);
}
