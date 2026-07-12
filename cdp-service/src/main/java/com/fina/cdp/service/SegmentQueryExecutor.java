package com.fina.cdp.service;

import java.util.List;
import java.util.Map;

public interface SegmentQueryExecutor {

    List<Map<String, Object>> query(Long datasourceId, String sql, Map<String, Object> params);
}
