package com.fina.metrics.service;

import com.fina.metrics.dto.DataSourceApiKeyRequest;
import com.fina.metrics.dto.DataSourceApiKeyVO;

import java.util.List;

public interface DataSourceApiKeyService {

    List<DataSourceApiKeyVO> list(Long datasourceId);

    DataSourceApiKeyVO create(Long datasourceId, DataSourceApiKeyRequest request);

    DataSourceApiKeyVO disable(Long datasourceId, Long keyId);
}
