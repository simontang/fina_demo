package com.fina.metrics.service;

import jakarta.servlet.http.HttpServletRequest;

public interface DataSourceApiKeyAuthService {

    AuthContext requirePermission(
            HttpServletRequest request,
            Long expectedDatasourceId,
            DataSourceApiKeyPermission permission);

    AuthContext requireAnyDatasourceKey(HttpServletRequest request);

    record AuthContext(
            Long datasourceId,
            String keyName
    ) {
    }
}
