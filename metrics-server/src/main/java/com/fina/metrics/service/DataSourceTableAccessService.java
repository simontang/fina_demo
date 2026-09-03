package com.fina.metrics.service;

import com.fina.metrics.dto.*;

import java.util.List;

public interface DataSourceTableAccessService {

    List<DataSourceTableGrantVO> listGrants(String tenantId, Long datasourceId);

    List<DataSourceTableGrantVO> listActiveGrants(String tenantId, Long datasourceId);

    DataSourceTableGrantVO createGrant(String tenantId, Long datasourceId, DataSourceTableGrantRequest request);

    DataSourceTableGrantVO updateGrant(String tenantId, Long datasourceId, Long grantId, DataSourceTableGrantRequest request);

    void deleteGrant(String tenantId, Long datasourceId, Long grantId);

    boolean hasActiveGrants(String tenantId, Long datasourceId);

    boolean isTableAuthorized(String tenantId, Long datasourceId, String schemaName, String tableName);

    boolean isTableAuthorizedIfGrantsConfigured(String tenantId, Long datasourceId, String schemaName, String tableName);

    void assertSqlAuthorized(String tenantId, Long datasourceId, String sql);

    List<DataSourceTableVO> listPhysicalTables(Long datasourceId, String schemaName);

    List<DataSourceTableVO> listAuthorizedTables(String tenantId, Long datasourceId);

    List<DataSourceColumnVO> listAuthorizedColumns(String tenantId, Long datasourceId, String schemaName, String tableName);

    MetricsQueryData queryDatasource(Long datasourceId, SqlProbeRequest request);

    MetricsQueryData probeSql(String tenantId, Long datasourceId, SqlProbeRequest request);
}
