package com.fina.metrics.service;

import com.fina.metrics.dto.MetricsMetaObjectRequest;
import com.fina.metrics.dto.MetricsMetaObjectVO;
import com.fina.metrics.dto.PageResult;

import java.util.Collection;
import java.util.List;

public interface MetricsMetaObjectService {

    PageResult<MetricsMetaObjectVO> list(
            Long datasourceId,
            String objectType,
            String objectKey,
            Integer page,
            Integer pageSize);

    MetricsMetaObjectVO getById(Long id);

    PageResult<MetricsMetaObjectVO> listByDatasourceAndTypes(
            Long datasourceId,
            Collection<String> objectTypes,
            String objectKey,
            Integer page,
            Integer pageSize);

    MetricsMetaObjectVO create(MetricsMetaObjectRequest request);

    MetricsMetaObjectVO update(Long id, MetricsMetaObjectRequest request);

    MetricsMetaObjectVO updateByDatasourceTypeKey(
            Long datasourceId,
            String objectType,
            String objectKey,
            MetricsMetaObjectRequest request);

    void delete(Long id);

    void deleteByDatasourceTypeKey(Long datasourceId, String objectType, String objectKey);

    /**
     * Global rows are returned first, datasource-scoped rows second, so callers
     * can apply them in order and let datasource-scoped metadata override global metadata.
     */
    List<MetricsMetaObjectVO> listActiveForOverlay(String objectType, Long datasourceId);
}
