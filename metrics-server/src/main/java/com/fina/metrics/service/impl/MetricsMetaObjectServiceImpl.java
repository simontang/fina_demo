package com.fina.metrics.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.metrics.dto.MetricsMetaObjectRequest;
import com.fina.metrics.dto.MetricsMetaObjectVO;
import com.fina.metrics.dto.PageResult;
import com.fina.metrics.entity.MetricsMetaObject;
import com.fina.metrics.mapper.MetricsMetaObjectMapper;
import com.fina.metrics.service.MetricsMetaObjectService;
import com.fina.metrics.service.MetricsMetaObjectTypes;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class MetricsMetaObjectServiceImpl implements MetricsMetaObjectService {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final MetricsMetaObjectMapper mapper;

    @Override
    public PageResult<MetricsMetaObjectVO> list(
            Long datasourceId,
            String objectType,
            String objectKey,
            Integer page,
            Integer pageSize) {
        int safePage = page == null || page <= 0 ? 1 : page;
        int safePageSize = pageSize == null || pageSize <= 0 ? 20 : Math.min(pageSize, 200);

        LambdaQueryWrapper<MetricsMetaObject> wrapper = new LambdaQueryWrapper<MetricsMetaObject>()
                .eq(MetricsMetaObject::getDeleted, 0);
        if (datasourceId != null) {
            wrapper.eq(MetricsMetaObject::getDatasourceId, datasourceId);
        }
        if (StringUtils.hasText(objectType)) {
            validateObjectType(objectType);
            wrapper.eq(MetricsMetaObject::getObjectType, objectType);
        }
        if (StringUtils.hasText(objectKey)) {
            wrapper.eq(MetricsMetaObject::getObjectKey, objectKey);
        }
        wrapper.orderByAsc(MetricsMetaObject::getObjectType)
                .orderByAsc(MetricsMetaObject::getObjectKey)
                .orderByAsc(MetricsMetaObject::getId);

        List<MetricsMetaObjectVO> all = mapper.selectList(wrapper).stream()
                .map(this::toVO)
                .collect(Collectors.toList());

        return page(all, safePage, safePageSize);
    }

    @Override
    public PageResult<MetricsMetaObjectVO> listByDatasourceAndTypes(
            Long datasourceId,
            Collection<String> objectTypes,
            String objectKey,
            Integer page,
            Integer pageSize) {
        int safePage = page == null || page <= 0 ? 1 : page;
        int safePageSize = pageSize == null || pageSize <= 0 ? 20 : Math.min(pageSize, 200);
        if (objectTypes == null || objectTypes.isEmpty()) {
            throw new IllegalArgumentException("objectTypes is required");
        }
        objectTypes.forEach(this::validateObjectType);

        LambdaQueryWrapper<MetricsMetaObject> wrapper = new LambdaQueryWrapper<MetricsMetaObject>()
                .eq(MetricsMetaObject::getDatasourceId, datasourceId)
                .in(MetricsMetaObject::getObjectType, objectTypes)
                .eq(MetricsMetaObject::getDeleted, 0);
        if (StringUtils.hasText(objectKey)) {
            wrapper.eq(MetricsMetaObject::getObjectKey, objectKey);
        }
        wrapper.orderByAsc(MetricsMetaObject::getObjectKey)
                .orderByAsc(MetricsMetaObject::getObjectType)
                .orderByAsc(MetricsMetaObject::getId);

        List<MetricsMetaObjectVO> all = mapper.selectList(wrapper).stream()
                .map(this::toVO)
                .collect(Collectors.toList());

        return page(all, safePage, safePageSize);
    }

    private PageResult<MetricsMetaObjectVO> page(
            List<MetricsMetaObjectVO> all,
            int safePage,
            int safePageSize) {

        int from = Math.min((safePage - 1) * safePageSize, all.size());
        int to = Math.min(from + safePageSize, all.size());
        return PageResult.<MetricsMetaObjectVO>builder()
                .items(all.subList(from, to))
                .total(all.size())
                .page(safePage)
                .pageSize(safePageSize)
                .build();
    }

    @Override
    public MetricsMetaObjectVO getById(Long id) {
        return toVO(requireMetaObject(id));
    }

    @Override
    @Transactional
    public MetricsMetaObjectVO create(MetricsMetaObjectRequest request) {
        validateRequest(request);
        MetricsMetaObject object = new MetricsMetaObject();
        object.setDatasourceId(request.getDatasourceId());
        object.setObjectType(request.getObjectType());
        object.setObjectKey(request.getObjectKey());
        object.setPayloadJson(toJson(request.getPayload()));
        object.setStatus(request.getStatus());
        object.setDeleted(0);
        mapper.insert(object);
        log.info("METRICS_META_OBJECT_AUDIT action=create id={} datasourceId={} type={} key={}",
                object.getId(), object.getDatasourceId(), object.getObjectType(), object.getObjectKey());
        return toVO(object);
    }

    @Override
    @Transactional
    public MetricsMetaObjectVO update(Long id, MetricsMetaObjectRequest request) {
        validateRequest(request);
        MetricsMetaObject object = requireMetaObject(id);
        object.setDatasourceId(request.getDatasourceId());
        object.setObjectType(request.getObjectType());
        object.setObjectKey(request.getObjectKey());
        object.setPayloadJson(toJson(request.getPayload()));
        object.setStatus(request.getStatus());
        mapper.updateById(object);
        log.info("METRICS_META_OBJECT_AUDIT action=update id={} datasourceId={} type={} key={}",
                object.getId(), object.getDatasourceId(), object.getObjectType(), object.getObjectKey());
        return toVO(object);
    }

    @Override
    @Transactional
    public MetricsMetaObjectVO updateByDatasourceTypeKey(
            Long datasourceId,
            String objectType,
            String objectKey,
            MetricsMetaObjectRequest request) {
        validateRequest(request);
        MetricsMetaObject object = requireMetaObject(datasourceId, objectType, objectKey);
        object.setDatasourceId(datasourceId);
        object.setObjectType(objectType);
        object.setObjectKey(objectKey);
        object.setPayloadJson(toJson(request.getPayload()));
        object.setStatus(request.getStatus());
        mapper.updateById(object);
        log.info("METRICS_META_OBJECT_AUDIT action=update id={} datasourceId={} type={} key={}",
                object.getId(), object.getDatasourceId(), object.getObjectType(), object.getObjectKey());
        return toVO(object);
    }

    @Override
    @Transactional
    public void delete(Long id) {
        requireMetaObject(id);
        mapper.deleteById(id);
        log.info("METRICS_META_OBJECT_AUDIT action=delete id={}", id);
    }

    @Override
    @Transactional
    public void deleteByDatasourceTypeKey(Long datasourceId, String objectType, String objectKey) {
        MetricsMetaObject object = requireMetaObject(datasourceId, objectType, objectKey);
        mapper.deleteById(object.getId());
        log.info("METRICS_META_OBJECT_AUDIT action=delete id={} datasourceId={} type={} key={}",
                object.getId(), datasourceId, objectType, objectKey);
    }

    @Override
    public List<MetricsMetaObjectVO> listActiveForOverlay(String objectType, Long datasourceId) {
        validateObjectType(objectType);
        try {
            List<MetricsMetaObjectVO> objects = new ArrayList<>();
            objects.addAll(selectActive(objectType, null));
            if (datasourceId != null) {
                objects.addAll(selectActive(objectType, datasourceId));
            }
            return objects;
        } catch (Exception e) {
            log.warn("Metrics meta object overlay unavailable type={} datasourceId={}: {}",
                    objectType, datasourceId, e.getMessage());
            return List.of();
        }
    }

    private List<MetricsMetaObjectVO> selectActive(String objectType, Long datasourceId) {
        LambdaQueryWrapper<MetricsMetaObject> wrapper = new LambdaQueryWrapper<MetricsMetaObject>()
                .eq(MetricsMetaObject::getObjectType, objectType)
                .eq(MetricsMetaObject::getStatus, 1)
                .eq(MetricsMetaObject::getDeleted, 0)
                .orderByAsc(MetricsMetaObject::getId);
        if (datasourceId == null) {
            wrapper.isNull(MetricsMetaObject::getDatasourceId);
        } else {
            wrapper.eq(MetricsMetaObject::getDatasourceId, datasourceId);
        }
        return mapper.selectList(wrapper).stream().map(this::toVO).collect(Collectors.toList());
    }

    private MetricsMetaObject requireMetaObject(Long id) {
        MetricsMetaObject object = mapper.selectOne(
                new LambdaQueryWrapper<MetricsMetaObject>()
                        .eq(MetricsMetaObject::getId, id)
                        .eq(MetricsMetaObject::getDeleted, 0));
        if (object == null) {
            throw new IllegalArgumentException("Metrics meta object not found: id=" + id);
        }
        return object;
    }

    private MetricsMetaObject requireMetaObject(Long datasourceId, String objectType, String objectKey) {
        validateObjectType(objectType);
        MetricsMetaObject object = mapper.selectOne(
                new LambdaQueryWrapper<MetricsMetaObject>()
                        .eq(MetricsMetaObject::getDatasourceId, datasourceId)
                        .eq(MetricsMetaObject::getObjectType, objectType)
                        .eq(MetricsMetaObject::getObjectKey, objectKey)
                        .eq(MetricsMetaObject::getDeleted, 0));
        if (object == null) {
            throw new IllegalArgumentException("Metrics meta object not found: datasourceId="
                    + datasourceId + " objectType=" + objectType + " objectKey=" + objectKey);
        }
        return object;
    }

    private void validateRequest(MetricsMetaObjectRequest request) {
        validateObjectType(request.getObjectType());
        if (request.getPayload() == null
                || request.getPayload().isMissingNode()
                || request.getPayload().isValueNode()) {
            throw new IllegalArgumentException("payload must be a JSON object or array");
        }
        if (request.getStatus() != 0 && request.getStatus() != 1) {
            throw new IllegalArgumentException("status must be 0 or 1");
        }
    }

    private void validateObjectType(String objectType) {
        if (!MetricsMetaObjectTypes.SUPPORTED.contains(objectType)) {
            throw new IllegalArgumentException("Unsupported metrics meta object type: " + objectType);
        }
    }

    private String toJson(JsonNode payload) {
        try {
            return MAPPER.writeValueAsString(payload);
        } catch (Exception e) {
            throw new IllegalArgumentException("payload must be valid JSON: " + e.getMessage());
        }
    }

    private MetricsMetaObjectVO toVO(MetricsMetaObject object) {
        MetricsMetaObjectVO vo = new MetricsMetaObjectVO();
        BeanUtils.copyProperties(object, vo, "payloadJson");
        try {
            vo.setPayload(StringUtils.hasText(object.getPayloadJson())
                    ? MAPPER.readTree(object.getPayloadJson())
                    : MAPPER.createObjectNode());
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Invalid payload_json for metrics meta object id=" + object.getId(), e);
        }
        return vo;
    }
}
