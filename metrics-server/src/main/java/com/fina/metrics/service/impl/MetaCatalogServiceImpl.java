package com.fina.metrics.service.impl;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fina.metrics.dto.MetricsMetaObjectVO;
import com.fina.metrics.service.MetaCatalogService;
import com.fina.metrics.service.MetricsMetaObjectService;
import com.fina.metrics.service.MetricsMetaObjectTypes;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Loads metrics-index-meta.json and metrics-detail-meta.json from the classpath
 * once at startup and keeps them in memory for fast lookup.
 *
 * Path: src/main/resources/meta/
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class MetaCatalogServiceImpl implements MetaCatalogService {

    private static final String INDEX_PATH  = "meta/metrics-index-meta.json";
    private static final String DETAIL_PATH = "meta/metrics-detail-meta.json";

    private final ObjectMapper mapper = new ObjectMapper();
    private final MetricsMetaObjectService metaObjectService;

    private String catalogVersion = "1.0";
    private List<String> domainCategories = Collections.emptyList();
    private List<JsonNode> indexItems = Collections.emptyList();

    /** metric_name → index JsonNode */
    private final Map<String, JsonNode> indexMap  = new ConcurrentHashMap<>();
    /** metric_name → detail JsonNode */
    private final Map<String, JsonNode> detailMap = new ConcurrentHashMap<>();

    @PostConstruct
    public void init() {
        loadIndex();
        loadDetail();
        log.info("MetaCatalog loaded: {} index items, {} detail items",
                indexItems.size(), detailMap.size());
    }

    @Override
    public List<JsonNode> getIndexItems() {
        return getIndexItems(null);
    }

    @Override
    public List<JsonNode> getIndexItems(Long datasourceId) {
        return Collections.unmodifiableList(overlayMetricObjects(
                indexItems,
                MetricsMetaObjectTypes.METRIC_INDEX,
                "metric_name",
                datasourceId));
    }

    @Override
    public List<JsonNode> getDetailItems(Long datasourceId) {
        List<JsonNode> base = new ArrayList<>(detailMap.values());
        return Collections.unmodifiableList(overlayMetricObjects(
                base,
                MetricsMetaObjectTypes.METRIC_DETAIL,
                "metric_name",
                datasourceId));
    }

    @Override
    public String getCatalogVersion() {
        return getCatalogVersion(null);
    }

    @Override
    public String getCatalogVersion(Long datasourceId) {
        JsonNode config = getCatalogConfig(datasourceId);
        if (config == null) {
            return catalogVersion;
        }
        String camel = config.path("catalogVersion").asText(null);
        if (StringUtils.hasText(camel)) {
            return camel;
        }
        String snake = config.path("metric_catalog_version").asText(null);
        return StringUtils.hasText(snake) ? snake : catalogVersion;
    }

    @Override
    public List<String> getDomainCategories() {
        return getDomainCategories(null);
    }

    @Override
    public List<String> getDomainCategories(Long datasourceId) {
        JsonNode config = getCatalogConfig(datasourceId);
        if (config == null) {
            return Collections.unmodifiableList(domainCategories);
        }
        JsonNode categories = config.has("domainCategories")
                ? config.get("domainCategories")
                : config.get("domain_categories");
        if (categories == null || !categories.isArray()) {
            return Collections.unmodifiableList(domainCategories);
        }
        List<String> merged = new ArrayList<>();
        categories.forEach(item -> merged.add(item.asText()));
        return Collections.unmodifiableList(merged);
    }

    @Override
    public Optional<JsonNode> findIndexItem(String metricName) {
        return findIndexItem(metricName, null);
    }

    @Override
    public Optional<JsonNode> findIndexItem(String metricName, Long datasourceId) {
        if (metricName == null) return Optional.empty();
        return getIndexItems(datasourceId).stream()
                .filter(item -> metricName.equals(item.path("metric_name").asText(null)))
                .findFirst()
                .or(() -> Optional.ofNullable(indexMap.get(metricName)));
    }

    @Override
    public Optional<JsonNode> findDetailItem(String metricName) {
        return findDetailItem(metricName, null);
    }

    @Override
    public Optional<JsonNode> findDetailItem(String metricName, Long datasourceId) {
        if (metricName == null) return Optional.empty();
        return getDetailItems(datasourceId).stream()
                .filter(item -> metricName.equals(item.path("metric_name").asText(null)))
                .findFirst()
                .or(() -> Optional.ofNullable(detailMap.get(metricName)));
    }

    // ─── private loaders ─────────────────────────────────────────────────────

    private void loadIndex() {
        try (InputStream is = new ClassPathResource(INDEX_PATH).getInputStream()) {
            JsonNode root = mapper.readTree(is);

            if (root.has("metric_catalog_version")) {
                catalogVersion = root.get("metric_catalog_version").asText("1.0");
            }

            if (root.has("domain_categories")) {
                List<String> cats = new ArrayList<>();
                root.get("domain_categories").forEach(n -> cats.add(n.asText()));
                domainCategories = cats;
            }

            List<JsonNode> items = new ArrayList<>();
            if (root.has("metrics_index")) {
                root.get("metrics_index").forEach(item -> {
                    items.add(item);
                    String name = item.path("metric_name").asText(null);
                    if (name != null) {
                        indexMap.put(name, item);
                    }
                });
            }
            indexItems = items;
        } catch (Exception e) {
            log.error("Failed to load metrics index catalog from {}: {}", INDEX_PATH, e.getMessage());
        }
    }

    private void loadDetail() {
        try (InputStream is = new ClassPathResource(DETAIL_PATH).getInputStream()) {
            JsonNode root = mapper.readTree(is);
            if (root.isArray()) {
                root.forEach(item -> {
                    String name = item.path("metric_name").asText(null);
                    if (name != null) {
                        detailMap.put(name, item);
                    }
                });
            }
        } catch (Exception e) {
            log.error("Failed to load metrics detail catalog from {}: {}", DETAIL_PATH, e.getMessage());
        }
    }

    private JsonNode getCatalogConfig(Long datasourceId) {
        List<MetricsMetaObjectVO> configs = metaObjectService.listActiveForOverlay(
                MetricsMetaObjectTypes.CATALOG_CONFIG,
                datasourceId);
        if (configs.isEmpty()) {
            return null;
        }
        return configs.get(configs.size() - 1).getPayload();
    }

    private List<JsonNode> overlayMetricObjects(
            List<JsonNode> baseItems,
            String objectType,
            String keyField,
            Long datasourceId) {
        Map<String, JsonNode> merged = new LinkedHashMap<>();
        for (JsonNode item : baseItems) {
            String key = item.path(keyField).asText(null);
            if (StringUtils.hasText(key)) {
                merged.put(key, item);
            }
        }
        for (MetricsMetaObjectVO object : metaObjectService.listActiveForOverlay(objectType, datasourceId)) {
            JsonNode payload = normalizePayloadObject(object.getPayload(), keyField, object.getObjectKey());
            if (payload == null || !payload.isObject()) {
                continue;
            }
            String key = payload.path(keyField).asText(object.getObjectKey());
            if (StringUtils.hasText(key)) {
                merged.put(key, payload);
            }
        }
        return new ArrayList<>(merged.values());
    }

    private JsonNode normalizePayloadObject(JsonNode payload, String keyField, String objectKey) {
        if (payload == null || !payload.isObject()) {
            return payload;
        }
        ObjectNode copy = payload.deepCopy();
        if (!StringUtils.hasText(copy.path(keyField).asText(null))) {
            copy.put(keyField, objectKey);
        }
        return copy;
    }
}
