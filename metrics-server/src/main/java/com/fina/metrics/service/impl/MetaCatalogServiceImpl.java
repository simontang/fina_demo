package com.fina.metrics.service.impl;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.metrics.service.MetaCatalogService;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
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
public class MetaCatalogServiceImpl implements MetaCatalogService {

    private static final String INDEX_PATH  = "meta/metrics-index-meta.json";
    private static final String DETAIL_PATH = "meta/metrics-detail-meta.json";

    private final ObjectMapper mapper = new ObjectMapper();

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
        return Collections.unmodifiableList(indexItems);
    }

    @Override
    public String getCatalogVersion() {
        return catalogVersion;
    }

    @Override
    public List<String> getDomainCategories() {
        return Collections.unmodifiableList(domainCategories);
    }

    @Override
    public Optional<JsonNode> findIndexItem(String metricName) {
        if (metricName == null) return Optional.empty();
        return Optional.ofNullable(indexMap.get(metricName));
    }

    @Override
    public Optional<JsonNode> findDetailItem(String metricName) {
        if (metricName == null) return Optional.empty();
        return Optional.ofNullable(detailMap.get(metricName));
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
}
