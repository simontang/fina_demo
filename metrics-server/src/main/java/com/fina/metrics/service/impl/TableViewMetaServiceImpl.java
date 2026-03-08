package com.fina.metrics.service.impl;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.metrics.dto.TableViewDetailResponse;
import com.fina.metrics.dto.TableViewIndexItem;
import com.fina.metrics.service.TableViewMetaService;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.ResourcePatternResolver;
import org.springframework.stereotype.Service;
import org.springframework.util.StreamUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Loads Table/View meta at startup from three classpath sources:
 *
 *   1. meta/table-catalog.json  — master catalog: tableName, docType, docTypeEn, shortDesc
 *                                  for ALL known tables/views (provides AI-friendly descriptions)
 *   2. meta/view-*.json          — per-view detail: selectSql + typed columns (takes precedence
 *                                  over CSV for column data)
 *   3. meta/MTC_VW_AI_*.csv      — column list for tables not covered by any view JSON
 *
 * Merge priority for index-level fields:  table-catalog.json first, then view-*.json if absent.
 * Merge priority for column data:         view-*.json first, then CSV.
 */
@Slf4j
@Service
public class TableViewMetaServiceImpl implements TableViewMetaService {

    private static final String CATALOG_PATH = "classpath:meta/table-catalog.json";
    private static final String VIEW_PATTERN = "classpath*:meta/view-*.json";
    private static final String CSV_PATTERN  = "classpath*:meta/MTC_VW_AI_*.csv";
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final ResourcePatternResolver resourceResolver;

    private List<TableViewIndexItem>   indexList  = Collections.emptyList();
    private List<TableViewDetailResponse> detailList = Collections.emptyList();

    public TableViewMetaServiceImpl(ResourcePatternResolver resourceResolver) {
        this.resourceResolver = resourceResolver;
    }

    @PostConstruct
    public void init() {
        // Step 1: load master catalog for display metadata (docType, shortDesc …)
        Map<String, CatalogEntry> catalogMap = loadTableCatalog();

        // Step 2: build working maps using catalog as the base
        Map<String, TableViewIndexItem.TableViewIndexItemBuilder>   idxBuilders    = new LinkedHashMap<>();
        Map<String, TableViewDetailResponse.TableViewDetailResponseBuilder> detBuilders = new LinkedHashMap<>();

        // Seed with catalog order so output order matches catalog
        for (Map.Entry<String, CatalogEntry> e : catalogMap.entrySet()) {
            String name = e.getKey();
            CatalogEntry cat = e.getValue();
            idxBuilders.put(name, TableViewIndexItem.builder()
                    .tableName(name)
                    .displayName(firstNonBlank(cat.docTypeEn, cat.docType, name))
                    .docType(cat.docType)
                    .docTypeEn(cat.docTypeEn)
                    .shortDesc(cat.shortDesc));
            detBuilders.put(name, TableViewDetailResponse.builder()
                    .tableName(name)
                    .docType(cat.docType)
                    .docTypeEn(cat.docTypeEn));
        }

        // Step 3: enrich from view-*.json (adds selectSql, columns, mainTable, lineTable)
        loadViewJson(catalogMap, idxBuilders, detBuilders);

        // Step 4: fill column data from CSV for tables still missing columns
        loadCsvOnly(idxBuilders, detBuilders);

        // Step 5: finalise — build any entries that came from view/CSV but not in catalog
        List<TableViewIndexItem>   idx = new ArrayList<>();
        List<TableViewDetailResponse> det = new ArrayList<>();
        for (String name : idxBuilders.keySet()) {
            idx.add(idxBuilders.get(name).build());
            det.add(detBuilders.get(name).build());
        }
        indexList  = Collections.unmodifiableList(idx);
        detailList = Collections.unmodifiableList(det);
        log.info("TableViewMeta loaded: {} tables/views", indexList.size());
    }

    @Override
    public List<TableViewIndexItem> getTableViewsIndex() {
        return indexList;
    }

    @Override
    public List<TableViewDetailResponse> getTableViewsDetails() {
        return detailList;
    }

    // ── Loaders ───────────────────────────────────────────────────────────────

    private Map<String, CatalogEntry> loadTableCatalog() {
        Map<String, CatalogEntry> map = new LinkedHashMap<>();
        try {
            Resource res = resourceResolver.getResource(CATALOG_PATH);
            if (!res.exists()) {
                log.warn("table-catalog.json not found at {}", CATALOG_PATH);
                return map;
            }
            String json = StreamUtils.copyToString(res.getInputStream(), StandardCharsets.UTF_8);
            JsonNode root = MAPPER.readTree(json);
            if (root.isArray()) {
                for (JsonNode node : root) {
                    String tableName = node.path("tableName").asText(null);
                    if (tableName == null || tableName.isBlank()) continue;
                    map.put(tableName, new CatalogEntry(
                            node.path("docType").asText(null),
                            node.path("docTypeEn").asText(null),
                            node.path("shortDesc").asText(null)
                    ));
                }
            }
        } catch (Exception e) {
            log.error("Failed to load table-catalog.json: {}", e.getMessage());
        }
        log.debug("table-catalog loaded {} entries", map.size());
        return map;
    }

    private void loadViewJson(
            Map<String, CatalogEntry> catalogMap,
            Map<String, TableViewIndexItem.TableViewIndexItemBuilder> idxBuilders,
            Map<String, TableViewDetailResponse.TableViewDetailResponseBuilder> detBuilders) {
        try {
            Resource[] resources = resourceResolver.getResources(VIEW_PATTERN);
            for (Resource resource : resources) {
                if (!resource.isReadable()) continue;
                try {
                    String json = StreamUtils.copyToString(resource.getInputStream(), StandardCharsets.UTF_8);
                    JsonNode root = MAPPER.readTree(json);
                    String viewName = root.path("viewName").asText(null);
                    if (viewName == null || viewName.isBlank()) continue;

                    String mainTable  = root.path("mainTable").asText(null);
                    String lineTable  = root.path("lineTable").asText(null);
                    int objTypeCode   = root.path("objTypeCode").asInt(0);
                    String selectSql  = root.has("selectSql") ? root.get("selectSql").asText(null) : null;

                    List<TableViewDetailResponse.ColumnMeta> columns = new ArrayList<>();
                    if (root.has("columns") && root.get("columns").isArray()) {
                        for (JsonNode col : root.get("columns")) {
                            columns.add(TableViewDetailResponse.ColumnMeta.builder()
                                    .name(col.path("name").asText(null))
                                    .label(col.path("label").asText(null))
                                    .example(col.path("example").asText(null))
                                    .build());
                        }
                    }

                    // Fallback display fields from view JSON when catalog has no entry
                    CatalogEntry cat = catalogMap.getOrDefault(viewName, new CatalogEntry(
                            root.path("docType").asText(null),
                            root.path("docTypeEn").asText(null),
                            null));

                    // Ensure builders exist (catalog may not list all views)
                    idxBuilders.computeIfAbsent(viewName, k -> TableViewIndexItem.builder()
                            .tableName(k)
                            .displayName(firstNonBlank(cat.docTypeEn, cat.docType, k))
                            .docType(cat.docType)
                            .docTypeEn(cat.docTypeEn)
                            .shortDesc(cat.shortDesc));
                    detBuilders.computeIfAbsent(viewName, k -> TableViewDetailResponse.builder()
                            .tableName(k)
                            .docType(cat.docType)
                            .docTypeEn(cat.docTypeEn));

                    idxBuilders.get(viewName)
                            .mainTable(mainTable)
                            .lineTable(lineTable)
                            .columnCount(columns.size());

                    detBuilders.get(viewName)
                            .objTypeCode(objTypeCode)
                            .mainTable(mainTable)
                            .lineTable(lineTable)
                            .selectSql(selectSql)
                            .columns(columns);

                } catch (Exception e) {
                    log.warn("Failed to load view meta from {}: {}", resource.getFilename(), e.getMessage());
                }
            }
        } catch (Exception e) {
            log.error("Failed to scan view-*.json: {}", e.getMessage());
        }
    }

    private void loadCsvOnly(
            Map<String, TableViewIndexItem.TableViewIndexItemBuilder> idxBuilders,
            Map<String, TableViewDetailResponse.TableViewDetailResponseBuilder> detBuilders) {
        try {
            Resource[] resources = resourceResolver.getResources(CSV_PATTERN);
            for (Resource resource : resources) {
                if (!resource.isReadable()) continue;
                String filename = resource.getFilename();
                if (filename == null || !filename.endsWith(".csv")) continue;
                String tableName = filename.substring(0, filename.length() - 4);

                // Skip if view-*.json already provided columns (check builder has columnCount set)
                TableViewDetailResponse.TableViewDetailResponseBuilder detBuilder = detBuilders.get(tableName);
                if (detBuilder != null) {
                    // Already has data from view-*.json or catalog — only fill columns if missing
                    TableViewDetailResponse built = detBuilder.build();
                    if (built.getColumns() != null && !built.getColumns().isEmpty()) continue;
                }

                try {
                    List<TableViewDetailResponse.ColumnMeta> columns = parseCsvColumns(resource);
                    if (columns.isEmpty()) continue;

                    idxBuilders.computeIfAbsent(tableName, k -> TableViewIndexItem.builder()
                            .tableName(k)
                            .displayName(k));
                    detBuilders.computeIfAbsent(tableName, k -> TableViewDetailResponse.builder()
                            .tableName(k));

                    idxBuilders.get(tableName).columnCount(columns.size());
                    detBuilders.get(tableName).columns(columns);
                } catch (Exception e) {
                    log.warn("Failed to load CSV meta from {}: {}", filename, e.getMessage());
                }
            }
        } catch (Exception e) {
            log.error("Failed to scan MTC_VW_AI_*.csv: {}", e.getMessage());
        }
    }

    // ── CSV parser ────────────────────────────────────────────────────────────

    private List<TableViewDetailResponse.ColumnMeta> parseCsvColumns(Resource resource) throws IOException {
        List<TableViewDetailResponse.ColumnMeta> columns = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8))) {
            String line = reader.readLine();
            if (line == null) return columns;
            line = stripBom(line);
            if (!line.contains("\t")) return columns;
            String[] header = line.split("\t", -1);
            int nameIdx = -1, descIdx = -1, typeIdx = -1;
            for (int i = 0; i < header.length; i++) {
                String h = header[i].trim();
                if ("字段名".equals(h) || "Field Name".equals(h))  nameIdx = i;
                else if ("描述".equals(h) || "Description".equals(h)) descIdx = i;
                else if ("类型".equals(h) || "Type".equals(h)) typeIdx = i;
            }
            if (nameIdx < 0) return columns;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split("\t", -1);
                String name = nameIdx < parts.length ? parts[nameIdx].trim() : "";
                String desc = descIdx >= 0 && descIdx < parts.length ? parts[descIdx].trim() : null;
                String type = typeIdx >= 0 && typeIdx < parts.length ? parts[typeIdx].trim() : null;
                if (name.isEmpty()) continue;
                columns.add(TableViewDetailResponse.ColumnMeta.builder()
                        .name(name).label(desc).description(desc).type(type).build());
            }
        }
        return columns;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static String stripBom(String s) {
        return (s != null && s.startsWith("\uFEFF")) ? s.substring(1) : s;
    }

    private static String firstNonBlank(String... values) {
        for (String v : values) {
            if (v != null && !v.isBlank()) return v;
        }
        return "";
    }

    private record CatalogEntry(String docType, String docTypeEn, String shortDesc) {}
}
