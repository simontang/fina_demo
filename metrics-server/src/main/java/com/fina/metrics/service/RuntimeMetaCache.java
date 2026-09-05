package com.fina.metrics.service;

import com.fina.metrics.dto.MetricsMetaFullResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

@Slf4j
@Component
public class RuntimeMetaCache {

    // Runtime grants are datasource-scoped, so the datasource is the complete cache scope.
    private final ConcurrentMap<Long, MetricsMetaFullResponse> entries = new ConcurrentHashMap<>();
    private final AtomicLong generation = new AtomicLong();

    @Value("${metrics.meta-cache.enabled:true}")
    private boolean enabled = true;

    public MetricsMetaFullResponse get(
            Long datasourceId,
            Supplier<MetricsMetaFullResponse> loader) {
        if (!enabled) {
            return loader.get();
        }
        MetricsMetaFullResponse cached = entries.get(datasourceId);
        if (cached != null) {
            log.debug("Runtime meta cache hit datasource={}", datasourceId);
            return cached;
        }

        long loadGeneration = generation.get();
        MetricsMetaFullResponse loaded = loader.get();
        if (loadGeneration != generation.get()) {
            return loaded;
        }

        MetricsMetaFullResponse existing = entries.putIfAbsent(datasourceId, loaded);
        log.debug("Runtime meta cache miss datasource={}", datasourceId);
        return existing != null ? existing : loaded;
    }

    public void invalidateAll(String reason) {
        generation.incrementAndGet();
        int entryCount = entries.size();
        entries.clear();
        log.info("Runtime meta cache invalidated reason={} entries={}", reason, entryCount);
    }

    int size() {
        return entries.size();
    }
}
