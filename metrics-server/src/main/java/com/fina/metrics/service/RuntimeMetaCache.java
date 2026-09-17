package com.fina.metrics.service;

import com.fina.metrics.dto.MetricsMetaFullResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;

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
        synchronized (this) {
            if (loadGeneration != generation.get()) {
                return loaded;
            }
            MetricsMetaFullResponse existing = entries.putIfAbsent(datasourceId, loaded);
            log.debug("Runtime meta cache miss datasource={}", datasourceId);
            return existing != null ? existing : loaded;
        }
    }

    public synchronized void invalidateAll(String reason) {
        generation.incrementAndGet();
        int entryCount = entries.size();
        entries.clear();
        log.info("Runtime meta cache invalidated reason={} entries={}", reason, entryCount);
    }

    public synchronized void invalidateDatasource(Long datasourceId, String reason) {
        generation.incrementAndGet();
        entries.remove(datasourceId);
        log.info("Runtime meta cache invalidated datasource={} reason={}", datasourceId, reason);
    }

    public void invalidateDatasourceAfterCommit(Long datasourceId, String reason) {
        if (TransactionSynchronizationManager.isSynchronizationActive()) {
            TransactionSynchronizationManager.registerSynchronization(new TransactionSynchronization() {
                @Override
                public void afterCommit() {
                    invalidateDatasource(datasourceId, reason);
                }
            });
        } else {
            invalidateDatasource(datasourceId, reason);
        }
    }

    int size() {
        return entries.size();
    }
}
