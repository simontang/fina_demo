package com.fina.metrics.service;

import com.fina.metrics.dto.MetricsMetaFullResponse;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;
import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;

import static org.assertj.core.api.Assertions.assertThat;

class RuntimeMetaCacheTest {

    @Test
    void cachesRuntimeMetaByDatasource() {
        RuntimeMetaCache cache = new RuntimeMetaCache();
        AtomicInteger loadCount = new AtomicInteger();
        MetricsMetaFullResponse expected = response();

        MetricsMetaFullResponse first = cache.get(15L, () -> {
            loadCount.incrementAndGet();
            return expected;
        });
        MetricsMetaFullResponse second = cache.get(15L, () -> {
            loadCount.incrementAndGet();
            return response();
        });

        assertThat(first).isSameAs(expected);
        assertThat(second).isSameAs(expected);
        assertThat(loadCount).hasValue(1);
        assertThat(cache.size()).isEqualTo(1);
    }

    @Test
    void reloadsRuntimeMetaAfterInvalidation() {
        RuntimeMetaCache cache = new RuntimeMetaCache();
        AtomicInteger loadCount = new AtomicInteger();

        MetricsMetaFullResponse first = cache.get(15L, () -> {
            loadCount.incrementAndGet();
            return response();
        });
        cache.invalidateAll("test");
        MetricsMetaFullResponse second = cache.get(15L, () -> {
            loadCount.incrementAndGet();
            return response();
        });

        assertThat(second).isNotSameAs(first);
        assertThat(loadCount).hasValue(2);
    }

    private MetricsMetaFullResponse response() {
        return MetricsMetaFullResponse.builder().build();
    }

    @Test
    void datasourceInvalidationKeepsOtherEntriesAndWaitsForCommit() {
        RuntimeMetaCache cache = new RuntimeMetaCache();
        MetricsMetaFullResponse first = cache.get(15L, this::response);
        MetricsMetaFullResponse other = cache.get(16L, this::response);
        TransactionSynchronizationManager.initSynchronization();
        try {
            cache.invalidateDatasourceAfterCommit(15L, "scope changed");
            assertThat(cache.get(15L, this::response)).isSameAs(first);
            TransactionSynchronizationManager.getSynchronizations().forEach(TransactionSynchronization::afterCommit);
            assertThat(cache.get(15L, this::response)).isNotSameAs(first);
            assertThat(cache.get(16L, this::response)).isSameAs(other);
        } finally {
            TransactionSynchronizationManager.clearSynchronization();
        }
    }

    @Test
    void invalidationDuringLoadDoesNotCacheOldMeta() {
        RuntimeMetaCache cache = new RuntimeMetaCache();
        cache.get(15L, () -> {
            cache.invalidateDatasource(15L, "scope changed during load");
            return response();
        });
        assertThat(cache.size()).isZero();
    }
}
