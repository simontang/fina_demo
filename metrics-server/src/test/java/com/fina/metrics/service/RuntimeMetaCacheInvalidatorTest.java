package com.fina.metrics.service;

import com.fina.metrics.dto.RuntimeMetaChangeState;
import com.fina.metrics.mapper.RuntimeMetaChangeMapper;
import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;
import java.util.List;

import static org.mockito.Mockito.*;

class RuntimeMetaCacheInvalidatorTest {

    @Test
    void invalidatesOnlyWhenPolledStateChanges() {
        RuntimeMetaChangeMapper mapper = mock(RuntimeMetaChangeMapper.class);
        RuntimeMetaCache cache = mock(RuntimeMetaCache.class);
        RuntimeMetaCacheInvalidator invalidator = new RuntimeMetaCacheInvalidator(mapper, cache);
        List<RuntimeMetaChangeState> initial = List.of(state(1L, LocalDateTime.parse("2026-09-05T10:00:00"), "hash-a"));
        List<RuntimeMetaChangeState> changed = List.of(state(1L, LocalDateTime.parse("2026-09-05T10:00:00"), "hash-b"));
        when(mapper.selectRuntimeMetaChangeState()).thenReturn(initial, initial, changed);

        invalidator.pollForChanges();
        verify(cache).invalidateAll("meta change baseline initialized");
        clearInvocations(cache);

        invalidator.pollForChanges();
        verifyNoInteractions(cache);

        invalidator.pollForChanges();
        verify(cache).invalidateAll("published meta changed");
    }

    @Test
    void clearsCacheWhenPollingRecoversWithoutBaseline() {
        RuntimeMetaChangeMapper mapper = mock(RuntimeMetaChangeMapper.class);
        RuntimeMetaCache cache = mock(RuntimeMetaCache.class);
        RuntimeMetaCacheInvalidator invalidator = new RuntimeMetaCacheInvalidator(mapper, cache);
        when(mapper.selectRuntimeMetaChangeState())
                .thenThrow(new IllegalStateException("database unavailable"))
                .thenReturn(List.of(state(1L, LocalDateTime.parse("2026-09-05T10:00:00"), "hash-a")));

        invalidator.pollForChanges();
        verifyNoInteractions(cache);

        invalidator.pollForChanges();

        verify(cache).invalidateAll("meta change baseline initialized");
    }

    private RuntimeMetaChangeState state(Long activeCount, LocalDateTime updatedAt, String contentFingerprint) {
        RuntimeMetaChangeState state = new RuntimeMetaChangeState();
        state.setSourceName("metrics_meta_object");
        state.setTotalCount(activeCount);
        state.setActiveCount(activeCount);
        state.setMaxId(activeCount);
        state.setMaxUpdatedAt(updatedAt);
        state.setContentFingerprint(contentFingerprint);
        return state;
    }
}
