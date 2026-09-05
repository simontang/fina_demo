package com.fina.metrics.service;

import com.fina.metrics.dto.RuntimeMetaChangeState;
import com.fina.metrics.mapper.RuntimeMetaChangeMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(
        name = "metrics.meta-cache.enabled",
        havingValue = "true",
        matchIfMissing = true)
public class RuntimeMetaCacheInvalidator {

    private final RuntimeMetaChangeMapper changeMapper;
    private final RuntimeMetaCache runtimeMetaCache;

    private volatile String lastFingerprint;

    @Scheduled(
            fixedDelayString = "${metrics.meta-cache.poll-interval-ms:60000}",
            initialDelayString = "${metrics.meta-cache.initial-delay-ms:60000}")
    void pollForChanges() {
        try {
            String currentFingerprint = readFingerprint();
            String previousFingerprint = lastFingerprint;
            lastFingerprint = currentFingerprint;

            if (previousFingerprint == null) {
                log.info("Runtime meta change baseline initialized");
            } else if (!previousFingerprint.equals(currentFingerprint)) {
                runtimeMetaCache.invalidateAll("published meta changed");
            }
        } catch (Exception e) {
            log.warn("Unable to poll runtime meta changes; keeping existing cache: {}", e.getMessage());
        }
    }

    private String readFingerprint() {
        List<RuntimeMetaChangeState> states = changeMapper.selectRuntimeMetaChangeState();
        if (states == null || states.isEmpty()) {
            throw new IllegalStateException("Runtime meta change query returned no state");
        }
        return states.stream()
                .sorted(Comparator.comparing(RuntimeMetaChangeState::getSourceName))
                .map(this::fingerprintPart)
                .collect(Collectors.joining("|"));
    }

    private String fingerprintPart(RuntimeMetaChangeState state) {
        return String.join(":",
                state.getSourceName(),
                String.valueOf(state.getTotalCount()),
                String.valueOf(state.getActiveCount()),
                String.valueOf(state.getMaxId()),
                String.valueOf(state.getMaxUpdatedAt()),
                String.valueOf(state.getContentFingerprint()));
    }
}
