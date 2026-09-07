package com.fina.metrics.util;

import org.junit.jupiter.api.Test;

import java.sql.Array;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class JdbcValueNormalizerTest {

    @Test
    void convertsJdbcArrayToJsonFriendlyList() throws Exception {
        Array array = mock(Array.class);
        when(array.getArray()).thenReturn(new Integer[]{1, 2, 3});

        Object normalized = JdbcValueNormalizer.normalize(array);

        assertThat(normalized).isEqualTo(java.util.List.of(1, 2, 3));
    }

    @Test
    void convertsNestedJavaArraysToLists() throws Exception {
        Object normalized = JdbcValueNormalizer.normalize(new Object[]{
                "a",
                new int[]{1, 2}
        });

        assertThat(normalized).isEqualTo(java.util.List.of("a", java.util.List.of(1, 2)));
    }
}
