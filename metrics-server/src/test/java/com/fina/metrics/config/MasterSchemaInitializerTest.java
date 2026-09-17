package com.fina.metrics.config;

import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.DependsOn;
import org.springframework.test.util.ReflectionTestUtils;

import javax.sql.DataSource;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;

class MasterSchemaInitializerTest {

    @Test
    void disabledInitializerDoesNotAccessDatabase() {
        DataSource dataSource = mock(DataSource.class);
        MasterSchemaInitializer initializer = new MasterSchemaInitializer(dataSource);
        ReflectionTestUtils.setField(initializer, "enabled", false);

        initializer.init();

        verifyNoInteractions(dataSource);
    }

    @Test
    void dynamicDatasourceLoadingDependsOnSchemaMigration() {
        assertThat(DynamicDataSourceManager.class.getAnnotation(DependsOn.class).value())
                .contains("masterSchemaInitializer");
    }
}
