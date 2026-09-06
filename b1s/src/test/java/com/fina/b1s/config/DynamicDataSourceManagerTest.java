package com.fina.b1s.config;

import com.fina.b1s.entity.DataSourceConfig;
import com.fina.b1s.mapper.DataSourceConfigMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.Mockito.mock;

class DynamicDataSourceManagerTest {

    private DynamicDataSourceManager manager;

    @BeforeEach
    void setUp() {
        manager = new DynamicDataSourceManager(mock(DataSourceConfigMapper.class));
        ReflectionTestUtils.setField(manager, "datasourceType", "SQLSERVER");
    }

    @Test
    void registerDataSourceSkipsPostgresUrlEvenWhenLegacyInstanceTypeIsSqlServer() {
        DataSourceConfig config = new DataSourceConfig();
        config.setId(15L);
        config.setName("Hankel PostgreSQL");
        config.setInstanceType("SQLSERVER");
        config.setUrl("jdbc:postgresql://postgres.example:5432/postgres");

        manager.registerDataSource(config);

        assertFalse(manager.isRegistered(15L));
    }
}
