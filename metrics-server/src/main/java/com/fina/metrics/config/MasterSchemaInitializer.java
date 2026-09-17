package com.fina.metrics.config;

import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.datasource.init.ResourceDatabasePopulator;
import org.springframework.stereotype.Component;

import javax.sql.DataSource;

@Slf4j
@Component
@RequiredArgsConstructor
public class MasterSchemaInitializer {

    private final DataSource dataSource;

    @Value("${metrics.master-schema-init:true}")
    private boolean enabled = true;

    @PostConstruct
    public void init() {
        if (!enabled) {
            log.info("Master schema initialization disabled");
            return;
        }
        ResourceDatabasePopulator populator = new ResourceDatabasePopulator(
                new ClassPathResource("sql/init.sql"),
                new ClassPathResource("sql/datasource_visible_scope_migration.sql"));
        populator.setContinueOnError(false);
        populator.execute(dataSource);
        log.info("Master schema initialization and datasource visibility migration completed");
    }
}
