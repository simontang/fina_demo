package com.fina.b1s.config;

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
    private boolean enabled;

    @jakarta.annotation.PostConstruct
    public void init() {
        if (!enabled) {
            log.info("Master schema initialization disabled");
            return;
        }
        ClassPathResource schema = new ClassPathResource("sql/init.sql");
        if (!schema.exists()) {
            log.warn("Master schema init script not found: sql/init.sql");
            return;
        }
        ResourceDatabasePopulator populator = new ResourceDatabasePopulator(schema);
        populator.setContinueOnError(false);
        populator.execute(dataSource);
        log.info("Master schema initialization completed");
    }
}
