package com.fina.platform.exception;

import org.junit.jupiter.api.Test;
import org.jooq.exception.DataAccessException;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.sql.SQLException;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class GlobalExceptionHandlerTest {

    private final GlobalExceptionHandler handler = new GlobalExceptionHandler();

    @RestController
    static class ThrowingController {
        @GetMapping("/boom-jooq")
        String jooq() {
            throw new DataAccessException("insert",
                    new SQLException("duplicate key value violates unique constraint", "23505"));
        }

        @GetMapping("/boom-spring")
        String spring() {
            throw new DuplicateKeyException("duplicate key");
        }

        @GetMapping("/boom-unknown")
        String unknown() {
            throw new DataAccessException("select",
                    new SQLException("connection reset", "08006"));
        }
    }

    private final MockMvc mockMvc = MockMvcBuilders.standaloneSetup(new ThrowingController())
            .setControllerAdvice(new GlobalExceptionHandler())
            .build();

    @Test
    void uniqueViolationMapsTo409() {
        DataAccessException error = new DataAccessException("insert",
                new SQLException("duplicate key value violates unique constraint", "23505"));

        var response = handler.handleDataAccess(error);

        assertThat(response.getStatusCode().value()).isEqualTo(409);
        assertThat(response.getBody()).containsEntry("code", "CONFLICT");
    }

    @Test
    void otherDataAccessErrorsStay500() {
        DataAccessException error = new DataAccessException("select",
                new SQLException("connection reset", "08006"));

        var response = handler.handleDataAccess(error);

        assertThat(response.getStatusCode().value()).isEqualTo(500);
        assertThat(response.getBody()).containsEntry("code", "INTERNAL_ERROR");
    }

    @Test
    void jooqUniqueViolationRoutesToConflictHandler() throws Exception {
        mockMvc.perform(get("/boom-jooq"))
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.code").value("CONFLICT"));
    }

    @Test
    void springDuplicateKeyRoutesToConflictHandler() throws Exception {
        mockMvc.perform(get("/boom-spring"))
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.code").value("CONFLICT"));
    }

    @Test
    void unknownDataAccessErrorRoutesToGeneric500Handler() throws Exception {
        mockMvc.perform(get("/boom-unknown"))
                .andExpect(status().isInternalServerError())
                .andExpect(jsonPath("$.code").value("INTERNAL_ERROR"));
    }
}
