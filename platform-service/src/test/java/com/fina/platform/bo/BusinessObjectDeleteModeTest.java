package com.fina.platform.bo;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.platform.bo.BusinessObjectDtos.FieldDefinition;
import com.fina.platform.bo.BusinessObjectDtos.ObjectDefinitionRequest;
import com.fina.platform.exception.ApiException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;

import java.sql.ResultSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.contains;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class BusinessObjectDeleteModeTest {

    private static final String OBJECT_KEY = "customer_tag";

    private static final BusinessObjectService.BoAuthContext AUTH = new BusinessObjectService.BoAuthContext(
            "key",
            List.of(new BusinessObjectService.BoGrant(1L, "elc", "elc_full", true, true, true))
    );

    private ObjectMapper objectMapper;
    private JdbcTemplate jdbcTemplate;
    private BusinessObjectService service;

    @BeforeEach
    void setUp() {
        objectMapper = new ObjectMapper();
        jdbcTemplate = mock(JdbcTemplate.class);
        service = new BusinessObjectService(jdbcTemplate, objectMapper);
    }

    @Test
    void legacySchemaWithoutDeleteModeDefaultsToHard() {
        stubDefinition(schemaJson(null));

        assertThat(service.getObject(AUTH, OBJECT_KEY).deleteMode()).isEqualTo("hard");
    }

    @Test
    void explicitSoftDeleteModeIsEchoed() {
        stubDefinition(schemaJson("soft"));

        assertThat(service.getObject(AUTH, OBJECT_KEY).deleteMode()).isEqualTo("soft");
    }

    @Test
    void updateObjectRejectsDeleteModeChange() {
        stubDefinition(schemaJson("soft"));
        ObjectDefinitionRequest request = new ObjectDefinitionRequest(
                null, OBJECT_KEY, "Customer Tag", null,
                List.of(field()), List.of(), 1, "hard");

        assertThatThrownBy(() -> service.updateObject(AUTH, OBJECT_KEY, request))
                .isInstanceOf(ApiException.class)
                .hasMessageContaining("deleteMode cannot be changed")
                .extracting("status")
                .isEqualTo(400);
    }

    @Test
    void updateObjectAcceptsOmittedDeleteMode() {
        stubDefinition(schemaJson("soft"));
        ObjectDefinitionRequest request = new ObjectDefinitionRequest(
                null, OBJECT_KEY, "Customer Tag", null,
                List.of(field()), List.of(), 1, null);

        // No store datasource is stubbed, so the call proceeds past the deleteMode
        // guard and fails later at store lookup. Reaching the store lookup proves the
        // omitted deleteMode was accepted (it must not be the deleteMode error).
        assertThatThrownBy(() -> service.updateObject(AUTH, OBJECT_KEY, request))
                .isInstanceOf(ApiException.class)
                .hasMessageNotContaining("deleteMode cannot be changed")
                .hasMessageStartingWith("business object store not found")
                .extracting("status", "code")
                .containsExactly(404, "NOT_FOUND");
    }

    private FieldDefinition field() {
        return new FieldDefinition("name", "string", null, null, null, null, null);
    }

    private String schemaJson(String deleteMode) {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("fields", List.of(Map.of("key", "name", "type", "string")));
        schema.put("indexes", List.of());
        if (deleteMode != null) {
            schema.put("deleteMode", deleteMode);
        }
        try {
            return objectMapper.writeValueAsString(schema);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private void stubDefinition(String schemaJson) {
        when(jdbcTemplate.query(contains("FROM bo_object_definitions"), any(RowMapper.class), any(Object[].class)))
                .thenAnswer(invocation -> {
                    RowMapper mapper = invocation.getArgument(1);
                    ResultSet rs = mock(ResultSet.class);
                    when(rs.getLong("id")).thenReturn(1L);
                    when(rs.getLong("store_id")).thenReturn(1L);
                    when(rs.getString("store_key")).thenReturn("elc");
                    when(rs.getString("object_key")).thenReturn(OBJECT_KEY);
                    when(rs.getString("table_name")).thenReturn("bo_customer_tag");
                    when(rs.getString("display_name")).thenReturn("Customer Tag");
                    when(rs.getString("description")).thenReturn(null);
                    when(rs.getString("schema_json")).thenReturn(schemaJson);
                    when(rs.getInt("status")).thenReturn(1);
                    return List.of(mapper.mapRow(rs, 0));
                });
    }
}
