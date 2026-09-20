package com.fina.platform.bo;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.platform.bo.BusinessObjectDtos.FieldDefinition;
import com.fina.platform.bo.BusinessObjectDtos.ObjectDefinitionRequest;
import com.fina.platform.exception.ApiException;
import jakarta.servlet.http.HttpServletRequest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;

import java.sql.ResultSet;
import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.contains;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class BusinessObjectServiceAuthTest {

    private JdbcTemplate jdbcTemplate;
    private BusinessObjectService service;

    @BeforeEach
    void setUp() {
        jdbcTemplate = mock(JdbcTemplate.class);
        service = new BusinessObjectService(jdbcTemplate, new ObjectMapper());
    }

    @Test
    void missingStoreKeyReturns401() {
        HttpServletRequest request = mockRequest(null);

        assertThatThrownBy(() -> service.authenticate(request))
                .isInstanceOf(ApiException.class)
                .extracting("status", "code")
                .containsExactly(401, "BO_STORE_KEY_REQUIRED");
    }

    @Test
    @SuppressWarnings({"unchecked", "rawtypes"})
    void invalidStoreKeyReturns401() {
        HttpServletRequest request = mockRequest("bad-key");
        when(jdbcTemplate.query(contains("FROM bo_store_api_keys"), any(RowMapper.class), any()))
                .thenReturn(List.of());

        assertThatThrownBy(() -> service.authenticate(request))
                .isInstanceOf(ApiException.class)
                .extracting("status", "code")
                .containsExactly(401, "BO_STORE_KEY_INVALID");
    }

    @Test
    @SuppressWarnings({"unchecked", "rawtypes"})
    void validStoreKeyResolvesExactlyOneStoreAndPermissions() {
        HttpServletRequest request = mockRequest("bos_secret");
        when(jdbcTemplate.query(contains("FROM bo_store_api_keys"), any(RowMapper.class), eq(BusinessObjectService.hash("bos_secret"))))
                .thenAnswer(invocation -> {
                    RowMapper<BusinessObjectService.BoGrant> mapper = invocation.getArgument(1);
                    ResultSet rs = mock(ResultSet.class);
                    when(rs.getLong("store_id")).thenReturn(7L);
                    when(rs.getString("store_key")).thenReturn("crm_store");
                    when(rs.getString("key_name")).thenReturn("read_key");
                    when(rs.getString("permissions_json")).thenReturn("[\"READ\"]");
                    return List.of(mapper.mapRow(rs, 0));
                });

        BusinessObjectService.BoAuthContext auth = service.authenticate(request);

        assertThat(auth.grants()).hasSize(1);
        BusinessObjectService.BoGrant grant = auth.grants().get(0);
        assertThat(grant.storeId()).isEqualTo(7L);
        assertThat(grant.storeKey()).isEqualTo("crm_store");
        assertThat(grant.canRead()).isTrue();
        assertThat(grant.canWrite()).isFalse();
        assertThat(grant.canManage()).isFalse();
    }

    @Test
    void objectCreateCannotTargetAnotherStore() {
        BusinessObjectService.BoAuthContext auth = new BusinessObjectService.BoAuthContext(
                "store-key",
                List.of(new BusinessObjectService.BoGrant(1L, "crm_store", "manage_key", true, true, true))
        );
        ObjectDefinitionRequest request = new ObjectDefinitionRequest(
                "other_store",
                "customer",
                "Customer",
                null,
                List.of(new FieldDefinition("name", "string", true, 255, null, null, null)),
                List.of(),
                1
        );

        assertThatThrownBy(() -> service.createObject(auth, request))
                .isInstanceOf(ApiException.class)
                .extracting("status", "code")
                .containsExactly(403, "BO_STORE_FORBIDDEN");
    }

    private HttpServletRequest mockRequest(String key) {
        HttpServletRequest request = mock(HttpServletRequest.class);
        when(request.getHeader("X-BO-Connection-Key")).thenReturn(key);
        when(request.getHeaderNames()).thenReturn(Collections.emptyEnumeration());
        return request;
    }
}
