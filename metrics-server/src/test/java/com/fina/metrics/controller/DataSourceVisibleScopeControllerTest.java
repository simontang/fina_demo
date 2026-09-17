package com.fina.metrics.controller;

import com.fina.metrics.dto.DataSourceTableGrantVO;
import com.fina.metrics.exception.GlobalExceptionHandler;
import com.fina.metrics.service.DataSourceService;
import com.fina.metrics.service.DataSourceTableAccessService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.util.List;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

class DataSourceVisibleScopeControllerTest {
    private DataSourceTableAccessService scopes;
    private MockMvc mvc;

    @BeforeEach
    void setUp() {
        scopes = mock(DataSourceTableAccessService.class);
        mvc = MockMvcBuilders.standaloneSetup(new DataSourceController(mock(DataSourceService.class), scopes))
                .setControllerAdvice(new GlobalExceptionHandler()).build();
    }

    @Test
    void preferredScopeRoutesContainNoTenantAndIgnoreHeader() throws Exception {
        DataSourceTableGrantVO rule = rule();
        when(scopes.listGrants(isNull(), eq(15L))).thenReturn(List.of(rule));
        when(scopes.createGrant(isNull(), eq(15L), any())).thenReturn(rule);
        when(scopes.updateGrant(isNull(), eq(15L), eq(8L), any())).thenReturn(rule);
        String body = "{\"schemaName\":\"public\",\"tablePattern\":\"hankel_\",\"patternType\":\"PREFIX\"}";

        mvc.perform(get("/api/v1/datasources/15/visible-scopes").header("X-Tenant-Id", "unrelated"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.data[0].tenantId").doesNotExist())
                .andExpect(jsonPath("$.data[0].datasourceId").value(15));
        mvc.perform(post("/api/v1/datasources/15/visible-scopes")
                        .contentType(MediaType.APPLICATION_JSON).content(body))
                .andExpect(status().isOk()).andExpect(jsonPath("$.data.tenantId").doesNotExist());
        mvc.perform(put("/api/v1/datasources/15/visible-scopes/8")
                        .contentType(MediaType.APPLICATION_JSON).content(body))
                .andExpect(status().isOk());
        mvc.perform(delete("/api/v1/datasources/15/visible-scopes/8"))
                .andExpect(status().isOk());
        verify(scopes).deleteGrant(null, 15L, 8L);
    }

    @Test
    void legacyRoutesRetainTheirWireShape() throws Exception {
        when(scopes.listGrants(any(), eq(15L))).thenReturn(List.of(rule()));
        mvc.perform(get("/api/v1/datasources/15/table-grants").header("X-Tenant-Id", "legacy"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.data[0].tenantId").value("__datasource__"))
                .andExpect(jsonPath("$.data[0].tablePattern").value("hankel_"));
    }

    private DataSourceTableGrantVO rule() {
        DataSourceTableGrantVO rule = new DataSourceTableGrantVO();
        rule.setId(8L);
        rule.setDatasourceId(15L);
        rule.setTenantId("__datasource__");
        rule.setSchemaName("public");
        rule.setTablePattern("hankel_");
        rule.setPatternType("PREFIX");
        rule.setStatus(1);
        return rule;
    }
}
