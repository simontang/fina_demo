package com.fina.platform.bo;

import com.fina.platform.bo.BusinessObjectDtos.QueryResponse;
import com.fina.platform.bo.BusinessObjectDtos.RecordRequest;
import com.fina.platform.bo.BusinessObjectDtos.RecordResponse;
import com.fina.platform.bo.BusinessObjectDtos.StoreRequest;
import com.fina.platform.bo.BusinessObjectDtos.StoreResponse;
import com.fina.platform.bo.BusinessObjectService.BoAuthContext;
import com.fina.platform.bo.BusinessObjectService.BoGrant;
import com.fina.platform.exception.GlobalExceptionHandler;
import jakarta.servlet.http.HttpServletRequest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@ExtendWith(MockitoExtension.class)
class BusinessObjectControllerTest {
    @Mock
    private BusinessObjectService service;

    private MockMvc mvc;
    private BoAuthContext auth;

    @BeforeEach
    void setUp() {
        mvc = MockMvcBuilders
                .standaloneSetup(new BusinessObjectController(service))
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
        auth = new BoAuthContext("tenant",
                List.of(new BoGrant(1L, "crm_store", "tenant", true, true, true)));
    }

    @Test
    void runtimeRecordQueryIsAddressedByObjectKeyWithoutStoreKey() throws Exception {
        when(service.authenticate(any(HttpServletRequest.class))).thenReturn(auth);
        when(service.queryRecords(eq(auth), eq("customer"), any()))
                .thenReturn(new QueryResponse("customer", 1, 50, 1L, List.of(Map.of("id", "r1"))));

        mvc.perform(post("/api/v1/bo/objects/customer/records/query")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"filters":[{"field":"name","op":"contains","value":"ACME"}]}
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.objectKey").value("customer"))
                .andExpect(jsonPath("$.total").value(1));

        verify(service).authenticate(any(HttpServletRequest.class));
        verify(service).queryRecords(eq(auth), eq("customer"), any());
    }

    @Test
    void recordCreateIsAddressedByObjectKeyWithoutStoreKey() throws Exception {
        when(service.authenticate(any(HttpServletRequest.class))).thenReturn(auth);
        when(service.createRecord(eq(auth), eq("customer"), any(RecordRequest.class)))
                .thenReturn(new RecordResponse("r1", "customer", Map.of("name", "ACME")));

        mvc.perform(post("/api/v1/bo/objects/customer/records")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"data":{"name":"ACME"}}
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value("r1"))
                .andExpect(jsonPath("$.objectKey").value("customer"));
    }

    @Test
    void storeManagementStillUsesExplicitStoreKey() throws Exception {
        when(service.createStore(any(StoreRequest.class)))
                .thenReturn(new StoreResponse(1L, "crm_store", "CRM", null,
                        "jdbc:postgresql://localhost/bo_crm", "public", "bo", 1));

        mvc.perform(post("/api/v1/bo/stores")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "storeKey": "crm_store",
                                  "name": "CRM",
                                  "jdbcUrl": "jdbc:postgresql://localhost/bo_crm",
                                  "username": "bo",
                                  "password": "secret"
                                }
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.storeKey").value("crm_store"))
                .andExpect(jsonPath("$.password").doesNotExist());
    }
}
