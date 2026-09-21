package com.fina.platform.bo;

import com.fina.platform.bo.BusinessObjectDtos.BatchCreateResponse;
import com.fina.platform.bo.BusinessObjectDtos.BatchDeleteRequest;
import com.fina.platform.bo.BusinessObjectDtos.BatchDeleteResponse;
import com.fina.platform.bo.BusinessObjectDtos.BatchRecordRequest;
import com.fina.platform.bo.BusinessObjectDtos.QueryResponse;
import com.fina.platform.bo.BusinessObjectDtos.RecordRequest;
import com.fina.platform.bo.BusinessObjectDtos.RecordResponse;
import com.fina.platform.bo.BusinessObjectDtos.StoreApiKeyRequest;
import com.fina.platform.bo.BusinessObjectDtos.StoreApiKeyResponse;
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
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
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
        BusinessObjectController controller = new BusinessObjectController(service);
        ReflectionTestUtils.setField(controller, "apiKey", "admin-secret");
        mvc = MockMvcBuilders
                .standaloneSetup(controller)
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
    void batchRecordCreateIsAddressedByObjectKeyWithoutStoreKey() throws Exception {
        when(service.authenticate(any(HttpServletRequest.class))).thenReturn(auth);
        when(service.createRecords(eq(auth), eq("customer"), any(BatchRecordRequest.class)))
                .thenReturn(new BatchCreateResponse("customer", 2, List.of("r1", "r2")));

        mvc.perform(post("/api/v1/bo/objects/customer/records/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"records":[{"name":"A"},{"name":"B"}]}
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.objectKey").value("customer"))
                .andExpect(jsonPath("$.created").value(2))
                .andExpect(jsonPath("$.ids[0]").value("r1"));

        verify(service).createRecords(eq(auth), eq("customer"), any(BatchRecordRequest.class));
    }

    @Test
    void batchRecordDeleteIsAddressedByObjectKeyWithoutStoreKey() throws Exception {
        when(service.authenticate(any(HttpServletRequest.class))).thenReturn(auth);
        when(service.deleteRecords(eq(auth), eq("customer"), any(BatchDeleteRequest.class)))
                .thenReturn(new BatchDeleteResponse("customer", 1, List.of("r1")));

        mvc.perform(post("/api/v1/bo/objects/customer/records/batch-delete")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"ids":["r1","missing"]}
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.objectKey").value("customer"))
                .andExpect(jsonPath("$.deleted").value(1))
                .andExpect(jsonPath("$.ids[0]").value("r1"));

        verify(service).deleteRecords(eq(auth), eq("customer"), any(BatchDeleteRequest.class));
    }

    @Test
    void storeManagementStillUsesExplicitStoreKey() throws Exception {
        when(service.createStore(any(StoreRequest.class)))
                .thenReturn(new StoreResponse(1L, "crm_store", "CRM", null,
                        "jdbc:postgresql://localhost/bo_crm", "public", "bo", 1));

        mvc.perform(post("/api/v1/bo/stores")
                        .header("X-Api-Key", "admin-secret")
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

    @Test
    void storeManagementRequiresPlatformAdminKey() throws Exception {
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
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.code").value("API_KEY_INVALID"));
    }

    @Test
    void currentStoreUsesStoreKeyAuthContext() throws Exception {
        when(service.authenticate(any(HttpServletRequest.class))).thenReturn(auth);
        when(service.currentStore(auth))
                .thenReturn(new StoreResponse(1L, "crm_store", "CRM", null,
                        "jdbc:postgresql://localhost/bo_crm", "public", "bo", 1));

        mvc.perform(get("/api/v1/bo/stores/current")
                        .header("X-BO-Connection-Key", "bos_secret"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.storeKey").value("crm_store"))
                .andExpect(jsonPath("$.password").doesNotExist());
    }

    @Test
    void storeKeyManagementUsesExplicitStoreKey() throws Exception {
        when(service.createStoreApiKey(eq("crm_store"), any(StoreApiKeyRequest.class)))
                .thenReturn(new StoreApiKeyResponse(10L, 1L, "crm_store", "agent_key",
                        List.of("READ", "WRITE"), 1, null, null, "bos_generated"));

        mvc.perform(post("/api/v1/bo/stores/crm_store/keys")
                        .header("X-Api-Key", "admin-secret")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "keyName": "agent_key",
                                  "permissions": ["READ", "WRITE"]
                                }
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.storeKey").value("crm_store"))
                .andExpect(jsonPath("$.keyName").value("agent_key"))
                .andExpect(jsonPath("$.rawKey").value("bos_generated"));
    }
}
