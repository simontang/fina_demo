package com.fina.platform.tenant;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;

class TenantContextInterceptorTest {

    @AfterEach
    void tearDown() {
        TenantContextHolder.clear();
    }

    @Test
    void businessObjectRuntimePathSkipsGlobalPlatformApiKey() throws Exception {
        TenantContextInterceptor interceptor = new TenantContextInterceptor("admin-secret", "");
        MockHttpServletRequest request = new MockHttpServletRequest("GET", "/api/v1/bo/objects");
        MockHttpServletResponse response = new MockHttpServletResponse();

        assertTrue(interceptor.preHandle(request, response, new Object()));
        assertEquals(200, response.getStatus());
    }

    @Test
    void nonBusinessObjectPathStillRequiresPlatformApiKeyWhenConfigured() throws Exception {
        TenantContextInterceptor interceptor = new TenantContextInterceptor("admin-secret", "");
        MockHttpServletRequest request = new MockHttpServletRequest("GET", "/api/v1/files");
        MockHttpServletResponse response = new MockHttpServletResponse();

        assertFalse(interceptor.preHandle(request, response, new Object()));
        assertEquals(401, response.getStatus());
        assertTrue(response.getContentAsString().contains("API_KEY_INVALID"));
    }

    @Test
    void nonBusinessObjectPathStillSetsTenantWhenAuthorized() throws Exception {
        TenantContextInterceptor interceptor = new TenantContextInterceptor("admin-secret", "");
        MockHttpServletRequest request = new MockHttpServletRequest("GET", "/api/v1/files");
        request.addHeader("X-Api-Key", "admin-secret");
        request.addHeader("X-Tenant-Id", "tenant_a");
        MockHttpServletResponse response = new MockHttpServletResponse();

        assertTrue(interceptor.preHandle(request, response, new Object()));
        assertEquals("tenant_a", TenantContextHolder.getTenant());
    }
}
