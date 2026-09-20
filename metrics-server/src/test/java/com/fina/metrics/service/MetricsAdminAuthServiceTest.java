package com.fina.metrics.service;

import com.fina.metrics.exception.UnauthorizedException;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.test.util.ReflectionTestUtils;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class MetricsAdminAuthServiceTest {

    @Test
    void rejectsWhenAdminKeyIsNotConfigured() {
        MetricsAdminAuthService service = serviceWithKey("");
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("Authorization", "Bearer secret");

        assertThatThrownBy(() -> service.requireAdmin(request))
                .isInstanceOf(UnauthorizedException.class)
                .hasMessageContaining("not configured");
    }

    @Test
    void acceptsBearerToken() {
        MetricsAdminAuthService service = serviceWithKey("secret");
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("Authorization", "Bearer secret");

        assertThatCode(() -> service.requireAdmin(request)).doesNotThrowAnyException();
    }

    @Test
    void acceptsExplicitAdminHeader() {
        MetricsAdminAuthService service = serviceWithKey("secret");
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("X-Metrics-Admin-Key", "secret");

        assertThatCode(() -> service.requireAdmin(request)).doesNotThrowAnyException();
    }

    @Test
    void rejectsInvalidToken() {
        MetricsAdminAuthService service = serviceWithKey("secret");
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("Authorization", "Bearer wrong");

        assertThatThrownBy(() -> service.requireAdmin(request))
                .isInstanceOf(UnauthorizedException.class)
                .hasMessageContaining("required");
    }

    private MetricsAdminAuthService serviceWithKey(String key) {
        MetricsAdminAuthService service = new MetricsAdminAuthService();
        ReflectionTestUtils.setField(service, "adminApiKey", key);
        return service;
    }
}
