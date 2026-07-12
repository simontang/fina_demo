package com.fina.cdp.config;

import jakarta.servlet.http.HttpServletRequest;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockHttpServletRequest;

import static org.assertj.core.api.Assertions.assertThat;

class TenantResolverTest {

    private final TenantResolver resolver = new TenantResolver();

    @Test
    void defaultsToDefaultTenantWhenHeaderIsMissing() {
        HttpServletRequest request = new MockHttpServletRequest();

        assertThat(resolver.resolve(request)).isEqualTo("default");
    }

    @Test
    void readsTenantFromHeaderAndTrimsWhitespace() {
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("X-Tenant-Id", " tenant_a ");

        assertThat(resolver.resolve(request)).isEqualTo("tenant_a");
    }
}
