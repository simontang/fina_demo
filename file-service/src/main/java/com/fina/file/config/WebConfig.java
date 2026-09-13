package com.fina.file.config;

import com.fina.file.tenant.TenantContextInterceptor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {

    private final String apiKey;
    private final String defaultTenant;

    public WebConfig(@Value("${file.api-key:}") String apiKey,
                     @Value("${file.tenant.default-tenant:}") String defaultTenant) {
        this.apiKey = apiKey;
        this.defaultTenant = defaultTenant;
    }

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(new TenantContextInterceptor(apiKey, defaultTenant))
                .addPathPatterns("/**")
                .excludePathPatterns("/actuator/**", "/error");
    }
}
