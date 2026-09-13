package com.fina.platform.config;

import com.fina.platform.tenant.TenantContextInterceptor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.ViewControllerRegistry;
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
                // portal pages are tenant-agnostic shells; their API calls carry
                // the tenant header per request
                .excludePathPatterns("/actuator/**", "/error", "/portal", "/portal/**");
    }

    @Override
    public void addViewControllers(ViewControllerRegistry registry) {
        registry.addViewController("/portal").setViewName("forward:/portal/index.html");
        registry.addViewController("/portal/").setViewName("forward:/portal/index.html");
    }
}
