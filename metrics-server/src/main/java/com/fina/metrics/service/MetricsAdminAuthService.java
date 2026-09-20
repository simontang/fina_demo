package com.fina.metrics.service;

import com.fina.metrics.exception.UnauthorizedException;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

@Service
public class MetricsAdminAuthService {

    private static final String BEARER_PREFIX = "Bearer ";

    @Value("${metrics.admin.api-key:}")
    private String adminApiKey;

    public void requireAdmin(HttpServletRequest request) {
        if (!StringUtils.hasText(adminApiKey)) {
            throw new UnauthorizedException("Metrics admin API key is not configured");
        }

        String provided = resolveProvidedKey(request);
        if (!StringUtils.hasText(provided) || !constantTimeEquals(adminApiKey.trim(), provided.trim())) {
            throw new UnauthorizedException("Metrics admin API key is required");
        }
    }

    private String resolveProvidedKey(HttpServletRequest request) {
        String direct = request.getHeader("X-Metrics-Admin-Key");
        if (StringUtils.hasText(direct)) {
            return direct;
        }
        String authorization = request.getHeader("Authorization");
        if (StringUtils.hasText(authorization) && authorization.startsWith(BEARER_PREFIX)) {
            return authorization.substring(BEARER_PREFIX.length());
        }
        return null;
    }

    private boolean constantTimeEquals(String expected, String provided) {
        byte[] expectedBytes = expected.getBytes(StandardCharsets.UTF_8);
        byte[] providedBytes = provided.getBytes(StandardCharsets.UTF_8);
        return MessageDigest.isEqual(expectedBytes, providedBytes);
    }
}
