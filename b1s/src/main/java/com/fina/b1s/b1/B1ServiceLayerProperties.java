package com.fina.b1s.b1;

import java.time.Duration;
import java.util.Map;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "b1.service-layer")
public record B1ServiceLayerProperties(
        String baseUrl,
        String defaultCompanyDb,
        String defaultUsername,
        String defaultPassword,
        Map<String, Credentials> companyCredentials,
        Duration connectTimeout,
        Duration readTimeout) {

    public Credentials credentialsFor(String companyDb) {
        String normalized = normalizeCompanyDb(companyDb);
        if (companyCredentials != null && !companyCredentials.isEmpty()) {
            Credentials exact = companyCredentials.get(normalized);
            if (exact != null) {
                return exact.withFallback(defaultUsername, defaultPassword);
            }
            for (Map.Entry<String, Credentials> entry : companyCredentials.entrySet()) {
                if (normalizeCompanyDb(entry.getKey()).equals(normalized)) {
                    return entry.getValue().withFallback(defaultUsername, defaultPassword);
                }
            }
        }
        return new Credentials(defaultUsername, defaultPassword);
    }

    public String resolveCompanyDb(String companyDb) {
        String normalized = normalizeCompanyDb(companyDb);
        return !normalized.isBlank() ? normalized : normalizeCompanyDb(defaultCompanyDb);
    }

    private static String normalizeCompanyDb(String companyDb) {
        return companyDb == null ? "" : companyDb.trim();
    }

    public record Credentials(String username, String password) {
        Credentials withFallback(String fallbackUsername, String fallbackPassword) {
            String resolvedUsername = username == null || username.isBlank() ? fallbackUsername : username;
            String resolvedPassword = password == null || password.isBlank() ? fallbackPassword : password;
            return new Credentials(resolvedUsername, resolvedPassword);
        }
    }
}
