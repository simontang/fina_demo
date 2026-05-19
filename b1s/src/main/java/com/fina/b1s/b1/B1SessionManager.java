package com.fina.b1s.b1;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.net.URI;
import java.time.Instant;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;

@Slf4j
@Component
@RequiredArgsConstructor
public class B1SessionManager {

    private static final int MAX_SESSIONS_PER_TENANT = 4;

    private final RestTemplate b1RestTemplate;
    private final B1ServiceLayerProperties properties;
    private final Map<String, Deque<B1Session>> sessionsByTenant = new ConcurrentHashMap<>();

    public B1Session getOrLogin(String tenantId) {
        Deque<B1Session> sessions = sessionsByTenant.computeIfAbsent(tenantKey(tenantId),
                key -> new ConcurrentLinkedDeque<>());
        B1Session existing = sessions.peekFirst();
        if (existing != null) {
            return existing;
        }

        B1Session created = login();
        sessions.addFirst(created);
        trim(sessions);
        return created;
    }

    public void discard(String tenantId, B1Session session) {
        Deque<B1Session> sessions = sessionsByTenant.get(tenantKey(tenantId));
        if (sessions != null) {
            sessions.remove(session);
        }
    }

    private B1Session login() {
        URI uri = UriComponentsBuilder.fromHttpUrl(properties.baseUrl())
                .path("/Login")
                .build(true)
                .toUri();
        LoginRequest request = new LoginRequest(
                properties.defaultCompanyDb(),
                properties.defaultUsername(),
                properties.defaultPassword());

        ResponseEntity<LoginResponse> response = b1RestTemplate.exchange(
                uri,
                HttpMethod.POST,
                new HttpEntity<>(request),
                LoginResponse.class);

        LoginResponse body = response.getBody();
        String sessionId = body != null ? body.sessionId() : null;
        if (sessionId == null || sessionId.isBlank()) {
            throw new IllegalStateException("B1 login did not return SessionId");
        }

        String routeId = extractCookie(response.getHeaders(), "ROUTEID");
        log.info("Created B1 Service Layer session routePresent={}", routeId != null);
        return new B1Session(sessionId, routeId, Instant.now());
    }

    private static String extractCookie(HttpHeaders headers, String name) {
        List<String> setCookies = headers.get(HttpHeaders.SET_COOKIE);
        if (setCookies == null) {
            return null;
        }
        String prefix = name + "=";
        for (String cookie : setCookies) {
            for (String part : cookie.split(";")) {
                String trimmed = part.trim();
                if (trimmed.startsWith(prefix)) {
                    return trimmed.substring(prefix.length());
                }
            }
        }
        return null;
    }

    private static String tenantKey(String tenantId) {
        return tenantId == null || tenantId.isBlank() ? "default" : tenantId.trim();
    }

    private static void trim(Deque<B1Session> sessions) {
        while (sessions.size() > MAX_SESSIONS_PER_TENANT) {
            sessions.pollLast();
        }
    }

    private record LoginRequest(
            @JsonProperty("CompanyDB") String companyDb,
            @JsonProperty("UserName") String username,
            @JsonProperty("Password") String password) {
    }

    private record LoginResponse(@JsonProperty("SessionId") String sessionId) {
    }
}
