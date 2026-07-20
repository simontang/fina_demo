package com.fina.b1s.b1;

import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.util.StreamUtils;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.io.IOException;
import java.net.URI;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.UUID;

@Slf4j
@RestController
@RequiredArgsConstructor
public class B1ProxyController {

    private static final List<String> REQUEST_HOP_BY_HOP_HEADERS = List.of(
            "host", "connection", "content-length", "transfer-encoding", "cookie", "accept-encoding",
            "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "trailers", "upgrade",
            "x-company-db", "companydb", "company-db");

    private static final List<String> RESPONSE_HOP_BY_HOP_HEADERS = List.of(
            "connection", "content-length", "transfer-encoding",
            "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "trailers", "upgrade");

    private final RestTemplate b1RestTemplate;
    private final B1ServiceLayerProperties properties;
    private final B1SessionManager sessionManager;
    private final B1ProxyProperties proxyProperties;
    private final B1RequestBodySanitizer requestBodySanitizer;

    @RequestMapping("/b1s/v1/**")
    public ResponseEntity<byte[]> proxy(HttpServletRequest request) throws IOException {
        String tenantId = firstHeader(request, "X-Tenant-Id", "Tenant-Id");
        String companyDb = firstHeaderOrDefault(request, properties.defaultCompanyDb(),
                "X-Company-DB", "CompanyDB", "Company-DB", "companydb");
        String requestId = firstHeaderOrDefault(request, UUID.randomUUID().toString(),
                "X-Request-Id", "X-Correlation-Id");
        byte[] body = StreamUtils.copyToByteArray(request.getInputStream());
        URI uri = buildTargetUri(request);
        HttpMethod method = HttpMethod.valueOf(request.getMethod());
        B1Session session = sessionManager.getOrLogin(tenantId, companyDb);
        try {
            return forward(request, body, uri, method, session);
        } catch (HttpStatusCodeException ex) {
            if (ex.getStatusCode().value() == 401 || ex.getStatusCode().value() == 403) {
                sessionManager.discard(tenantId, companyDb, session);
                B1Session refreshed = sessionManager.getOrLogin(tenantId, companyDb);
                try {
                    return forward(request, body, uri, method, refreshed);
                } catch (HttpStatusCodeException retryEx) {
                    return upstreamErrorResponse(request, body, tenantId, companyDb, requestId, retryEx);
                }
            }
            return upstreamErrorResponse(request, body, tenantId, companyDb, requestId, ex);
        }
    }

    private ResponseEntity<byte[]> forward(
            HttpServletRequest request,
            byte[] body,
            URI uri,
            HttpMethod method,
            B1Session session) {
        HttpHeaders headers = copyHeaders(request);
        headers.put(HttpHeaders.COOKIE, session.cookies());
        headers.set(HttpHeaders.ACCEPT_ENCODING, "identity");

        ResponseEntity<byte[]> response = b1RestTemplate.exchange(uri, method, new HttpEntity<>(body, headers), byte[].class);
        return ResponseEntity.status(response.getStatusCode())
                .headers(copyResponseHeaders(response.getHeaders()))
                .body(response.getBody());
    }

    private ResponseEntity<byte[]> upstreamErrorResponse(
            HttpServletRequest request,
            byte[] body,
            String tenantId,
            String companyDb,
            String requestId,
            HttpStatusCodeException ex) {
        logUpstreamError(request, body, tenantId, companyDb, requestId, ex);
        return ResponseEntity.status(ex.getStatusCode())
                .headers(copyResponseHeaders(ex.getResponseHeaders()))
                .header("X-Request-Id", requestId)
                .body(ex.getResponseBodyAsByteArray());
    }

    private void logUpstreamError(
            HttpServletRequest request,
            byte[] body,
            String tenantId,
            String companyDb,
            String requestId,
            HttpStatusCodeException ex) {
        String requestBody = proxyProperties.logRequestBodyOnError()
                ? requestBodySanitizer.sanitize(body, proxyProperties.maxRequestBodyLogChars())
                : "[disabled]";
        String upstreamBody = requestBodySanitizer.sanitize(
                ex.getResponseBodyAsByteArray(),
                proxyProperties.maxRequestBodyLogChars());
        log.warn(
                "B1 upstream error requestId={} method={} path={} query={} tenantId={} companyDb={} upstreamStatus={} upstreamErrorBody={} sanitizedRequestBody={}",
                requestId,
                request.getMethod(),
                request.getRequestURI(),
                request.getQueryString(),
                tenantId,
                companyDb,
                ex.getStatusCode().value(),
                upstreamBody,
                requestBody);
    }

    private URI buildTargetUri(HttpServletRequest request) {
        String prefix = request.getContextPath() + "/b1s/v1";
        String path = request.getRequestURI().substring(prefix.length());
        UriComponentsBuilder builder = UriComponentsBuilder.fromHttpUrl(properties.baseUrl()).path(path);
        if (request.getQueryString() != null && !request.getQueryString().isBlank()) {
            builder.query(request.getQueryString());
        }
        return builder.build(true).toUri();
    }

    private static HttpHeaders copyHeaders(HttpServletRequest request) {
        HttpHeaders headers = new HttpHeaders();
        Enumeration<String> names = request.getHeaderNames();
        while (names.hasMoreElements()) {
            String name = names.nextElement();
            if (REQUEST_HOP_BY_HOP_HEADERS.contains(name.toLowerCase())) {
                continue;
            }
            headers.put(name, Collections.list(request.getHeaders(name)));
        }
        return headers;
    }

    private static HttpHeaders copyResponseHeaders(HttpHeaders source) {
        HttpHeaders headers = new HttpHeaders();
        if (source == null) {
            return headers;
        }
        source.forEach((name, values) -> {
            if (!RESPONSE_HOP_BY_HOP_HEADERS.contains(name.toLowerCase())) {
                headers.put(name, values);
            }
        });
        return headers;
    }

    private static String firstHeader(HttpServletRequest request, String... names) {
        return firstHeaderOrDefault(request, "default", names);
    }

    private static String firstHeaderOrDefault(HttpServletRequest request, String defaultValue, String... names) {
        for (String name : names) {
            String value = request.getHeader(name);
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return defaultValue;
    }
}
