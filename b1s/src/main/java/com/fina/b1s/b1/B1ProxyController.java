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
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;

@Slf4j
@RestController
@RequiredArgsConstructor
public class B1ProxyController {

    private static final List<String> HOP_BY_HOP_HEADERS = List.of(
            "host", "connection", "content-length", "transfer-encoding", "cookie");

    private final RestTemplate b1RestTemplate;
    private final B1ServiceLayerProperties properties;
    private final B1SessionManager sessionManager;

    @RequestMapping("/b1s/v1/**")
    public ResponseEntity<byte[]> proxy(HttpServletRequest request) throws IOException {
        String tenantId = firstHeader(request, "X-Tenant-Id", "Tenant-Id");
        B1Session session = sessionManager.getOrLogin(tenantId);
        try {
            return forward(request, session);
        } catch (HttpStatusCodeException ex) {
            if (ex.getStatusCode().value() == 401 || ex.getStatusCode().value() == 403) {
                sessionManager.discard(tenantId, session);
                B1Session refreshed = sessionManager.getOrLogin(tenantId);
                return forward(request, refreshed);
            }
            throw ex;
        }
    }

    private ResponseEntity<byte[]> forward(HttpServletRequest request, B1Session session) throws IOException {
        HttpHeaders headers = copyHeaders(request);
        headers.put(HttpHeaders.COOKIE, session.cookies());

        byte[] body = StreamUtils.copyToByteArray(request.getInputStream());
        URI uri = buildTargetUri(request);
        HttpMethod method = HttpMethod.valueOf(request.getMethod());
        return b1RestTemplate.exchange(uri, method, new HttpEntity<>(body, headers), byte[].class);
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
            if (HOP_BY_HOP_HEADERS.contains(name.toLowerCase())) {
                continue;
            }
            headers.put(name, Collections.list(request.getHeaders(name)));
        }
        return headers;
    }

    private static String firstHeader(HttpServletRequest request, String... names) {
        for (String name : names) {
            String value = request.getHeader(name);
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return "default";
    }
}
