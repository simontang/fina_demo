package com.fina.b1s.b1;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestTemplate;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.client.ExpectedCount.once;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.content;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.header;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withStatus;

class B1ProxyControllerTest {

    private RestTemplate restTemplate;
    private MockRestServiceServer server;
    private B1SessionManager sessionManager;
    private B1ProxyController controller;
    private ListAppender<ILoggingEvent> logAppender;

    @BeforeEach
    void setUp() {
        restTemplate = new RestTemplate();
        server = MockRestServiceServer.bindTo(restTemplate).build();
        sessionManager = mock(B1SessionManager.class);
        when(sessionManager.getOrLogin(anyString(), anyString()))
                .thenReturn(new B1Session("session-1", "route-1", Instant.now()));
        controller = new B1ProxyController(
                restTemplate,
                new B1ServiceLayerProperties(
                        "https://sap.example/b1s/v1",
                        "SBOTEST",
                        "manager",
                        "password",
                        Map.of(),
                        Duration.ofSeconds(1),
                        Duration.ofSeconds(1)),
                sessionManager,
                new B1ProxyProperties(true, 8000),
                new B1RequestBodySanitizer(new ObjectMapper()));
    }

    @AfterEach
    void tearDown() {
        if (logAppender != null) {
            Logger logger = (Logger) LoggerFactory.getLogger(B1ProxyController.class);
            logger.detachAppender(logAppender);
            logAppender.stop();
        }
    }

    @Test
    void passesThroughUpstreamBadRequestBodyAndStatus() throws Exception {
        String requestBody = "{\"CardCode\":\"C_TROPICAL\"}";
        String upstreamBody = "{\"error\":{\"code\":-10,\"message\":{\"value\":\"bad draft\"}}}";
        server.expect(once(), requestTo("https://sap.example/b1s/v1/Drafts"))
                .andExpect(method(org.springframework.http.HttpMethod.POST))
                .andExpect(header(HttpHeaders.ACCEPT_ENCODING, "identity"))
                .andExpect(content().bytes(requestBody.getBytes(StandardCharsets.UTF_8)))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body(upstreamBody));

        ResponseEntity<byte[]> response = controller.proxy(postRequest("/b1s/v1/Drafts", requestBody));

        assertEquals(HttpStatus.BAD_REQUEST, response.getStatusCode());
        assertEquals(upstreamBody, new String(response.getBody(), StandardCharsets.UTF_8));
        server.verify();
    }

    @Test
    void passesThroughUpstreamNotFoundBodyAndStatus() throws Exception {
        String upstreamBody = "{\"error\":{\"code\":-2028,\"message\":{\"value\":\"Invalid BP code\"}}}";
        server.expect(once(), requestTo("https://sap.example/b1s/v1/Drafts"))
                .andExpect(method(org.springframework.http.HttpMethod.POST))
                .andRespond(withStatus(HttpStatus.NOT_FOUND)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body(upstreamBody));

        ResponseEntity<byte[]> response = controller.proxy(postRequest("/b1s/v1/Drafts", "{}"));

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertEquals(upstreamBody, new String(response.getBody(), StandardCharsets.UTF_8));
        server.verify();
    }

    @Test
    void retriesAuthFailureWithOriginalRequestBody() throws Exception {
        B1Session first = new B1Session("expired", "route-1", Instant.now());
        B1Session refreshed = new B1Session("fresh", "route-2", Instant.now());
        when(sessionManager.getOrLogin("default", "SBOTEST")).thenReturn(first, refreshed);
        String requestBody = "{\"DocObjectCode\":17,\"CardCode\":\"C_TROPICAL\"}";
        String upstreamBody = "{\"DocEntry\":123}";

        server.expect(once(), requestTo("https://sap.example/b1s/v1/Drafts"))
                .andExpect(method(org.springframework.http.HttpMethod.POST))
                .andExpect(content().bytes(requestBody.getBytes(StandardCharsets.UTF_8)))
                .andRespond(withStatus(HttpStatus.UNAUTHORIZED).body("{\"error\":\"expired\"}"));
        server.expect(once(), requestTo("https://sap.example/b1s/v1/Drafts"))
                .andExpect(method(org.springframework.http.HttpMethod.POST))
                .andExpect(content().bytes(requestBody.getBytes(StandardCharsets.UTF_8)))
                .andRespond(withStatus(HttpStatus.CREATED)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body(upstreamBody));

        ResponseEntity<byte[]> response = controller.proxy(postRequest("/b1s/v1/Drafts", requestBody));

        assertEquals(HttpStatus.CREATED, response.getStatusCode());
        assertEquals(upstreamBody, new String(response.getBody(), StandardCharsets.UTF_8));
        verify(sessionManager).discard("default", "SBOTEST", first);
        server.verify();
    }

    @Test
    void logsSanitizedRequestBodyForUpstreamError() throws Exception {
        attachLogAppender();
        String requestBody = """
                {"CardCode":"C_TROPICAL","Password":"secret","nested":{"accessToken":"abc"}}
                """;
        server.expect(once(), requestTo("https://sap.example/b1s/v1/Drafts"))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body("{\"error\":{\"message\":{\"value\":\"bad\"}}}"));
        MockHttpServletRequest request = postRequest("/b1s/v1/Drafts", requestBody);
        request.addHeader("X-Request-Id", "req-123");

        controller.proxy(request);

        String logs = logAppender.list.stream()
                .map(ILoggingEvent::getFormattedMessage)
                .collect(Collectors.joining("\n"));
        assertTrue(logs.contains("requestId=req-123"));
        assertTrue(logs.contains("path=/b1s/v1/Drafts"));
        assertTrue(logs.contains("upstreamStatus=400"));
        assertTrue(logs.contains("\"CardCode\":\"C_TROPICAL\""));
        assertTrue(logs.contains("\"Password\":\"***\""));
        assertTrue(logs.contains("\"accessToken\":\"***\""));
        assertFalse(logs.contains("secret"));
        assertFalse(logs.contains("abc"));
        server.verify();
    }

    private MockHttpServletRequest postRequest(String path, String body) {
        MockHttpServletRequest request = new MockHttpServletRequest("POST", path);
        request.setContentType(MediaType.APPLICATION_JSON_VALUE);
        request.addHeader(HttpHeaders.ACCEPT_ENCODING, "gzip");
        request.setContent(body.getBytes(StandardCharsets.UTF_8));
        return request;
    }

    private void attachLogAppender() {
        Logger logger = (Logger) LoggerFactory.getLogger(B1ProxyController.class);
        logAppender = new ListAppender<>();
        logAppender.start();
        logger.addAppender(logAppender);
    }
}
