package com.fina.b1s.agent;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.http.HttpClient;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentRunClientImplTest {

    private HttpServer server;
    private ExecutorService executor;

    @AfterEach
    void tearDown() {
        if (server != null) {
            server.stop(0);
            server = null;
        }
        if (executor != null) {
            executor.shutdownNow();
            executor = null;
        }
    }

    @Test
    void runPostsInboundPayloadWithBearerAuth() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        AtomicReference<String> authorization = new AtomicReference<>();
        AtomicReference<String> tenantHeader = new AtomicReference<>();
        startServer(exchange -> {
            requestBody.set(readRequestBody(exchange));
            authorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
            tenantHeader.set(exchange.getRequestHeaders().getFirst("x-tenant-id"));
            String body = "{\"accepted\":true}";
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(202, bytes.length);
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(bytes);
            }
        });

        AgentRunClientImpl client = new AgentRunClientImpl(
                HttpClient.newHttpClient(),
                properties("http://127.0.0.1:" + server.getAddress().getPort()),
                new ObjectMapper());

        AgentInboundRequest request = AgentInboundRequest.builder()
                .channel("email")
                .channelInstallationId("0f13c809-86bb-410b-ac8e-946be7772ba6")
                .tenantId("tenant_2")
                .sender(new AgentInboundRequest.Sender("sender@example.com"))
                .content(new AgentInboundRequest.Content("hello"))
                .build();

        AgentRunResult result = client.run(request);

        assertTrue(result.sent());
        assertEquals("202", result.status());
        assertTrue(result.rawResponse().contains("\"accepted\":true"));
        assertTrue(requestBody.get().contains("\"channel\":\"email\""));
        assertTrue(requestBody.get().contains("\"channelInstallationId\":\"0f13c809-86bb-410b-ac8e-946be7772ba6\""));
        assertTrue(requestBody.get().contains("\"tenantId\":\"tenant_2\""));
        assertTrue(requestBody.get().contains("\"sender\":{\"id\":\"sender@example.com\"}"));
        assertTrue(requestBody.get().contains("\"content\":{\"text\":\"hello\"}"));
        assertEquals("Bearer test-token", authorization.get());
        assertNull(tenantHeader.get());
    }

    @Test
    void runReturnsFailureForNon2xxResponse() throws Exception {
        startServer(exchange -> {
            String body = "{\"error\":\"upstream failed\"}";
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(504, bytes.length);
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(bytes);
            }
        });

        AgentRunClientImpl client = new AgentRunClientImpl(
                HttpClient.newHttpClient(),
                properties("http://127.0.0.1:" + server.getAddress().getPort()),
                new ObjectMapper());

        AgentInboundRequest request = AgentInboundRequest.builder()
                .channel("email")
                .channelInstallationId("0f13c809-86bb-410b-ac8e-946be7772ba6")
                .tenantId("tenant_2")
                .sender(new AgentInboundRequest.Sender("sender@example.com"))
                .content(new AgentInboundRequest.Content("hello"))
                .build();

        AgentRunResult result = client.run(request);

        assertFalse(result.sent());
        assertEquals("504", result.status());
        assertNull(result.runId());
        assertTrue(result.rawResponse().contains("\"error\":\"upstream failed\""));
        assertFalse(result.rawResponse().isBlank());
    }

    private void startServer(HttpHandler handler) throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        executor = Executors.newCachedThreadPool(r -> {
            Thread thread = new Thread(r);
            thread.setDaemon(true);
            thread.setName("agent-run-test-" + thread.getId());
            return thread;
        });
        server.setExecutor(executor);
        server.createContext("/api/channels/inbound", handler);
        server.start();
    }

    private String readRequestBody(HttpExchange exchange) throws IOException {
        return new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
    }

    private AgentRunProperties properties(String baseUrl) {
        return new AgentRunProperties(
                true,
                baseUrl,
                "Bearer test-token",
                "tenant_2",
                "0f13c809-86bb-410b-ac8e-946be7772ba6",
                1000,
                5000
        );
    }
}
