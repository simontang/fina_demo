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
import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
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
    void runStreamingReturnsAfterHeadersWithoutWaitingForSseBody() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        AtomicReference<String> authorization = new AtomicReference<>();
        startServer(exchange -> {
            requestBody.set(readRequestBody(exchange));
            authorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
            exchange.getResponseHeaders().add("Content-Type", "text/event-stream");
            exchange.sendResponseHeaders(200, 0);
            try (OutputStream out = exchange.getResponseBody()) {
                out.write("data: started\n\n".getBytes(StandardCharsets.UTF_8));
                out.flush();
                try {
                    Thread.sleep(1500);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        });

        AgentRunClientImpl client = new AgentRunClientImpl(
                HttpClient.newHttpClient(),
                properties("http://127.0.0.1:" + server.getAddress().getPort(), true),
                new ObjectMapper());

        AgentRunRequest request = AgentRunRequest.builder()
                .assistantId("assistant-1")
                .threadId("thread-1")
                .message("hello")
                .streaming(true)
                .background(null)
                .build();

        long start = System.nanoTime();
        AgentRunResult result = client.run(request);
        long elapsedMs = Duration.ofNanos(System.nanoTime() - start).toMillis();

        assertTrue(result.sent());
        assertEquals("200", result.status());
        assertNotNull(result.rawResponse());
        assertTrue(result.rawResponse().contains("\"contentType\":\"text/event-stream\""));
        assertTrue(elapsedMs < 1000, "streaming call should return after headers");
        assertNotNull(requestBody.get());
        assertNotNull(authorization.get());
        assertTrue(requestBody.get().contains("\"assistant_id\":\"assistant-1\""));
        assertTrue(requestBody.get().contains("\"streaming\":true"));
        assertEquals("Bearer test-token", authorization.get());
    }

    @Test
    void runNonStreamingExtractsRunIdFromJsonBody() throws Exception {
        startServer(exchange -> {
            String body = "{\"id\":\"run-123\",\"status\":\"queued\"}";
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, bytes.length);
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(bytes);
            }
        });

        AgentRunClientImpl client = new AgentRunClientImpl(
                HttpClient.newHttpClient(),
                properties("http://127.0.0.1:" + server.getAddress().getPort(), false),
                new ObjectMapper());

        AgentRunRequest request = AgentRunRequest.builder()
                .assistantId("assistant-1")
                .threadId("thread-1")
                .message("hello")
                .streaming(false)
                .background(false)
                .build();

        AgentRunResult result = client.run(request);

        assertTrue(result.sent());
        assertEquals("200", result.status());
        assertEquals("run-123", result.runId());
        assertTrue(result.rawResponse().contains("\"status\":\"queued\""));
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
        server.createContext("/api/runs", handler);
        server.start();
    }

    private String readRequestBody(HttpExchange exchange) throws IOException {
        return new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
    }

    private AgentRunProperties properties(String baseUrl, boolean streaming) {
        return new AgentRunProperties(
                true,
                baseUrl,
                "Bearer test-token",
                "tenant_2",
                "workspace_1",
                "project_1",
                "assistant-1",
                "demo",
                streaming,
                false,
                1000,
                5000
        );
    }
}
