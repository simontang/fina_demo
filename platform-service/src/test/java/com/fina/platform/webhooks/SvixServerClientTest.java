package com.fina.platform.webhooks;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SvixServerClientTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

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
    void createEndpointSendsSvixChannels() throws Exception {
        AtomicReference<JsonNode> endpointBody = new AtomicReference<>();
        startServer(exchange -> {
            String path = exchange.getRequestURI().getPath();
            if (path.startsWith("/api/v1/event-type/")) {
                sendJson(exchange, 200, "{}");
                return;
            }
            if (path.equals("/api/v1/app/app_1/endpoint/")) {
                endpointBody.set(readJson(exchange));
                sendJson(exchange, 200, """
                        {"id":"ep_1","url":"https://receiver.example/hook"}
                        """);
                return;
            }
            if (path.equals("/api/v1/app/app_1/endpoint/ep_1/secret/")) {
                sendJson(exchange, 200, "{\"key\":\"whsec_demo\"}");
                return;
            }
            sendJson(exchange, 404, "{}");
        });

        Map<String, Object> created = client().createEndpoint(
                "app_1",
                "https://receiver.example/hook",
                List.of("job.completed"),
                List.of("vip", "ops", "vip"),
                "demo");

        JsonNode body = endpointBody.get();
        assertEquals("https://receiver.example/hook", body.path("url").asText());
        assertEquals("job.completed", body.path("filterTypes").get(0).asText());
        assertEquals("vip", body.path("channels").get(0).asText());
        assertEquals("ops", body.path("channels").get(1).asText());
        assertEquals(2, body.path("channels").size());
        assertEquals(List.of("vip", "ops"), created.get("channels"));
        assertEquals(List.of("job.completed"), created.get("filterTypes"));
    }

    @Test
    void publishSendsSvixChannels() throws Exception {
        AtomicReference<JsonNode> messageBody = new AtomicReference<>();
        startServer(exchange -> {
            String path = exchange.getRequestURI().getPath();
            if (path.startsWith("/api/v1/event-type/")) {
                sendJson(exchange, 200, "{}");
                return;
            }
            if (path.equals("/api/v1/app/app_1/msg/")) {
                messageBody.set(readJson(exchange));
                sendJson(exchange, 200, "{\"id\":\"msg_1\"}");
                return;
            }
            sendJson(exchange, 404, "{}");
        });

        String messageId = client().publish(
                "app_1",
                "job.completed",
                Map.of("jobId", "1"),
                List.of("vip", "ops"));

        JsonNode body = messageBody.get();
        assertEquals("msg_1", messageId);
        assertEquals("job.completed", body.path("eventType").asText());
        assertEquals("1", body.path("payload").path("jobId").asText());
        assertEquals("vip", body.path("channels").get(0).asText());
        assertEquals("ops", body.path("channels").get(1).asText());
    }

    @Test
    void publishWithoutChannelsKeepsBroadcastPayloadShape() throws Exception {
        AtomicReference<JsonNode> messageBody = new AtomicReference<>();
        startServer(exchange -> {
            String path = exchange.getRequestURI().getPath();
            if (path.startsWith("/api/v1/event-type/")) {
                sendJson(exchange, 200, "{}");
                return;
            }
            if (path.equals("/api/v1/app/app_1/msg/")) {
                messageBody.set(readJson(exchange));
                sendJson(exchange, 200, "{\"id\":\"msg_2\"}");
                return;
            }
            sendJson(exchange, 404, "{}");
        });

        client().publish("app_1", "job.completed", Map.of("jobId", "2"), null);

        JsonNode body = messageBody.get();
        assertEquals("job.completed", body.path("eventType").asText());
        assertTrue(body.path("channels").isMissingNode());
    }

    private SvixServerClient client() {
        SvixProps props = new SvixProps();
        props.setServerUrl("http://127.0.0.1:" + server.getAddress().getPort());
        props.setJwtSecret("test-secret");
        return new SvixServerClient(props, new SvixTokenService(props));
    }

    private void startServer(ExchangeHandler handler) throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        executor = Executors.newCachedThreadPool(r -> {
            Thread thread = new Thread(r);
            thread.setDaemon(true);
            thread.setName("svix-server-client-test-" + thread.getId());
            return thread;
        });
        server.setExecutor(executor);
        server.createContext("/", handler::handle);
        server.start();
    }

    private JsonNode readJson(HttpExchange exchange) throws IOException {
        String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        return MAPPER.readTree(body);
    }

    private void sendJson(HttpExchange exchange, int status, String body) throws IOException {
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    @FunctionalInterface
    private interface ExchangeHandler {
        void handle(HttpExchange exchange) throws IOException;
    }
}
