package com.fina.b1s.document;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

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
import static org.junit.jupiter.api.Assertions.assertTrue;

class DocumentServiceClientImplTest {

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
    void stripsMailMimeParametersBeforeUploadingAsset() throws Exception {
        AtomicReference<String> assetRequestBody = new AtomicReference<>();
        startServer(assetRequestBody);

        DocumentServiceClientImpl client = new DocumentServiceClientImpl(
                HttpClient.newHttpClient(),
                properties("http://127.0.0.1:" + server.getAddress().getPort()),
                new ObjectMapper()
        );

        DocumentParseClient.ParseResult result = client.parse(
                "dummy-so-001-tropical.pdf",
                "application/pdf; name=\"dummy-so-001-tropical.pdf\"",
                "%PDF-1.7".getBytes(StandardCharsets.ISO_8859_1)
        );

        assertEquals("EXTRACTED", result.status());
        String body = assetRequestBody.get();
        assertTrue(body.contains("filename=\"dummy-so-001-tropical.pdf\""));
        assertTrue(body.contains("Content-Type: application/pdf\r\n\r\n"));
        assertFalse(body.contains("Content-Type: application/pdf;"));
    }

    @Test
    void infersPdfContentTypeFromFileBytesWhenMailHeadersAreGeneric() throws Exception {
        AtomicReference<String> assetRequestBody = new AtomicReference<>();
        startServer(assetRequestBody);

        DocumentServiceClientImpl client = new DocumentServiceClientImpl(
                HttpClient.newHttpClient(),
                properties("http://127.0.0.1:" + server.getAddress().getPort()),
                new ObjectMapper()
        );

        DocumentParseClient.ParseResult result = client.parse(
                "download",
                "application/octet-stream",
                "%PDF-1.7".getBytes(StandardCharsets.ISO_8859_1)
        );

        assertEquals("EXTRACTED", result.status());
        String body = assetRequestBody.get();
        assertTrue(body.contains("filename=\"download.pdf\""));
        assertTrue(body.contains("Content-Type: application/pdf\r\n\r\n"));
    }

    private void startServer(AtomicReference<String> assetRequestBody) throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        executor = Executors.newCachedThreadPool(r -> {
            Thread thread = new Thread(r);
            thread.setDaemon(true);
            thread.setName("document-service-client-test-" + thread.getId());
            return thread;
        });
        server.setExecutor(executor);
        server.createContext("/", exchange -> {
            String path = exchange.getRequestURI().getPath();
            if ("/v1/assets".equals(path)) {
                assetRequestBody.set(readRequestBody(exchange));
                sendJson(exchange, 201, "{\"asset_id\":\"asset_1\"}");
            } else if ("/v1/runs".equals(path)) {
                sendJson(exchange, 202, "{\"run_id\":\"run_1\"}");
            } else if ("/v1/runs/run_1".equals(path)) {
                sendJson(exchange, 200, """
                        {"run_id":"run_1","status":"succeeded","selected_engine":"datalab","outputs":{"markdown":"asset_markdown"}}
                        """);
            } else if ("/v1/runs/run_1/outputs/markdown".equals(path)) {
                sendText(exchange, 200, "# Parsed PO");
            } else {
                sendText(exchange, 404, "not found");
            }
        });
        server.start();
    }

    private String readRequestBody(HttpExchange exchange) throws IOException {
        return new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.ISO_8859_1);
    }

    private void sendJson(HttpExchange exchange, int status, String body) throws IOException {
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        sendText(exchange, status, body);
    }

    private void sendText(HttpExchange exchange, int status, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    private DocumentServiceProperties properties(String baseUrl) {
        return new DocumentServiceProperties(
                true,
                baseUrl,
                "datalab",
                "accurate",
                "en",
                1_000,
                5_000,
                10,
                5_000
        );
    }
}
