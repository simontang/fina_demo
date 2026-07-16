package com.fina.b1s.document;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Slf4j
@Service
public class DocumentServiceClientImpl implements DocumentParseClient {

    private final HttpClient httpClient;
    private final DocumentServiceProperties properties;
    private final ObjectMapper objectMapper;

    public DocumentServiceClientImpl(@Qualifier("documentServiceHttpClient") HttpClient httpClient,
                                     DocumentServiceProperties properties,
                                     ObjectMapper objectMapper) {
        this.httpClient = httpClient;
        this.properties = properties;
        this.objectMapper = objectMapper;
    }

    @Override
    public ParseResult parse(String fileName, String contentType, byte[] bytes) {
        if (!properties.isConfigured()) {
            return new ParseResult(null, "DISABLED", null, null, null, "document service is not configured");
        }
        if (bytes == null || bytes.length == 0) {
            return new ParseResult(null, "EMPTY", null, null, null, "document bytes are empty");
        }
        try {
            String assetId = uploadAsset(fileName, contentType, bytes);
            String runId = createRun(assetId);
            JsonNode run = waitForRun(runId);
            String status = run.path("status").asText("");
            String engine = run.path("selected_engine").asText(null);
            if (!"succeeded".equals(status)) {
                String error = firstNonBlank(run.path("error_message").asText(null), run.path("error_code").asText(null));
                return new ParseResult(null, status.toUpperCase(), engine, assetId, runId,
                        firstNonBlank(error, "document parse did not succeed"));
            }
            String markdown = downloadMarkdown(runId);
            return new ParseResult(markdown, StringUtils.hasText(markdown) ? "EXTRACTED" : "NO_TEXT",
                    engine, assetId, runId, null);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.warn("Document service parse interrupted file={}: {}", fileName, e.getMessage());
            return new ParseResult(null, "INTERRUPTED", null, null, null, e.getMessage());
        } catch (Exception e) {
            log.warn("Document service parse failed file={}: {}", fileName, e.getMessage(), e);
            return new ParseResult(null, "FAILED", null, null, null, e.getMessage());
        }
    }

    private String uploadAsset(String fileName, String contentType, byte[] bytes) throws Exception {
        String boundary = "b1s-" + UUID.randomUUID();
        byte[] body = multipartBody(boundary, fileName, contentType, bytes);
        HttpRequest request = HttpRequest.newBuilder(uri("/v1/assets"))
                .timeout(readTimeout())
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .header("Accept", "application/json")
                .POST(HttpRequest.BodyPublishers.ofByteArray(body))
                .build();
        JsonNode response = sendJson(request, 201, "upload document asset");
        String assetId = response.path("asset_id").asText(null);
        if (!StringUtils.hasText(assetId)) {
            throw new IOException("document service upload response did not include asset_id");
        }
        return assetId;
    }

    private String createRun(String assetId) throws Exception {
        Map<String, Object> payload = Map.of(
                "operation", "document.parse",
                "engine", firstNonBlank(properties.engine(), "datalab"),
                "inputs", Map.of("source_asset_id", assetId),
                "params", Map.of(
                        "output_formats", List.of("markdown", "json"),
                        "mode", firstNonBlank(properties.mode(), "accurate"),
                        "language_hint", languageHints()
                )
        );
        HttpRequest request = HttpRequest.newBuilder(uri("/v1/runs"))
                .timeout(readTimeout())
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(payload), StandardCharsets.UTF_8))
                .build();
        JsonNode response = sendJson(request, 202, "create document parse run");
        String runId = response.path("run_id").asText(null);
        if (!StringUtils.hasText(runId)) {
            throw new IOException("document service run response did not include run_id");
        }
        return runId;
    }

    private JsonNode waitForRun(String runId) throws Exception {
        long started = System.currentTimeMillis();
        long maxWaitMs = timeoutOrDefault(properties.maxWaitMs(), 180_000);
        long pollIntervalMs = timeoutOrDefault(properties.pollIntervalMs(), 3_000);
        while (true) {
            HttpRequest request = HttpRequest.newBuilder(uri("/v1/runs/" + runId))
                    .timeout(readTimeout())
                    .header("Accept", "application/json")
                    .GET()
                    .build();
            JsonNode response = sendJson(request, 200, "poll document parse run");
            String status = response.path("status").asText("");
            if ("succeeded".equals(status) || "failed".equals(status) || "cancelled".equals(status)) {
                return response;
            }
            if (System.currentTimeMillis() - started > maxWaitMs) {
                throw new IOException("document parse timed out runId=" + runId + " status=" + status);
            }
            Thread.sleep(pollIntervalMs);
        }
    }

    private String downloadMarkdown(String runId) throws Exception {
        HttpRequest request = HttpRequest.newBuilder(uri("/v1/runs/" + runId + "/outputs/markdown"))
                .timeout(readTimeout())
                .header("Accept", "text/markdown")
                .GET()
                .build();
        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        if (response.statusCode() != 200) {
            throw new IOException("download document markdown returned HTTP " + response.statusCode() + ": "
                    + trim(response.body(), 1000));
        }
        return response.body();
    }

    private JsonNode sendJson(HttpRequest request, int expectedStatus, String action) throws Exception {
        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        if (response.statusCode() != expectedStatus) {
            throw new IOException(action + " returned HTTP " + response.statusCode() + ": " + trim(response.body(), 1000));
        }
        return objectMapper.readTree(response.body());
    }

    private URI uri(String path) {
        return URI.create(normalizeBaseUrl(properties.baseUrl()) + path);
    }

    private String normalizeBaseUrl(String baseUrl) {
        String normalized = baseUrl.trim();
        while (normalized.endsWith("/")) {
            normalized = normalized.substring(0, normalized.length() - 1);
        }
        return normalized;
    }

    private Duration readTimeout() {
        return Duration.ofMillis(timeoutOrDefault(properties.readTimeoutMs(), 600_000));
    }

    private long timeoutOrDefault(long value, long defaultValue) {
        return value > 0 ? value : defaultValue;
    }

    private List<String> languageHints() {
        if (!StringUtils.hasText(properties.languageHints())) {
            return List.of();
        }
        return Arrays.stream(properties.languageHints().split(","))
                .map(String::trim)
                .filter(StringUtils::hasText)
                .toList();
    }

    private byte[] multipartBody(String boundary, String fileName, String contentType, byte[] bytes) throws IOException {
        String safeFileName = StringUtils.hasText(fileName) ? fileName : "attachment.bin";
        String safeContentType = StringUtils.hasText(contentType) ? contentType : "application/octet-stream";
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8));
        out.write(("Content-Disposition: form-data; name=\"file\"; filename=\"" + escapeHeader(safeFileName) + "\"\r\n")
                .getBytes(StandardCharsets.UTF_8));
        out.write(("Content-Type: " + safeContentType + "\r\n\r\n").getBytes(StandardCharsets.UTF_8));
        out.write(bytes);
        out.write(("\r\n--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
        return out.toByteArray();
    }

    private String escapeHeader(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private String firstNonBlank(String first, String fallback) {
        return StringUtils.hasText(first) ? first : fallback;
    }

    private String trim(String value, int maxLength) {
        if (value == null || value.length() <= maxLength) {
            return value;
        }
        return value.substring(0, maxLength);
    }
}
