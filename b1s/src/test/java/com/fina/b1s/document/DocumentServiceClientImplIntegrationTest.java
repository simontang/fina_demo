package com.fina.b1s.document;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.net.http.HttpClient;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DocumentServiceClientImplIntegrationTest {

    @Test
    @EnabledIfEnvironmentVariable(named = "B1S_DOCUMENT_SERVICE_IT", matches = "true")
    void parsesTropicalPurchaseOrderPdfThroughDocumentService() throws Exception {
        Path pdfPath = Path.of(System.getenv().getOrDefault(
                "B1S_DOCUMENT_SERVICE_IT_FILE",
                "/Users/cid/Downloads/FOR SO POSTING/dummy-so-001-tropical.pdf"
        ));
        byte[] bytes = Files.readAllBytes(pdfPath);
        DocumentServiceClientImpl client = new DocumentServiceClientImpl(
                HttpClient.newBuilder()
                        .connectTimeout(Duration.ofSeconds(15))
                        .followRedirects(HttpClient.Redirect.NORMAL)
                        .version(HttpClient.Version.HTTP_1_1)
                        .build(),
                new DocumentServiceProperties(
                        true,
                        System.getenv().getOrDefault("DOCUMENT_SERVICE_BASE_URL", "https://ada.alphafina.cn/api/documents"),
                        "auto",
                        "accurate",
                        "en",
                        15_000,
                        600_000,
                        3_000,
                        180_000
                ),
                new ObjectMapper()
        );

        DocumentParseClient.ParseResult result = client.parse(
                pdfPath.getFileName().toString(),
                "application/pdf",
                bytes
        );

        assertEquals("EXTRACTED", result.status());
        String markdown = result.markdown();
        assertTrue(markdown.contains("TROPICAL ISLAND ENTERPRISES INC."));
        assertTrue(markdown.contains("PO No. 123456789"));
        assertTrue(markdown.contains("Office Laptops"));
        assertTrue(markdown.contains("Ergonomic Chairs"));
        assertTrue(markdown.contains("330,000.00 PHP"));
    }
}
