package com.fina.b1s.document;

public interface DocumentParseClient {

    ParseResult parse(String fileName, String contentType, byte[] bytes);

    record ParseResult(
            String markdown,
            String status,
            String engine,
            String assetId,
            String runId,
            String errorMessage
    ) {
    }
}
