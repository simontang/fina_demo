package com.fina.b1s.exception;

import com.fina.b1s.dto.ApiResponse;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.servlet.resource.NoResourceFoundException;

import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GlobalExceptionHandlerTest {

    private final GlobalExceptionHandler handler = new GlobalExceptionHandler();

    @Test
    void noResourceFoundUsesNotFoundEnvelope() {
        ApiResponse<Void> response = handler.handleNoResourceFound(
                new NoResourceFoundException(HttpMethod.GET, "/api/env"));

        assertEquals(404, response.getCode());
        assertEquals("Not found: /api/env", response.getMessage());
    }

    @Test
    void httpStatusCodeExceptionPassesThroughStatusAndBody() {
        String body = "{\"error\":{\"message\":{\"value\":\"bad\"}}}";
        HttpClientErrorException exception = HttpClientErrorException.create(
                HttpStatus.BAD_REQUEST,
                "Bad Request",
                null,
                body.getBytes(StandardCharsets.UTF_8),
                StandardCharsets.UTF_8);

        ResponseEntity<byte[]> response = handler.handleHttpStatusCodeException(exception);

        assertEquals(HttpStatus.BAD_REQUEST, response.getStatusCode());
        assertEquals(body, new String(response.getBody(), StandardCharsets.UTF_8));
    }
}
