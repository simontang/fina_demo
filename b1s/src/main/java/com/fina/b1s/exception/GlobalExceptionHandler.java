package com.fina.b1s.exception;

import com.fina.b1s.dto.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.servlet.resource.NoResourceFoundException;

import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final List<String> RESPONSE_HOP_BY_HOP_HEADERS = List.of(
            "connection", "content-length", "transfer-encoding",
            "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "trailers", "upgrade");

    @ExceptionHandler(MethodArgumentNotValidException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public ApiResponse<Void> handleValidation(MethodArgumentNotValidException ex) {
        String message = ex.getBindingResult().getFieldErrors().stream()
                .map(FieldError::getDefaultMessage)
                .collect(Collectors.joining("; "));
        log.warn("Validation failed on {}: {}", ex.getObjectName(), message);
        return ApiResponse.badRequest(message);
    }

    @ExceptionHandler(IllegalArgumentException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public ApiResponse<Void> handleIllegalArgument(IllegalArgumentException ex) {
        log.warn("Bad request: {}", ex.getMessage());
        return ApiResponse.badRequest(ex.getMessage());
    }

    @ExceptionHandler(IllegalStateException.class)
    @ResponseStatus(HttpStatus.CONFLICT)
    public ApiResponse<Void> handleIllegalState(IllegalStateException ex) {
        log.warn("Conflict: {}", ex.getMessage());
        return ApiResponse.fail(409, ex.getMessage());
    }

    @ExceptionHandler(NoResourceFoundException.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public ApiResponse<Void> handleNoResourceFound(NoResourceFoundException ex) {
        log.warn("Not found: {}", ex.getResourcePath());
        return ApiResponse.notFound("Not found: " + ex.getResourcePath());
    }

    @ExceptionHandler(HttpStatusCodeException.class)
    public ResponseEntity<byte[]> handleHttpStatusCodeException(HttpStatusCodeException ex) {
        log.warn("HTTP status exception: status={} body={}", ex.getStatusCode().value(), ex.getResponseBodyAsString());
        return ResponseEntity.status(ex.getStatusCode())
                .headers(copyResponseHeaders(ex.getResponseHeaders()))
                .body(ex.getResponseBodyAsByteArray());
    }

    @ExceptionHandler(Exception.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public ApiResponse<Void> handleGeneral(Exception ex) {
        log.error("Unhandled error", ex);
        return ApiResponse.fail("Internal server error: " + ex.getMessage());
    }

    private static HttpHeaders copyResponseHeaders(HttpHeaders source) {
        HttpHeaders headers = new HttpHeaders();
        if (source == null) {
            return headers;
        }
        source.forEach((name, values) -> {
            if (!RESPONSE_HOP_BY_HOP_HEADERS.contains(name.toLowerCase())) {
                headers.put(name, values);
            }
        });
        return headers;
    }
}
