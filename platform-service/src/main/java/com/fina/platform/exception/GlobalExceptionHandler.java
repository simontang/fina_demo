package com.fina.platform.exception;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MissingRequestValueException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.support.MissingServletRequestPartException;

import java.util.Map;

@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    /** Malformed requests (missing file part / parameter) are 400, not 500. */
    @ExceptionHandler({MissingServletRequestPartException.class,
            MissingServletRequestParameterException.class,
            MissingRequestValueException.class})
    public ResponseEntity<Map<String, Object>> handleMissingValue(Exception e) {
        return ResponseEntity.badRequest()
                .body(Map.of("code", "BAD_REQUEST", "message", String.valueOf(e.getMessage())));
    }

    /** Wrong content type (e.g. uploading without multipart) is 415, not 500. */
    @ExceptionHandler({org.springframework.web.multipart.MultipartException.class,
            org.springframework.web.HttpMediaTypeNotSupportedException.class,
            org.springframework.web.HttpMediaTypeNotAcceptableException.class})
    public ResponseEntity<Map<String, Object>> handleMediaType(Exception e) {
        return ResponseEntity.status(415)
                .body(Map.of("code", "UNSUPPORTED_MEDIA_TYPE", "message", String.valueOf(e.getMessage())));
    }

    @ExceptionHandler(ApiException.class)
    public ResponseEntity<Map<String, Object>> handleApi(ApiException e) {
        return ResponseEntity.status(e.getStatus())
                .body(Map.of("code", e.getCode(), "message", e.getMessage()));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<Map<String, Object>> handleUnexpected(Exception e) {
        log.error("unexpected error", e);
        return ResponseEntity.status(500)
                .body(Map.of("code", "INTERNAL_ERROR", "message", String.valueOf(e.getMessage())));
    }
}
