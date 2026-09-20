package com.fina.metrics.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.util.List;

@Data
public class DataSourceApiKeyRequest {

    @NotBlank(message = "keyName is required")
    private String keyName;

    /**
     * Optional caller-provided key. When omitted, the service generates one and
     * returns it once in rawKey.
     */
    private String rawKey;

    private List<String> permissions;
}
