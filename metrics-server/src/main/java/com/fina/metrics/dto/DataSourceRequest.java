package com.fina.metrics.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import lombok.Data;

/**
 * Request body for creating or updating a datasource.
 */
@Data
public class DataSourceRequest {

    @NotBlank(message = "name is required")
    private String name;

    @NotBlank(message = "url is required")
    private String url;

    @NotBlank(message = "username is required")
    private String username;

    /** Plain-text password — will be encrypted before persisting */
    @NotBlank(message = "password is required")
    private String password;

    private String schemaName;

    /** Optional. When absent, inferred from url for backward compatibility. */
    private String sourceType;

    @Pattern(regexp = "ALL|RESTRICTED", message = "visibleScopeMode must be ALL or RESTRICTED")
    private String visibleScopeMode = "RESTRICTED";

    private String description;

    @NotNull(message = "status is required")
    private Integer status;
}
