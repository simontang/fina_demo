package com.fina.metrics.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import lombok.Data;

/**
 * Request body for updating an existing SAP B1 HANA datasource.
 *
 * Password is optional: if left blank/null, the existing encrypted password is kept.
 * visibleScopeMode is optional: null preserves the existing mode.
 * Other fields retain their existing replacement semantics.
 */
@Data
public class DataSourceUpdateRequest {

    @NotBlank(message = "name is required")
    private String name;

    @NotBlank(message = "url is required")
    private String url;

    @NotBlank(message = "username is required")
    private String username;

    /**
     * Plain-text password.
     * Leave blank to keep the current password unchanged.
     */
    private String password;

    private String schemaName;

    /** Optional. When absent, inferred from url for backward compatibility. */
    private String sourceType;

    /** Omit or supply null to keep the existing visibility mode. */
    @Pattern(regexp = "ALL|RESTRICTED", message = "visibleScopeMode must be ALL or RESTRICTED")
    private String visibleScopeMode;

    private String description;

    @NotNull(message = "status is required (1=active, 0=inactive)")
    private Integer status;
}
