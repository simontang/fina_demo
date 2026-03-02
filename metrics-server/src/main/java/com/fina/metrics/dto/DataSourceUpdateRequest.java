package com.fina.metrics.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * Request body for updating an existing SAP B1 HANA datasource.
 *
 * Password is optional: if left blank/null, the existing encrypted password is kept.
 * All other fields are required and will overwrite the current values.
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

    private String description;

    @NotNull(message = "status is required (1=active, 0=inactive)")
    private Integer status;
}
