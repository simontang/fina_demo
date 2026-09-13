package com.fina.platform.webhooks;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

/**
 * Connection to the self-hosted svix-server container. The wrapper is the
 * only caller — business code never sees Svix concepts.
 */
@Data
@Configuration
@ConfigurationProperties(prefix = "svix")
public class SvixProps {

    /** svix-server base URL (compose-internal). */
    private String serverUrl;

    /** Shared JWT secret — must equal svix-server's SVIX_JWT_SECRET. */
    private String jwtSecret;

    /** Org identity embedded in the minted token (one deployment = one org). */
    private String orgId = "org_23platformdemo0000000000";
}
