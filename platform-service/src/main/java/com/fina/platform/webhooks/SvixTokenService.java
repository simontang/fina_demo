package com.fina.platform.webhooks;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Base64;

/**
 * Mints the bearer token svix-server expects: an HS256 JWT whose `sub` is the
 * org id, signed with the shared SVIX_JWT_SECRET. Avoids a one-shot
 * `svix-server jwt generate` step at deploy time.
 */
@Service
@RequiredArgsConstructor
public class SvixTokenService {

    private static final long TEN_YEARS_SECONDS = 10L * 365 * 24 * 3600;

    private final SvixProps props;

    private volatile String cached;

    public String bearerToken() {
        if (cached == null) {
            synchronized (this) {
                if (cached == null) {
                    cached = mint(props.getOrgId(), Instant.now().getEpochSecond() + TEN_YEARS_SECONDS);
                }
            }
        }
        return cached;
    }

    String mint(String orgId, long expiresAtEpochSeconds) {
        String header = base64Url("{\"alg\":\"HS256\",\"typ\":\"JWT\"}");
        String payload = base64Url("{\"sub\":\"" + orgId + "\",\"exp\":" + expiresAtEpochSeconds + "}");
        String signingInput = header + "." + payload;
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(props.getJwtSecret().getBytes(StandardCharsets.UTF_8), "HmacSHA256"));
            String signature = base64Url(mac.doFinal(signingInput.getBytes(StandardCharsets.UTF_8)));
            return signingInput + "." + signature;
        } catch (Exception e) {
            throw new IllegalStateException("failed to mint svix token", e);
        }
    }

    private String base64Url(String s) {
        return Base64.getUrlEncoder().withoutPadding()
                .encodeToString(s.getBytes(StandardCharsets.UTF_8));
    }

    private String base64Url(byte[] bytes) {
        return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
    }
}
