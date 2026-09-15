package com.fina.platform.webhooks;

import com.fina.platform.exception.ApiException;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class WebhookChannelsTest {

    @Test
    void normalizesChannelsUsingSvixRules() {
        assertEquals(
                List.of("vip", "ops.team+1", "project:123"),
                WebhookChannels.normalize(List.of(" vip ", "ops.team+1", "vip", "", "project:123")));
    }

    @Test
    void rejectsInvalidChannelNamesBeforeCallingSvix() {
        assertThrows(ApiException.class, () -> WebhookChannels.normalize(List.of("bad channel")));
    }

    @Test
    void rejectsTooManyChannelsBeforeCallingSvix() {
        assertThrows(ApiException.class, () -> WebhookChannels.normalize(List.of(
                "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11")));
    }
}
