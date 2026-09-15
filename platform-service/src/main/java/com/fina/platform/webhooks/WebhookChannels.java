package com.fina.platform.webhooks;

import com.fina.platform.exception.ApiException;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

final class WebhookChannels {

    private static final int MAX_CHANNELS = 10;
    private static final int MAX_CHANNEL_LENGTH = 128;
    private static final Pattern SVIX_CHANNEL_PATTERN = Pattern.compile("^[a-zA-Z0-9\\-_.:+]+$");

    private WebhookChannels() {
    }

    static List<String> normalize(List<String> values) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        Set<String> normalized = new LinkedHashSet<>();
        for (String value : values) {
            if (value == null) {
                continue;
            }
            String trimmed = value.trim();
            if (!trimmed.isEmpty()) {
                normalized.add(trimmed);
            }
        }
        if (normalized.size() > MAX_CHANNELS) {
            throw ApiException.badRequest("channels supports at most " + MAX_CHANNELS + " values");
        }
        for (String channel : normalized) {
            if (channel.length() > MAX_CHANNEL_LENGTH || !SVIX_CHANNEL_PATTERN.matcher(channel).matches()) {
                throw ApiException.badRequest(
                        "channel must match Svix channel format: letters, digits, '-', '_', '.', ':', '+', max 128 chars");
            }
        }
        return List.copyOf(normalized);
    }
}
