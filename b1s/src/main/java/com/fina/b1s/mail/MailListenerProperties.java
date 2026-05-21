package com.fina.b1s.mail;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.time.Duration;

@ConfigurationProperties(prefix = "mail.listener")
public record MailListenerProperties(
        boolean enabled,
        String imapHost,
        int imapPort,
        String username,
        String password,
        String folder,
        Duration pollInterval,
        int batchSize,
        boolean markSeen
) {
}
