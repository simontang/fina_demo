package com.fina.b1s.mail;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicBoolean;

@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mail.listener", name = "enabled", havingValue = "true")
public class MailPollScheduler {

    private final MailIngestService mailIngestService;
    private final AtomicBoolean running = new AtomicBoolean(false);

    @Scheduled(fixedDelayString = "${mail.listener.poll-interval-ms:60000}")
    public void poll() {
        if (!running.compareAndSet(false, true)) {
            log.info("Previous mail polling task is still running; skip this tick");
            return;
        }
        try {
            mailIngestService.pollInbox();
        } finally {
            running.set(false);
        }
    }
}
