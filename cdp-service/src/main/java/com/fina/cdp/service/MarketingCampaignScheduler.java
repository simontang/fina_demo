package com.fina.cdp.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "cdp.campaign-scheduler", name = "enabled", havingValue = "true", matchIfMissing = true)
public class MarketingCampaignScheduler {

    private final MarketingCampaignService campaignService;

    @Scheduled(fixedDelayString = "${cdp.campaign-scheduler.fixed-delay-ms:60000}")
    public void transitionDueCampaigns() {
        int updated = campaignService.transitionDueCampaigns();
        if (updated > 0) {
            log.info("Transitioned {} marketing campaigns by schedule", updated);
        }
    }
}
