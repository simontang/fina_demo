package com.fina.cdp.service;

import com.fina.cdp.dto.MarketingCampaignRequest;
import com.fina.cdp.dto.MarketingCampaignScheduleRequest;
import com.fina.cdp.dto.MarketingCampaignVO;
import com.fina.cdp.dto.PageResponse;

public interface MarketingCampaignService {

    PageResponse<MarketingCampaignVO> list(String tenantId, String type, String status, Integer page, Integer pageSize);

    MarketingCampaignVO getById(String tenantId, Long id);

    MarketingCampaignVO create(String tenantId, MarketingCampaignRequest request);

    MarketingCampaignVO update(String tenantId, Long id, MarketingCampaignRequest request);

    void delete(String tenantId, Long id);

    MarketingCampaignVO start(String tenantId, Long id);

    MarketingCampaignVO stop(String tenantId, Long id);

    MarketingCampaignVO schedule(String tenantId, Long id, MarketingCampaignScheduleRequest request);

    int transitionDueCampaigns();
}
