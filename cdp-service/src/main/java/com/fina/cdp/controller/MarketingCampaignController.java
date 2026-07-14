package com.fina.cdp.controller;

import com.fina.cdp.config.TenantResolver;
import com.fina.cdp.dto.ApiResponse;
import com.fina.cdp.dto.MarketingCampaignRequest;
import com.fina.cdp.dto.MarketingCampaignScheduleRequest;
import com.fina.cdp.dto.MarketingCampaignVO;
import com.fina.cdp.dto.PageResponse;
import com.fina.cdp.service.MarketingCampaignService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/marketing-campaigns")
@RequiredArgsConstructor
public class MarketingCampaignController {

    private final TenantResolver tenantResolver;
    private final MarketingCampaignService campaignService;

    @GetMapping
    public ApiResponse<PageResponse<MarketingCampaignVO>> list(
            HttpServletRequest request,
            @RequestParam(required = false) String type,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) Integer page,
            @RequestParam(required = false) Integer pageSize) {
        return ApiResponse.ok(campaignService.list(tenantResolver.resolve(request), type, status, page, pageSize));
    }

    @GetMapping("/{id}")
    public ApiResponse<MarketingCampaignVO> getById(HttpServletRequest request, @PathVariable Long id) {
        return ApiResponse.ok(campaignService.getById(tenantResolver.resolve(request), id));
    }

    @PostMapping
    public ApiResponse<MarketingCampaignVO> create(
            HttpServletRequest request,
            @Valid @RequestBody MarketingCampaignRequest body) {
        return ApiResponse.ok(campaignService.create(tenantResolver.resolve(request), body));
    }

    @PutMapping("/{id}")
    public ApiResponse<MarketingCampaignVO> update(
            HttpServletRequest request,
            @PathVariable Long id,
            @Valid @RequestBody MarketingCampaignRequest body) {
        return ApiResponse.ok(campaignService.update(tenantResolver.resolve(request), id, body));
    }

    @DeleteMapping("/{id}")
    public ApiResponse<Void> delete(HttpServletRequest request, @PathVariable Long id) {
        campaignService.delete(tenantResolver.resolve(request), id);
        return ApiResponse.ok();
    }

    @PostMapping("/{id}/start")
    public ApiResponse<MarketingCampaignVO> start(HttpServletRequest request, @PathVariable Long id) {
        return ApiResponse.ok(campaignService.start(tenantResolver.resolve(request), id));
    }

    @PostMapping("/{id}/stop")
    public ApiResponse<MarketingCampaignVO> stop(HttpServletRequest request, @PathVariable Long id) {
        return ApiResponse.ok(campaignService.stop(tenantResolver.resolve(request), id));
    }

    @PostMapping("/{id}/schedule")
    public ApiResponse<MarketingCampaignVO> schedule(
            HttpServletRequest request,
            @PathVariable Long id,
            @RequestBody(required = false) MarketingCampaignScheduleRequest body) {
        return ApiResponse.ok(campaignService.schedule(tenantResolver.resolve(request), id, body));
    }
}
