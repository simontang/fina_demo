package com.fina.cdp.controller;

import com.fina.cdp.config.TenantResolver;
import com.fina.cdp.dto.ApiResponse;
import com.fina.cdp.dto.PageResponse;
import com.fina.cdp.dto.SegmentDataRequest;
import com.fina.cdp.dto.SegmentDataVO;
import com.fina.cdp.service.SegmentDataService;
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
@RequestMapping("/api/v1/segment-data")
@RequiredArgsConstructor
public class SegmentDataController {

    private final TenantResolver tenantResolver;
    private final SegmentDataService dataService;

    @GetMapping
    public ApiResponse<PageResponse<SegmentDataVO>> list(
            HttpServletRequest request,
            @RequestParam(required = false) Long definitionId,
            @RequestParam(required = false) Integer page,
            @RequestParam(required = false) Integer pageSize) {
        return ApiResponse.ok(dataService.list(tenantResolver.resolve(request), definitionId, page, pageSize));
    }

    @GetMapping("/{id}")
    public ApiResponse<SegmentDataVO> getById(HttpServletRequest request, @PathVariable Long id) {
        return ApiResponse.ok(dataService.getById(tenantResolver.resolve(request), id));
    }

    @PostMapping
    public ApiResponse<SegmentDataVO> create(
            HttpServletRequest request,
            @Valid @RequestBody SegmentDataRequest body) {
        return ApiResponse.ok(dataService.create(tenantResolver.resolve(request), body));
    }

    @PutMapping("/{id}")
    public ApiResponse<SegmentDataVO> update(
            HttpServletRequest request,
            @PathVariable Long id,
            @Valid @RequestBody SegmentDataRequest body) {
        return ApiResponse.ok(dataService.update(tenantResolver.resolve(request), id, body));
    }

    @DeleteMapping("/{id}")
    public ApiResponse<Void> delete(HttpServletRequest request, @PathVariable Long id) {
        dataService.delete(tenantResolver.resolve(request), id);
        return ApiResponse.ok();
    }
}
