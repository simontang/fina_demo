package com.fina.cdp.controller;

import com.fina.cdp.config.TenantResolver;
import com.fina.cdp.dto.ApiResponse;
import com.fina.cdp.dto.PageResponse;
import com.fina.cdp.dto.SegmentDataVO;
import com.fina.cdp.dto.SegmentDefinitionRequest;
import com.fina.cdp.dto.SegmentDefinitionVO;
import com.fina.cdp.dto.SegmentProcessRequest;
import com.fina.cdp.service.SegmentDefinitionService;
import com.fina.cdp.service.SegmentProcessingService;
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

import java.util.List;

@RestController
@RequestMapping("/api/v1/segment-definitions")
@RequiredArgsConstructor
public class SegmentDefinitionController {

    private final TenantResolver tenantResolver;
    private final SegmentDefinitionService definitionService;
    private final SegmentProcessingService processingService;

    @GetMapping
    public ApiResponse<List<SegmentDefinitionVO>> list(HttpServletRequest request) {
        return ApiResponse.ok(definitionService.list(tenantResolver.resolve(request)));
    }

    @GetMapping("/page")
    public ApiResponse<PageResponse<SegmentDefinitionVO>> page(
            HttpServletRequest request,
            @RequestParam(required = false) Integer page,
            @RequestParam(required = false) Integer pageSize,
            @RequestParam(required = false) String keyword) {
        return ApiResponse.ok(definitionService.page(tenantResolver.resolve(request), page, pageSize, keyword));
    }

    @GetMapping("/{id}")
    public ApiResponse<SegmentDefinitionVO> getById(HttpServletRequest request, @PathVariable Long id) {
        return ApiResponse.ok(definitionService.getById(tenantResolver.resolve(request), id));
    }

    @PostMapping
    public ApiResponse<SegmentDefinitionVO> create(
            HttpServletRequest request,
            @Valid @RequestBody SegmentDefinitionRequest body) {
        return ApiResponse.ok(definitionService.create(tenantResolver.resolve(request), body));
    }

    @PutMapping("/{id}")
    public ApiResponse<SegmentDefinitionVO> update(
            HttpServletRequest request,
            @PathVariable Long id,
            @Valid @RequestBody SegmentDefinitionRequest body) {
        return ApiResponse.ok(definitionService.update(tenantResolver.resolve(request), id, body));
    }

    @DeleteMapping("/{id}")
    public ApiResponse<Void> delete(HttpServletRequest request, @PathVariable Long id) {
        definitionService.delete(tenantResolver.resolve(request), id);
        return ApiResponse.ok();
    }

    @PostMapping("/{id}/process")
    public ApiResponse<SegmentDataVO> process(
            HttpServletRequest request,
            @PathVariable Long id,
            @RequestBody(required = false) SegmentProcessRequest body) {
        return ApiResponse.ok(processingService.process(tenantResolver.resolve(request), id, body));
    }
}
