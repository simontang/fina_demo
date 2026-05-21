package com.fina.b1s.controller;

import com.fina.b1s.dto.ApiResponse;
import com.fina.b1s.dto.MailMessageVO;
import com.fina.b1s.mail.MailIngestService;
import com.fina.b1s.mail.MailQueryService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/mail")
@RequiredArgsConstructor
public class MailController {

    private final MailQueryService mailQueryService;
    private final MailIngestService mailIngestService;

    @GetMapping("/messages")
    public ApiResponse<List<MailMessageVO>> listRecent(
            @RequestParam(defaultValue = "20") int limit) {
        return ApiResponse.ok(mailQueryService.listRecent(limit));
    }

    @GetMapping("/messages/{id}")
    public ApiResponse<MailMessageVO> getById(@PathVariable Long id) {
        return ApiResponse.ok(mailQueryService.getById(id));
    }

    @PostMapping("/poll")
    public ApiResponse<Map<String, Object>> pollNow() {
        int processed = mailIngestService.pollInbox();
        return ApiResponse.ok(Map.of("processed", processed));
    }
}
