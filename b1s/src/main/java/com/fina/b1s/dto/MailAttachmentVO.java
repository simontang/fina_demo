package com.fina.b1s.dto;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class MailAttachmentVO {

    private Long id;
    private Long mailMessageId;
    private String fileName;
    private String contentType;
    private Long sizeBytes;
    private String tosBucket;
    private String tosKey;
    private String tosUrl;
    private String uploadStatus;
    private String errorMessage;
    private LocalDateTime createdAt;
}
