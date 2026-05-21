package com.fina.b1s.dto;

import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
public class MailMessageVO {

    private Long id;
    private String provider;
    private String mailbox;
    private String folderName;
    private Long uid;
    private String messageId;
    private String subject;
    private String fromAddress;
    private String toAddresses;
    private String ccAddresses;
    private String sentAt;
    private String receivedAt;
    private Boolean hasAttachments;
    private Integer attachmentCount;
    private String snippet;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<MailAttachmentVO> attachments;
}
