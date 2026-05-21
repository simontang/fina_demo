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
    private String bodyText;
    private String snippet;
    private Boolean orderIntent;
    private String workflowStatus;
    private String workflowThreadId;
    private String workflowRunId;
    private String workflowRequest;
    private String workflowResponse;
    private String workflowError;
    private LocalDateTime workflowTriggeredAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<MailAttachmentVO> attachments;
}
