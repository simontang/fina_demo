package com.fina.b1s.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("t_mail_message")
public class MailMessage {

    @TableId(type = IdType.AUTO)
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

    private String attachmentSummary;

    private String attachmentText;

    private String purchaseOrderSummary;

    private String agentMessage;

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
}
