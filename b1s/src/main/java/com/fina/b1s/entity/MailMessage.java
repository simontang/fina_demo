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

    private String snippet;

    private LocalDateTime createdAt;

    private LocalDateTime updatedAt;
}
