package com.fina.b1s.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("t_mail_attachment")
public class MailAttachment {

    @TableId(type = IdType.AUTO)
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
