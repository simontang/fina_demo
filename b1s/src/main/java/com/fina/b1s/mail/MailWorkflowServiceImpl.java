package com.fina.b1s.mail;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.b1s.agent.AgentInboundRequest;
import com.fina.b1s.agent.AgentRunClient;
import com.fina.b1s.agent.AgentRunProperties;
import com.fina.b1s.agent.AgentRunResult;
import com.fina.b1s.entity.MailMessage;
import com.fina.b1s.entity.MailAttachment;
import com.fina.b1s.mapper.MailAttachmentMapper;
import com.fina.b1s.mapper.MailMessageMapper;
import jakarta.mail.internet.InternetAddress;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.time.LocalDateTime;
import java.util.List;
import java.util.concurrent.Executor;

@Slf4j
@Service
@RequiredArgsConstructor
public class MailWorkflowServiceImpl implements MailWorkflowService {

    private final AgentRunClient agentRunClient;
    private final AgentRunProperties agentRunProperties;
    private final MailMessageMapper messageMapper;
    private final MailAttachmentMapper attachmentMapper;
    private final ObjectMapper objectMapper;
    @Qualifier("mailDispatchExecutor")
    private final Executor mailDispatchExecutor;

    @Override
    public AgentRunResult dispatchIfOrderIntent(MailMessage mailMessage) {
        return dispatch(mailMessage);
    }

    @Override
    public void dispatchAsyncIfOrderIntent(MailMessage mailMessage) {
        updateWorkflow(mailMessage, "QUEUED", null, null, null, null, null);
        mailDispatchExecutor.execute(() -> {
            try {
                MailMessage fresh = messageMapper.selectById(mailMessage.getId());
                if (fresh == null) {
                    log.warn("Mail message disappeared before async dispatch id={}", mailMessage.getId());
                    return;
                }
                dispatch(fresh);
            } catch (Exception e) {
                log.warn("Async mail dispatch failed id={}: {}", mailMessage.getId(), e.getMessage(), e);
                MailMessage fresh = messageMapper.selectById(mailMessage.getId());
                if (fresh != null) {
                    updateWorkflow(fresh, "ERROR", fresh.getWorkflowThreadId(), fresh.getWorkflowRunId(),
                            fresh.getWorkflowRequest(), fresh.getWorkflowResponse(), e.getMessage());
                }
            }
        });
    }

    @Override
    public AgentRunResult dispatch(MailMessage mailMessage) {
        if (!agentRunProperties.isConfigured()) {
            AgentRunResult result = new AgentRunResult(false, "DISABLED", null, null, "agent inbound dispatch is not configured");
            updateWorkflow(mailMessage, "DISABLED", null, null, null, null, result.errorMessage());
            return result;
        }

        String body = buildAgentMessage(mailMessage);
        if (!StringUtils.hasText(body)) {
            AgentRunResult result = new AgentRunResult(false, "NO_BODY", null, null, "mail body is empty");
            updateWorkflow(mailMessage, "NO_BODY", null, null, null, null, result.errorMessage());
            return result;
        }

        String threadId = buildThreadId(mailMessage);
        AgentInboundRequest request = AgentInboundRequest.builder()
                .channel("email")
                .channelInstallationId(agentRunProperties.channelInstallationId())
                .tenantId(agentRunProperties.tenantId())
                .sender(new AgentInboundRequest.Sender(resolveSenderId(mailMessage)))
                .content(new AgentInboundRequest.Content(body))
                .build();

        AgentRunResult result = agentRunClient.run(request);
        updateWorkflow(mailMessage,
                result.sent() ? "DISPATCHED" : result.status(),
                threadId,
                result.runId(),
                serialize(request),
                result.rawResponse(),
                result.errorMessage());
        return result;
    }

    private void updateWorkflow(MailMessage mailMessage,
                                String status,
                                String threadId,
                                String runId,
                                String requestJson,
                                String responseJson,
                                String errorMessage) {
        MailMessage update = new MailMessage();
        BeanUtils.copyProperties(mailMessage, update);
        update.setWorkflowStatus(status);
        update.setWorkflowThreadId(threadId);
        update.setWorkflowRunId(runId);
        update.setWorkflowRequest(requestJson);
        update.setWorkflowResponse(responseJson);
        update.setWorkflowError(errorMessage);
        update.setWorkflowTriggeredAt(LocalDateTime.now());
        update.setUpdatedAt(LocalDateTime.now());
        messageMapper.updateById(update);
        BeanUtils.copyProperties(update, mailMessage);
    }

    private String buildThreadId(MailMessage mailMessage) {
        if (StringUtils.hasText(mailMessage.getMessageId())) {
            return "mail:" + mailMessage.getMessageId().replaceAll("[^A-Za-z0-9_-]", "_");
        }
        if (mailMessage.getUid() != null) {
            return "mail:" + mailMessage.getMailbox() + ":" + mailMessage.getUid();
        }
        return "mail:" + mailMessage.getId();
    }

    private String serialize(AgentInboundRequest request) {
        try {
            return objectMapper.writeValueAsString(request);
        } catch (JsonProcessingException e) {
            return null;
        }
    }

    private String buildAgentMessage(MailMessage mailMessage) {
        StringBuilder sb = new StringBuilder();
        appendSection(sb, "Mail From", mailMessage.getFromAddress());
        appendSection(sb, "Mail Subject", mailMessage.getSubject());
        appendSection(sb, "Attachment Markdown", loadAttachmentMarkdown(mailMessage));
        String result = sb.toString().trim();
        return StringUtils.hasText(result) ? result : null;
    }

    private String resolveSenderId(MailMessage mailMessage) {
        String fromAddress = mailMessage.getFromAddress();
        if (!StringUtils.hasText(fromAddress)) {
            return null;
        }
        try {
            InternetAddress[] addresses = InternetAddress.parse(fromAddress, false);
            if (addresses.length > 0 && StringUtils.hasText(addresses[0].getAddress())) {
                return addresses[0].getAddress().trim();
            }
        } catch (Exception e) {
            log.debug("Failed to parse sender address '{}': {}", fromAddress, e.getMessage());
        }
        return fromAddress.trim();
    }

    private String loadAttachmentMarkdown(MailMessage mailMessage) {
        Long mailMessageId = mailMessage.getId();
        if (mailMessageId == null) {
            return mailMessage.getAttachmentText();
        }
        List<MailAttachment> attachments = attachmentMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<MailAttachment>()
                        .eq(MailAttachment::getMailMessageId, mailMessageId)
                        .orderByAsc(MailAttachment::getId)
        );
        StringBuilder sb = new StringBuilder();
        if (attachments != null) {
            for (MailAttachment attachment : attachments) {
                appendAttachmentText(sb, attachment);
            }
        }
        if (sb.isEmpty() && StringUtils.hasText(mailMessage.getAttachmentText())) {
            sb.append(mailMessage.getAttachmentText().trim());
        }
        String result = sb.toString().trim();
        return StringUtils.hasText(result) ? result : null;
    }

    private void appendAttachmentText(StringBuilder sb, MailAttachment attachment) {
        if (!StringUtils.hasText(attachment.getExtractedText())) {
            return;
        }
        if (!sb.isEmpty()) {
            sb.append("\n\n");
        }
        sb.append("Attachment: ")
                .append(firstNonBlank(attachment.getFileName(), String.valueOf(attachment.getId())))
                .append("\n\n")
                .append(attachment.getExtractedText().trim());
    }

    private void appendSection(StringBuilder sb, String title, String value) {
        if (!StringUtils.hasText(value)) {
            return;
        }
        if (!sb.isEmpty()) {
            sb.append("\n\n");
        }
        sb.append(title).append(":\n").append(value.trim());
    }

    private String firstNonBlank(String first, String fallback) {
        return StringUtils.hasText(first) ? first : fallback;
    }
}
