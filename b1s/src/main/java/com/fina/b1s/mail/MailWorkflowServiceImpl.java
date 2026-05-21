package com.fina.b1s.mail;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.b1s.agent.AgentRunClient;
import com.fina.b1s.agent.AgentRunProperties;
import com.fina.b1s.agent.AgentRunRequest;
import com.fina.b1s.agent.AgentRunResult;
import com.fina.b1s.entity.MailMessage;
import com.fina.b1s.mapper.MailMessageMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.time.LocalDateTime;
import java.util.LinkedHashMap;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class MailWorkflowServiceImpl implements MailWorkflowService {

    private final MailIntentService mailIntentService;
    private final AgentRunClient agentRunClient;
    private final AgentRunProperties agentRunProperties;
    private final MailMessageMapper messageMapper;
    private final ObjectMapper objectMapper;

    @Override
    public AgentRunResult dispatchIfOrderIntent(MailMessage mailMessage) {
        if (!mailIntentService.isOrderIntent(mailMessage)) {
            updateWorkflow(mailMessage, "NOT_ORDER_INTENT", null, null, null, null, null);
            return new AgentRunResult(false, "NOT_ORDER_INTENT", null, null, null);
        }
        return dispatch(mailMessage);
    }

    @Override
    public AgentRunResult dispatch(MailMessage mailMessage) {
        if (!StringUtils.hasText(agentRunProperties.assistantId())) {
            AgentRunResult result = new AgentRunResult(false, "DISABLED", null, null, "assistant_id is missing");
            updateWorkflow(mailMessage, "DISABLED", null, null, null, null, result.errorMessage());
            return result;
        }

        String body = StringUtils.hasText(mailMessage.getBodyText())
                ? mailMessage.getBodyText()
                : mailMessage.getSnippet();
        if (!StringUtils.hasText(body)) {
            AgentRunResult result = new AgentRunResult(false, "NO_BODY", null, null, "mail body is empty");
            updateWorkflow(mailMessage, "NO_BODY", null, null, null, null, result.errorMessage());
            return result;
        }

        String threadId = buildThreadId(mailMessage);
        Map<String, Object> custom = new LinkedHashMap<>();
        custom.put("mail_message_id", mailMessage.getId());
        custom.put("mail_uid", mailMessage.getUid());
        custom.put("mail_subject", mailMessage.getSubject());
        custom.put("mail_from", mailMessage.getFromAddress());
        custom.put("mail_to", mailMessage.getToAddresses());
        custom.put("mail_provider", mailMessage.getProvider());

        AgentRunRequest request = AgentRunRequest.builder()
                .assistantId(agentRunProperties.assistantId())
                .threadId(threadId)
                .message(body)
                .streaming(agentRunProperties.streaming())
                .background(agentRunProperties.background())
                .mode(agentRunProperties.mode())
                .customRunConfig(custom)
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

    private String serialize(AgentRunRequest request) {
        try {
            return objectMapper.writeValueAsString(request);
        } catch (JsonProcessingException e) {
            return null;
        }
    }
}
