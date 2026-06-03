package com.fina.b1s.mail;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.b1s.agent.AgentRunClient;
import com.fina.b1s.agent.AgentRunProperties;
import com.fina.b1s.agent.AgentRunRequest;
import com.fina.b1s.agent.AgentRunResult;
import com.fina.b1s.entity.MailMessage;
import com.fina.b1s.mapper.MailAttachmentMapper;
import com.fina.b1s.mapper.MailMessageMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Map;
import java.util.concurrent.Executor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class MailWorkflowServiceImplTest {

    @Mock
    private MailIntentService mailIntentService;
    @Mock
    private AgentRunClient agentRunClient;
    @Mock
    private MailMessageMapper mailMessageMapper;
    @Mock
    private MailAttachmentMapper mailAttachmentMapper;

    @Test
    void dispatchSendsMailHeadersInAgentMessageAndCustomConfig() {
        MailWorkflowServiceImpl service = new MailWorkflowServiceImpl(
                mailIntentService,
                agentRunClient,
                properties(),
                mailMessageMapper,
                mailAttachmentMapper,
                new ObjectMapper(),
                sameThreadExecutor()
        );

        MailMessage mailMessage = new MailMessage();
        mailMessage.setId(42L);
        mailMessage.setMailbox("sales@alphafina.cn");
        mailMessage.setUid(1001L);
        mailMessage.setMessageId("<msg-1@example.com>");
        mailMessage.setFromAddress("sender@example.com");
        mailMessage.setToAddresses("buyer@example.com;ops@example.com");
        mailMessage.setSubject("PO-20260603");
        mailMessage.setBodyText("Please create the order.");

        when(agentRunClient.run(any())).thenReturn(new AgentRunResult(true, "200", "run-1", "{\"id\":\"run-1\"}", null));

        AgentRunResult result = service.dispatch(mailMessage);

        ArgumentCaptor<AgentRunRequest> captor = ArgumentCaptor.forClass(AgentRunRequest.class);
        verify(agentRunClient).run(captor.capture());
        verify(mailMessageMapper).updateById(any(MailMessage.class));

        AgentRunRequest request = captor.getValue();
        assertTrue(result.sent());
        assertNotNull(request);
        assertTrue(request.message().contains("Mail From:\nsender@example.com"));
        assertTrue(request.message().contains("Mail To:\nbuyer@example.com;ops@example.com"));
        assertTrue(request.message().contains("Mail Subject:\nPO-20260603"));
        assertTrue(request.message().contains("Mail Body:\nPlease create the order."));

        Map<String, Object> customRunConfig = request.customRunConfig();
        assertNotNull(customRunConfig);
        assertEquals("sender@example.com", customRunConfig.get("mail_from"));
        assertEquals("buyer@example.com;ops@example.com", customRunConfig.get("mail_to"));
        assertEquals("PO-20260603", customRunConfig.get("mail_subject"));
    }

    private AgentRunProperties properties() {
        return new AgentRunProperties(
                true,
                "http://agent.example.com",
                "Bearer test-token",
                "tenant_1",
                "workspace_1",
                "project_1",
                "assistant-1",
                "demo",
                false,
                false,
                1000,
                5000
        );
    }

    private Executor sameThreadExecutor() {
        return Runnable::run;
    }
}
