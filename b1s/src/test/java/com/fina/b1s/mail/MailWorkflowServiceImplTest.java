package com.fina.b1s.mail;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fina.b1s.agent.AgentInboundRequest;
import com.fina.b1s.agent.AgentRunClient;
import com.fina.b1s.agent.AgentRunProperties;
import com.fina.b1s.agent.AgentRunResult;
import com.fina.b1s.entity.MailAttachment;
import com.fina.b1s.entity.MailMessage;
import com.fina.b1s.mapper.MailAttachmentMapper;
import com.fina.b1s.mapper.MailMessageMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.concurrent.Executor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
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
    void dispatchSendsInboundPayloadWithParsedSenderAndSerializedWorkflowRequest() {
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
        mailMessage.setFromAddress("Sender Name <sender@example.com>");
        mailMessage.setToAddresses("buyer@example.com;ops@example.com");
        mailMessage.setSubject("PO-20260603");
        mailMessage.setBodyText("Please create the order.");

        when(agentRunClient.run(any())).thenReturn(new AgentRunResult(true, "202", null, "{\"accepted\":true}", null));

        AgentRunResult result = service.dispatch(mailMessage);

        ArgumentCaptor<AgentInboundRequest> captor = ArgumentCaptor.forClass(AgentInboundRequest.class);
        ArgumentCaptor<MailMessage> updateCaptor = ArgumentCaptor.forClass(MailMessage.class);
        verify(agentRunClient).run(captor.capture());
        verify(mailMessageMapper).updateById(updateCaptor.capture());

        AgentInboundRequest request = captor.getValue();
        assertTrue(result.sent());
        assertNotNull(request);
        assertEquals("email", request.channel());
        assertEquals("0f13c809-86bb-410b-ac8e-946be7772ba6", request.channelInstallationId());
        assertEquals("tenant_2", request.tenantId());
        assertEquals("sender@example.com", request.sender().id());
        assertTrue(request.content().text().contains("Mail From:\nSender Name <sender@example.com>"));
        assertTrue(request.content().text().contains("Mail To:\nbuyer@example.com;ops@example.com"));
        assertTrue(request.content().text().contains("Mail Subject:\nPO-20260603"));
        assertTrue(request.content().text().contains("Mail Body:\nPlease create the order."));

        MailMessage updated = updateCaptor.getValue();
        assertEquals("DISPATCHED", updated.getWorkflowStatus());
        assertEquals("mail:_msg-1_example_com_", updated.getWorkflowThreadId());
        assertNull(updated.getWorkflowRunId());
        assertNotNull(updated.getWorkflowRequest());
        assertTrue(updated.getWorkflowRequest().contains("\"channel\":\"email\""));
        assertTrue(updated.getWorkflowRequest().contains("\"channelInstallationId\":\"0f13c809-86bb-410b-ac8e-946be7772ba6\""));
        assertTrue(updated.getWorkflowRequest().contains("\"tenantId\":\"tenant_2\""));
        assertTrue(updated.getWorkflowRequest().contains("\"sender\":{\"id\":\"sender@example.com\"}"));
        assertTrue(updated.getWorkflowRequest().contains("\"content\":{\"text\":\"Mail From:\\nSender Name <sender@example.com>"));
    }

    @Test
    void dispatchFallsBackToRawSenderWhenAddressParsingFails() {
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
        mailMessage.setId(7L);
        mailMessage.setFromAddress("Broken Sender <>");
        mailMessage.setSubject("PO-20260603");
        mailMessage.setBodyText("Please create the order.");

        when(agentRunClient.run(any())).thenReturn(new AgentRunResult(true, "202", null, "{\"accepted\":true}", null));

        service.dispatch(mailMessage);

        ArgumentCaptor<AgentInboundRequest> captor = ArgumentCaptor.forClass(AgentInboundRequest.class);
        verify(agentRunClient).run(captor.capture());

        AgentInboundRequest request = captor.getValue();
        assertEquals("Broken Sender <>", request.sender().id());
    }

    @Test
    void dispatchIncludesAttachmentMetadataAndTextEvenWhenAgentMessageExists() {
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
        mailMessage.setId(99L);
        mailMessage.setFromAddress("buyer@example.com");
        mailMessage.setSubject("purchase order");
        mailMessage.setBodyText("Please follow up.");
        mailMessage.setAgentMessage("Short generated agent message.");
        mailMessage.setAttachmentSummary("po.pdf (123 bytes) [DOCUMENT_SERVICE_EXTRACTED]");

        MailAttachment attachment = new MailAttachment();
        attachment.setId(1L);
        attachment.setMailMessageId(99L);
        attachment.setFileName("po.pdf");
        attachment.setContentType("application/pdf");
        attachment.setSizeBytes(123L);
        attachment.setUploadStatus("UPLOADED");
        attachment.setTosBucket("evario-demo");
        attachment.setTosKey("b1s/mail-attachments/99/po.pdf");
        attachment.setTosUrl("https://evario-demo.tos-s3-cn-shanghai.volces.com/b1s/mail-attachments/99/po.pdf");
        attachment.setExtractionStatus("DOCUMENT_SERVICE_EXTRACTED");
        attachment.setExtractedText("PO No. 123456789\nOffice Laptops - Model Z1");

        when(mailAttachmentMapper.selectList(any())).thenReturn(List.of(attachment));
        when(agentRunClient.run(any())).thenReturn(new AgentRunResult(true, "202", null, "{\"accepted\":true}", null));

        service.dispatch(mailMessage);

        ArgumentCaptor<AgentInboundRequest> captor = ArgumentCaptor.forClass(AgentInboundRequest.class);
        verify(agentRunClient).run(captor.capture());

        String text = captor.getValue().content().text();
        assertTrue(text.contains("Agent Message:\nShort generated agent message."));
        assertTrue(text.contains("Attachment Summary:\npo.pdf (123 bytes) [DOCUMENT_SERVICE_EXTRACTED]"));
        assertTrue(text.contains("TOS URL: https://evario-demo.tos-s3-cn-shanghai.volces.com/b1s/mail-attachments/99/po.pdf"));
        assertTrue(text.contains("Extraction Status: DOCUMENT_SERVICE_EXTRACTED"));
        assertTrue(text.contains("Extracted Text:\nPO No. 123456789"));
        assertTrue(text.contains("Office Laptops - Model Z1"));
    }

    private AgentRunProperties properties() {
        return new AgentRunProperties(
                true,
                "http://agent.example.com",
                "Bearer test-token",
                "tenant_2",
                "0f13c809-86bb-410b-ac8e-946be7772ba6",
                1000,
                5000
        );
    }

    private Executor sameThreadExecutor() {
        return Runnable::run;
    }
}
