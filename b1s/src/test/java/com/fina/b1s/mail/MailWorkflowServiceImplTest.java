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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class MailWorkflowServiceImplTest {

    @Mock
    private AgentRunClient agentRunClient;
    @Mock
    private MailMessageMapper mailMessageMapper;
    @Mock
    private MailAttachmentMapper mailAttachmentMapper;

    @Test
    void dispatchSendsInboundPayloadWithParsedSenderAndSerializedWorkflowRequest() {
        MailWorkflowServiceImpl service = new MailWorkflowServiceImpl(
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
        assertTrue(request.content().text().contains("Mail Subject:\nPO-20260603"));
        assertFalse(request.content().text().contains("Mail Body:\nPlease create the order."));

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
        assertFalse(updated.getWorkflowRequest().contains("Please create the order."));
    }

    @Test
    void dispatchFallsBackToRawSenderWhenAddressParsingFails() {
        MailWorkflowServiceImpl service = new MailWorkflowServiceImpl(
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
    void dispatchIncludesAttachmentMarkdownWithoutDerivedSummaryOrStorageMetadata() {
        MailWorkflowServiceImpl service = new MailWorkflowServiceImpl(
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
        mailMessage.setPurchaseOrderSummary("Generated PO summary.");
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
        assertFalse(text.contains("Agent Message:"));
        assertFalse(text.contains("Short generated agent message."));
        assertFalse(text.contains("Purchase Order Summary:"));
        assertFalse(text.contains("Generated PO summary."));
        assertFalse(text.contains("Attachment Summary:"));
        assertFalse(text.contains("po.pdf (123 bytes) [DOCUMENT_SERVICE_EXTRACTED]"));
        assertFalse(text.contains("Attachment:\npo.pdf"));
        assertFalse(text.contains("TOS URL:"));
        assertFalse(text.contains("TOS Bucket:"));
        assertFalse(text.contains("TOS Key:"));
        assertFalse(text.contains("Upload Status:"));
        assertFalse(text.contains("Extraction Status:"));
        assertTrue(text.contains("Attachment Markdown:\nAttachment: po.pdf"));
        assertTrue(text.contains("PO No. 123456789"));
        assertTrue(text.contains("Office Laptops - Model Z1"));
    }

    @Test
    void asyncDispatchQueuesAndSendsWithoutIntentGate() {
        MailWorkflowServiceImpl service = new MailWorkflowServiceImpl(
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

        MailMessage fresh = new MailMessage();
        fresh.setId(42L);
        fresh.setMailbox("sales@alphafina.cn");
        fresh.setUid(1001L);
        fresh.setFromAddress("Sender Name <sender@example.com>");
        fresh.setSubject("Weekly FYI");

        when(mailMessageMapper.selectById(42L)).thenReturn(fresh);
        when(agentRunClient.run(any())).thenReturn(new AgentRunResult(true, "202", null, "{\"accepted\":true}", null));

        service.dispatchAsyncIfOrderIntent(mailMessage);

        ArgumentCaptor<AgentInboundRequest> requestCaptor = ArgumentCaptor.forClass(AgentInboundRequest.class);
        ArgumentCaptor<MailMessage> updateCaptor = ArgumentCaptor.forClass(MailMessage.class);
        verify(agentRunClient).run(requestCaptor.capture());
        verify(mailMessageMapper, times(2)).updateById(updateCaptor.capture());

        List<MailMessage> updates = updateCaptor.getAllValues();
        assertEquals("QUEUED", updates.get(0).getWorkflowStatus());
        assertEquals("DISPATCHED", updates.get(1).getWorkflowStatus());
        assertEquals("Weekly FYI", fresh.getSubject());
        assertTrue(requestCaptor.getValue().content().text().contains("Mail Subject:\nWeekly FYI"));
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
