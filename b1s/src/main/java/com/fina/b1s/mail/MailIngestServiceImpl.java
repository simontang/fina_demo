package com.fina.b1s.mail;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fina.b1s.entity.MailAttachment;
import com.fina.b1s.entity.MailMessage;
import com.fina.b1s.mapper.MailAttachmentMapper;
import com.fina.b1s.mapper.MailMessageMapper;
import com.fina.b1s.tos.TosStorageService;
import jakarta.mail.Address;
import jakarta.mail.BodyPart;
import jakarta.mail.Flags;
import jakarta.mail.Folder;
import jakarta.mail.Message;
import jakarta.mail.MessagingException;
import jakarta.mail.Multipart;
import jakarta.mail.Part;
import jakarta.mail.Session;
import jakarta.mail.Store;
import jakarta.mail.UIDFolder;
import jakarta.mail.search.FlagTerm;
import jakarta.mail.search.SearchTerm;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Properties;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class MailIngestServiceImpl implements MailIngestService {

    private static final ZoneId ZONE_ID = ZoneId.of("Asia/Shanghai");
    private static final int SNIPPET_LIMIT = 1000;

    private final MailListenerProperties properties;
    private final MailMessageMapper messageMapper;
    private final MailAttachmentMapper attachmentMapper;
    private final TosStorageService tosStorageService;
    private final com.fina.b1s.tos.TosProperties tosProperties;
    private final MailIntentService mailIntentService;
    private final MailWorkflowService mailWorkflowService;

    @Override
    public int pollInbox() {
        if (!properties.enabled()) {
            return 0;
        }
        validateConfig();

        Properties props = new Properties();
        props.put("mail.store.protocol", "imaps");
        props.put("mail.imaps.ssl.enable", "true");
        props.put("mail.imaps.host", properties.imapHost());
        props.put("mail.imaps.port", String.valueOf(properties.imapPort()));

        Session session = Session.getInstance(props);
        try (Store store = session.getStore("imaps")) {
            store.connect(properties.imapHost(), properties.imapPort(), properties.username(), properties.password());
            Folder folder = store.getFolder(properties.folder());
            folder.open(Folder.READ_WRITE);
            try {
                return ingestFolder(folder);
            } finally {
                folder.close(false);
            }
        } catch (Exception e) {
            log.warn("Mail polling failed: {}", e.getMessage(), e);
            return 0;
        }
    }

    private int ingestFolder(Folder folder) throws Exception {
        Message[] messages = selectMessages(folder);
        if (messages == null || messages.length == 0) {
            log.info("Mail polling completed mailbox={} folder={} processed={}",
                    properties.username(), properties.folder(), 0);
            return 0;
        }
        int processed = 0;
        for (Message message : messages) {
            long uid = resolveUid(folder, message);
            if (exists(uid, message)) {
                continue;
            }
            MailMessage saved = saveMessage(folder, uid, message);
            mailWorkflowService.dispatchIfOrderIntent(saved);
            if (properties.markSeen()) {
                message.setFlag(Flags.Flag.SEEN, true);
            }
            processed++;
        }
        log.info("Mail polling completed mailbox={} folder={} processed={}",
                properties.username(), properties.folder(), processed);
        return processed;
    }

    private Message[] selectMessages(Folder folder) throws Exception {
        int batchSize = Math.max(properties.batchSize(), 1);
        if (folder instanceof UIDFolder uidFolder) {
            long maxUid = findMaxUid();
            long uidStart = Math.max(1L, maxUid + 1L);
            long uidEnd = UIDFolder.LASTUID;
            Message[] messages = uidFolder.getMessagesByUID(uidStart, uidEnd);
            if (messages == null || messages.length == 0) {
                return new Message[0];
            }
            return trimToLast(messages, batchSize);
        }

        SearchTerm unseenOnly = new FlagTerm(new Flags(Flags.Flag.SEEN), false);
        Message[] messages = folder.search(unseenOnly);
        if (messages == null || messages.length == 0) {
            return new Message[0];
        }
        return trimToLast(messages, batchSize);
    }

    private MailMessage saveMessage(Folder folder, long uid, Message message) throws Exception {
        MailMessage entity = new MailMessage();
        entity.setProvider("larksuite");
        entity.setMailbox(properties.username());
        entity.setFolderName(folder.getFullName());
        entity.setUid(uid);
        entity.setMessageId(firstHeader(message, "Message-ID"));
        entity.setSubject(message.getSubject());
        entity.setFromAddress(addressesToString(message.getFrom()));
        entity.setToAddresses(addressesToString(message.getRecipients(Message.RecipientType.TO)));
        entity.setCcAddresses(addressesToString(message.getRecipients(Message.RecipientType.CC)));
        entity.setSentAt(formatDate(message.getSentDate()));
        entity.setReceivedAt(formatDate(message.getReceivedDate()));
        String bodyText = extractBodyText(message);
        entity.setBodyText(bodyText);
        entity.setSnippet(trim(bodyText, SNIPPET_LIMIT));
        entity.setOrderIntent(mailIntentService.isOrderIntent(entity));
        entity.setWorkflowStatus(entity.getOrderIntent() ? "PENDING" : "NOT_ORDER_INTENT");

        List<AttachmentPayload> attachments = collectAttachments(message);
        entity.setAttachmentCount(attachments.size());
        entity.setHasAttachments(!attachments.isEmpty());
        entity.setCreatedAt(LocalDateTime.now());
        entity.setUpdatedAt(LocalDateTime.now());
        messageMapper.insert(entity);

        for (AttachmentPayload attachment : attachments) {
            saveAttachment(entity.getId(), attachment);
        }
        return entity;
    }

    private boolean exists(long uid, Message message) throws MessagingException {
        String messageId = firstHeader(message, "Message-ID");
        LambdaQueryWrapper<MailMessage> wrapper = new LambdaQueryWrapper<MailMessage>()
                .eq(MailMessage::getMailbox, properties.username());
        if (uid > 0) {
            wrapper.eq(MailMessage::getUid, uid);
        } else if (StringUtils.hasText(messageId)) {
            wrapper.eq(MailMessage::getMessageId, messageId);
        } else {
            return false;
        }
        return messageMapper.selectCount(wrapper) > 0;
    }

    private long findMaxUid() {
        QueryWrapper<MailMessage> wrapper = new QueryWrapper<>();
        wrapper.select("MAX(uid) AS uid")
                .eq("mailbox", properties.username())
                .eq("folder_name", properties.folder())
                .isNotNull("uid");
        MailMessage row = messageMapper.selectOne(wrapper);
        return row != null && row.getUid() != null ? row.getUid() : 0L;
    }

    private long resolveUid(Folder folder, Message message) throws MessagingException {
        if (folder instanceof UIDFolder uidFolder) {
            return uidFolder.getUID(message);
        }
        return -1;
    }

    private static Message[] trimToLast(Message[] messages, int batchSize) {
        if (messages.length <= batchSize) {
            return messages;
        }
        Message[] trimmed = new Message[batchSize];
        System.arraycopy(messages, messages.length - batchSize, trimmed, 0, batchSize);
        return trimmed;
    }

    private void saveAttachment(Long mailMessageId, AttachmentPayload payload) {
        MailAttachment attachment = new MailAttachment();
        attachment.setMailMessageId(mailMessageId);
        attachment.setFileName(payload.fileName());
        attachment.setContentType(payload.contentType());
        attachment.setSizeBytes((long) payload.bytes().length);
        attachment.setCreatedAt(LocalDateTime.now());

        String key = buildTosKey(mailMessageId, payload.fileName());
        try (InputStream inputStream = new ByteArrayInputStream(payload.bytes())) {
            TosStorageService.UploadResult result =
                    tosStorageService.upload(key, inputStream, payload.bytes().length);
            attachment.setTosBucket(result.bucket());
            attachment.setTosKey(result.key());
            attachment.setTosUrl(result.url());
            attachment.setUploadStatus("UPLOADED");
        } catch (Exception e) {
            attachment.setTosKey(key);
            attachment.setUploadStatus("FAILED");
            attachment.setErrorMessage(trim(e.getMessage(), 1000));
            log.warn("Attachment upload failed mailMessageId={} file={}: {}",
                    mailMessageId, payload.fileName(), e.getMessage());
        }
        attachmentMapper.insert(attachment);
    }

    private List<AttachmentPayload> collectAttachments(Part part) throws Exception {
        List<AttachmentPayload> out = new ArrayList<>();
        collectAttachments(part, out);
        return out;
    }

    private void collectAttachments(Part part, List<AttachmentPayload> out) throws Exception {
        if (part.isMimeType("multipart/*")) {
            Multipart multipart = (Multipart) part.getContent();
            for (int i = 0; i < multipart.getCount(); i++) {
                BodyPart bodyPart = multipart.getBodyPart(i);
                collectAttachments(bodyPart, out);
            }
            return;
        }

        String disposition = part.getDisposition();
        String fileName = part.getFileName();
        if (Part.ATTACHMENT.equalsIgnoreCase(disposition)
                || Part.INLINE.equalsIgnoreCase(disposition) && StringUtils.hasText(fileName)) {
            out.add(new AttachmentPayload(
                    StringUtils.hasText(fileName) ? fileName : "attachment-" + UUID.randomUUID(),
                    part.getContentType(),
                    readAllBytes(part.getInputStream())));
        }
    }

    private String extractBodyText(Part part) throws Exception {
        if (part.isMimeType("text/plain")) {
            Object content = part.getContent();
            return normalizeBody(String.valueOf(content));
        }
        if (part.isMimeType("text/html")) {
            Object content = part.getContent();
            String text = String.valueOf(content).replaceAll("<[^>]+>", " ");
            return normalizeBody(text);
        }
        if (part.isMimeType("multipart/*")) {
            Multipart multipart = (Multipart) part.getContent();
            String firstHtml = null;
            for (int i = 0; i < multipart.getCount(); i++) {
                BodyPart bodyPart = multipart.getBodyPart(i);
                String body = extractBodyText(bodyPart);
                if (StringUtils.hasText(body) && bodyPart.isMimeType("text/plain")) {
                    return body;
                }
                if (firstHtml == null && StringUtils.hasText(body)) {
                    firstHtml = body;
                }
            }
            return firstHtml;
        }
        return null;
    }

    private String normalizeBody(String value) {
        if (value == null) {
            return null;
        }
        String normalized = value.replace('\u00a0', ' ').replaceAll("[ \\t\\x0B\\f\\r]+", " ").trim();
        return StringUtils.hasText(normalized) ? normalized : null;
    }

    private String buildTosKey(Long mailMessageId, String fileName) {
        String prefix = StringUtils.hasText(tosProperties.keyPrefix())
                ? tosProperties.keyPrefix().replaceAll("/+$", "")
                : "b1s/mail-attachments";
        String safeName = fileName == null ? "attachment" : fileName.replaceAll("[\\\\/]+", "_");
        return prefix + "/" + mailMessageId + "/" + UUID.randomUUID() + "-" + safeName;
    }

    private byte[] readAllBytes(InputStream inputStream) throws Exception {
        try (InputStream in = inputStream; ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            in.transferTo(out);
            return out.toByteArray();
        }
    }

    private String addressesToString(Address[] addresses) {
        if (addresses == null || addresses.length == 0) {
            return null;
        }
        List<String> values = new ArrayList<>();
        for (Address address : addresses) {
            values.add(address.toString());
        }
        return String.join(", ", values);
    }

    private String firstHeader(Part part, String name) throws MessagingException {
        String[] values = part.getHeader(name);
        if (values == null || values.length == 0) {
            return null;
        }
        return values[0];
    }

    private String formatDate(Date date) {
        if (date == null) {
            return null;
        }
        return LocalDateTime.ofInstant(date.toInstant(), ZONE_ID).toString();
    }

    private String trim(String value, int limit) {
        if (value == null) {
            return null;
        }
        String normalized = value.strip();
        if (normalized.length() <= limit) {
            return normalized;
        }
        return normalized.substring(0, limit);
    }

    private void validateConfig() {
        if (!StringUtils.hasText(properties.username()) || !StringUtils.hasText(properties.password())) {
            throw new IllegalStateException("Mail listener username/password are required");
        }
    }

    private record AttachmentPayload(String fileName, String contentType, byte[] bytes) {
    }
}
