package com.fina.b1s.mail;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
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
import jakarta.mail.internet.MimeMessage;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Date;
import java.util.Base64;
import java.util.List;
import java.util.Properties;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class MailIngestServiceImpl implements MailIngestService {

    private static final String PYTHON_IMAP_FETCH_SCRIPT = """
import base64
import imaplib
import json
import sys

host = sys.argv[1]
port = int(sys.argv[2])
username = sys.argv[3]
password = sys.argv[4]
folder = sys.argv[5]
min_uid = int(sys.argv[6])
mark_seen = sys.argv[7] == "1"

mail = imaplib.IMAP4_SSL(host, port)
mail.login(username, password)
mail.select(folder, readonly=not mark_seen)
_, search_data = mail.uid("search", None, "ALL")
uids = []
if search_data and search_data[0]:
    for token in search_data[0].split():
        try:
            uid = int(token)
        except ValueError:
            continue
        if uid > min_uid:
            uids.append(uid)

out = []
for uid in uids:
    _, fetch_data = mail.uid("fetch", str(uid), "(RFC822)")
    raw = b""
    if fetch_data:
        for part in fetch_data:
            if isinstance(part, tuple) and len(part) > 1 and part[1]:
                raw = part[1]
                break
    if not raw:
        continue
    out.append({
        "uid": uid,
        "raw": base64.b64encode(raw).decode("ascii"),
    })
    if mark_seen:
        try:
            mail.uid("store", str(uid), "+FLAGS", r"(\\\\Seen)")
        except Exception:
            pass

print(json.dumps(out))
mail.logout()
""";

    private static final ZoneId ZONE_ID = ZoneId.of("Asia/Shanghai");
    private static final int SNIPPET_LIMIT = 1000;

    private final MailListenerProperties properties;
    private final MailMessageMapper messageMapper;
    private final MailAttachmentMapper attachmentMapper;
    private final TosStorageService tosStorageService;
    private final com.fina.b1s.tos.TosProperties tosProperties;
    private final MailAttachmentTextExtractor attachmentTextExtractor;
    private final PurchaseOrderSummaryService purchaseOrderSummaryService;
    private final MailWorkflowService mailWorkflowService;
    private final ObjectMapper objectMapper;

    @Override
    public int pollInbox() {
        if (!properties.enabled()) {
            return 0;
        }
        validateConfig();
        try {
            return pollInboxViaPython();
        } catch (Exception pythonError) {
            log.warn("Python IMAP polling failed: {}", pythonError.getMessage(), pythonError);
            try {
                return pollInboxViaJavaMail();
            } catch (Exception javaMailError) {
                log.warn("Mail polling failed: {}", javaMailError.getMessage(), javaMailError);
                return 0;
            }
        }
    }

    private int pollInboxViaPython() throws Exception {
        long maxUid = findMaxUid();
        List<PythonEnvelope> envelopes = fetchPythonEnvelopes(maxUid);
        if (envelopes.isEmpty()) {
            log.info("Mail polling completed mailbox={} folder={} processed={}",
                    properties.username(), properties.folder(), 0);
            return 0;
        }
        int processed = 0;
        Session messageSession = Session.getInstance(new Properties());
        for (PythonEnvelope envelope : envelopes) {
            if (envelope.uid() <= maxUid) {
                continue;
            }
            MimeMessage message = new MimeMessage(messageSession, new ByteArrayInputStream(envelope.rawBytes()));
            if (exists(envelope.uid(), message)) {
                continue;
            }
            MailMessage saved = saveMessage(properties.folder(), envelope.uid(), message);
            mailWorkflowService.dispatchAsyncIfOrderIntent(saved);
            processed++;
        }
        log.info("Mail polling completed mailbox={} folder={} processed={}",
                properties.username(), properties.folder(), processed);
        return processed;
    }

    private List<PythonEnvelope> fetchPythonEnvelopes(long minUid) throws IOException, InterruptedException {
        ProcessBuilder builder = new ProcessBuilder(
                "python3",
                "-c",
                PYTHON_IMAP_FETCH_SCRIPT,
                properties.imapHost(),
                String.valueOf(properties.imapPort()),
                properties.username(),
                properties.password(),
                properties.folder(),
                String.valueOf(minUid),
                properties.markSeen() ? "1" : "0"
        );
        builder.redirectErrorStream(true);
        Process process = builder.start();
        String output;
        try (InputStream in = process.getInputStream()) {
            output = new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new IOException("Python IMAP poll exited with code " + exitCode + ": " + trim(output, 2000));
        }
        if (!StringUtils.hasText(output)) {
            return List.of();
        }
        List<PythonEnvelope> envelopes = objectMapper.readValue(output, new TypeReference<List<PythonEnvelope>>() {});
        List<PythonEnvelope> out = new ArrayList<>(envelopes.size());
        for (PythonEnvelope envelope : envelopes) {
            if (envelope == null || envelope.raw() == null) {
                continue;
            }
            out.add(envelope);
        }
        return out;
    }

    private MailMessage saveMessage(String folderName, long uid, Message message) throws Exception {
        MailMessage entity = new MailMessage();
        entity.setProvider("larksuite");
        entity.setMailbox(properties.username());
        entity.setFolderName(folderName);
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

        List<AttachmentPayload> attachments = collectAttachments(message);
        entity.setAttachmentCount(attachments.size());
        entity.setHasAttachments(!attachments.isEmpty());
        entity.setCreatedAt(LocalDateTime.now());
        entity.setUpdatedAt(LocalDateTime.now());
        messageMapper.insert(entity);

        List<MailAttachment> savedAttachments = new ArrayList<>();
        for (AttachmentPayload attachment : attachments) {
            savedAttachments.add(saveAttachment(entity.getId(), attachment));
        }
        enrichMessage(entity, savedAttachments);
        return entity;
    }

    private int pollInboxViaJavaMail() {
        Properties props = new Properties();
        props.put("mail.store.protocol", "imaps");
        props.put("mail.imaps.ssl.enable", "true");
        props.put("mail.imaps.connectionpoolsize", "0");
        props.put("mail.imap.connectionpoolsize", "0");
        props.put("mail.imaps.connectionpooltimeout", "1000");
        props.put("mail.imap.connectionpooltimeout", "1000");
        props.put("mail.imaps.statuscachetimeout", "0");
        props.put("mail.imap.statuscachetimeout", "0");
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
            throw new RuntimeException(e);
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
            mailWorkflowService.dispatchAsyncIfOrderIntent(saved);
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
        int windowSize = Math.max(batchSize * 5, 50);
        log.info("Mail poll inspect mailbox={} folder={} folderClass={} uidFolder={} batchSize={} windowSize={} messageCount={}",
                properties.username(),
                properties.folder(),
                folder.getClass().getName(),
                folder instanceof UIDFolder,
                batchSize,
                windowSize,
                folder.getMessageCount());
        if (folder instanceof UIDFolder uidFolder) {
            long maxUid = findMaxUid();
            Message[] uidMessages = compactMessages(uidFolder.getMessagesByUID(1L, UIDFolder.LASTUID));
            if (uidMessages.length > 0) {
                Message[] trimmed = trimToLast(uidMessages, batchSize);
                log.info("Mail poll selected {} message(s) by uid mailbox={} folder={} maxUid={} firstSubject={}",
                        trimmed.length,
                        properties.username(),
                        properties.folder(),
                        maxUid,
                        trimmed[0] != null ? trimmed[0].getSubject() : null);
                return trimmed;
            }
        }

        Message[] recent = recentMessages(folder, windowSize);
        if (recent.length == 0) {
            log.info("Mail poll selected no messages mailbox={} folder={}", properties.username(), properties.folder());
            return new Message[0];
        }
        log.info("Mail poll selected {} message(s) by recent window mailbox={} folder={} firstSubject={}",
                recent.length,
                properties.username(),
                properties.folder(),
                recent[0] != null ? recent[0].getSubject() : null);
        return recent;
    }

    private MailMessage saveMessage(Folder folder, long uid, Message message) throws Exception {
        return saveMessage(folder.getFullName(), uid, message);
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

    private static Message[] compactMessages(Message[] messages) {
        if (messages == null || messages.length == 0) {
            return new Message[0];
        }
        List<Message> out = new ArrayList<>(messages.length);
        for (Message message : messages) {
            if (message != null) {
                out.add(message);
            }
        }
        return out.toArray(Message[]::new);
    }

    private Message[] recentMessages(Folder folder, int windowSize) throws MessagingException {
        int messageCount = folder.getMessageCount();
        if (messageCount <= 0) {
            return new Message[0];
        }
        int safeWindow = Math.max(1, windowSize);
        int start = Math.max(1, messageCount - safeWindow + 1);
        return compactMessages(folder.getMessages(start, messageCount));
    }

    private MailAttachment saveAttachment(Long mailMessageId, AttachmentPayload payload) {
        MailAttachment attachment = new MailAttachment();
        attachment.setMailMessageId(mailMessageId);
        attachment.setFileName(payload.fileName());
        attachment.setContentType(payload.contentType());
        attachment.setSizeBytes((long) payload.bytes().length);
        attachment.setCreatedAt(LocalDateTime.now());

        MailAttachmentTextExtractor.ExtractionResult extraction =
                attachmentTextExtractor.extract(payload.fileName(), payload.contentType(), payload.bytes());
        attachment.setExtractedText(trim(extraction.text(), 40000));
        attachment.setExtractionStatus(extraction.status());
        attachment.setExtractionError(trim(extraction.errorMessage(), 1000));

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
        return attachment;
    }

    private void enrichMessage(MailMessage entity, List<MailAttachment> attachments) {
        String attachmentSummary = summarizeAttachments(attachments);
        String attachmentText = joinAttachmentText(attachments);
        PurchaseOrderSummaryService.PurchaseOrderSummary poSummary =
                purchaseOrderSummaryService.summarize(entity.getSubject(), entity.getBodyText(), attachmentText);

        entity.setAttachmentSummary(attachmentSummary);
        entity.setAttachmentText(attachmentText);
        entity.setPurchaseOrderSummary(poSummary.summaryText());
        entity.setAgentMessage(poSummary.agentMessage());
        entity.setSnippet(trim(firstNonBlank(poSummary.summaryText(), entity.getBodyText(), attachmentText), SNIPPET_LIMIT));
        entity.setOrderIntent(true);
        entity.setWorkflowStatus("PENDING");
        entity.setUpdatedAt(LocalDateTime.now());
        messageMapper.updateById(entity);
    }

    private String summarizeAttachments(List<MailAttachment> attachments) {
        if (attachments == null || attachments.isEmpty()) {
            return null;
        }
        List<String> parts = new ArrayList<>();
        for (MailAttachment attachment : attachments) {
            StringBuilder line = new StringBuilder();
            line.append(attachment.getFileName());
            if (attachment.getSizeBytes() != null) {
                line.append(" (").append(attachment.getSizeBytes()).append(" bytes)");
            }
            if (StringUtils.hasText(attachment.getExtractionStatus())) {
                line.append(" [").append(attachment.getExtractionStatus()).append("]");
            }
            parts.add(line.toString());
        }
        return trim(String.join("\n", parts), 4000);
    }

    private String joinAttachmentText(List<MailAttachment> attachments) {
        if (attachments == null || attachments.isEmpty()) {
            return null;
        }
        List<String> chunks = new ArrayList<>();
        for (MailAttachment attachment : attachments) {
            if (!StringUtils.hasText(attachment.getExtractedText())) {
                continue;
            }
            chunks.add("Attachment: " + firstNonBlank(attachment.getFileName(), "unnamed"));
            chunks.add(attachment.getExtractedText());
        }
        if (chunks.isEmpty()) {
            return null;
        }
        return trim(String.join("\n\n", chunks), 40000);
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
            byte[] bytes = readAllBytes(part.getInputStream());
            String contentType = part.getContentType();
            String resolvedFileName = MailAttachmentFileNameResolver.resolve(fileName, contentType, bytes);
            out.add(new AttachmentPayload(
                    StringUtils.hasText(resolvedFileName) ? resolvedFileName : "attachment-" + UUID.randomUUID(),
                    contentType,
                    bytes));
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

    private String firstNonBlank(String... values) {
        for (String value : values) {
            if (StringUtils.hasText(value)) {
                return value;
            }
        }
        return null;
    }

    private void validateConfig() {
        if (!StringUtils.hasText(properties.username()) || !StringUtils.hasText(properties.password())) {
            throw new IllegalStateException("Mail listener username/password are required");
        }
    }

    private record AttachmentPayload(String fileName, String contentType, byte[] bytes) {
    }

    private record PythonEnvelope(long uid, String raw) {
        private byte[] rawBytes() {
            return raw == null ? new byte[0] : Base64.getDecoder().decode(raw);
        }
    }
}
