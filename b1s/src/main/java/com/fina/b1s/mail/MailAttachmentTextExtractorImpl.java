package com.fina.b1s.mail;

import com.fina.b1s.document.DocumentParseClient;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.nio.charset.StandardCharsets;
import java.util.Set;

@Slf4j
@Service
public class MailAttachmentTextExtractorImpl implements MailAttachmentTextExtractor {

    private static final int TEXT_LIMIT = 40000;
    private static final Set<String> DOCUMENT_SERVICE_EXTENSIONS = Set.of(
            ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".doc", ".docx"
    );

    private final DocumentParseClient documentParseClient;

    public MailAttachmentTextExtractorImpl(DocumentParseClient documentParseClient) {
        this.documentParseClient = documentParseClient;
    }

    @Override
    public ExtractionResult extract(String fileName, String contentType, byte[] bytes) {
        if (bytes == null || bytes.length == 0) {
            return new ExtractionResult(null, "EMPTY", null);
        }
        try {
            if (isText(contentType, fileName)) {
                return new ExtractionResult(trimPlainText(new String(bytes, StandardCharsets.UTF_8)), "EXTRACTED", null);
            }
            if (shouldUseDocumentService(fileName, contentType, bytes)) {
                DocumentParseClient.ParseResult result = documentParseClient.parse(fileName, contentType, bytes);
                String status = "DOCUMENT_SERVICE_" + firstNonBlank(result.status(), "FAILED");
                return new ExtractionResult(trimMarkdown(result.markdown()), status, result.errorMessage());
            }
            return new ExtractionResult(null, "UNSUPPORTED", null);
        } catch (Exception e) {
            log.warn("Attachment text extraction failed file={}: {}", fileName, e.getMessage());
            return new ExtractionResult(null, "FAILED", e.getMessage());
        }
    }

    private boolean shouldUseDocumentService(String fileName, String contentType, byte[] bytes) {
        if (StringUtils.hasText(contentType) && contentType.toLowerCase().contains("pdf")) {
            return true;
        }
        if (StringUtils.hasText(contentType) && contentType.toLowerCase().startsWith("image/")) {
            return true;
        }
        if (StringUtils.hasText(fileName) && fileName.toLowerCase().endsWith(".pdf")) {
            return true;
        }
        if (StringUtils.hasText(fileName)) {
            String lower = fileName.toLowerCase();
            for (String extension : DOCUMENT_SERVICE_EXTENSIONS) {
                if (lower.endsWith(extension)) {
                    return true;
                }
            }
        }
        return bytes.length >= 4
                && bytes[0] == '%'
                && bytes[1] == 'P'
                && bytes[2] == 'D'
                && bytes[3] == 'F';
    }

    private boolean isText(String contentType, String fileName) {
        if (StringUtils.hasText(contentType) && contentType.toLowerCase().startsWith("text/")) {
            return true;
        }
        return StringUtils.hasText(fileName)
                && (fileName.toLowerCase().endsWith(".txt") || fileName.toLowerCase().endsWith(".csv"));
    }

    private String trimPlainText(String value) {
        if (!StringUtils.hasText(value)) {
            return null;
        }
        String normalized = value.replace('\u00a0', ' ')
                .replaceAll("[ \\t\\x0B\\f\\r]+", " ")
                .replaceAll("\\n{3,}", "\n\n")
                .trim();
        if (normalized.length() <= TEXT_LIMIT) {
            return normalized;
        }
        return normalized.substring(0, TEXT_LIMIT);
    }

    private String trimMarkdown(String value) {
        if (!StringUtils.hasText(value)) {
            return null;
        }
        String normalized = value.replace('\u00a0', ' ')
                .replace("\r\n", "\n")
                .replace('\r', '\n')
                .trim();
        if (normalized.length() <= TEXT_LIMIT) {
            return normalized;
        }
        return normalized.substring(0, TEXT_LIMIT);
    }

    private String firstNonBlank(String first, String fallback) {
        return StringUtils.hasText(first) ? first : fallback;
    }
}
