package com.fina.b1s.mail;

import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;

@Slf4j
@Service
public class MailAttachmentTextExtractorImpl implements MailAttachmentTextExtractor {

    private static final int TEXT_LIMIT = 40000;

    @Override
    public ExtractionResult extract(String fileName, String contentType, byte[] bytes) {
        if (bytes == null || bytes.length == 0) {
            return new ExtractionResult(null, "EMPTY", null);
        }
        try {
            if (isPdf(fileName, contentType, bytes)) {
                return new ExtractionResult(trim(extractPdf(bytes)), "EXTRACTED", null);
            }
            if (isText(contentType, fileName)) {
                return new ExtractionResult(trim(new String(bytes, StandardCharsets.UTF_8)), "EXTRACTED", null);
            }
            return new ExtractionResult(null, "UNSUPPORTED", null);
        } catch (Exception e) {
            log.warn("Attachment text extraction failed file={}: {}", fileName, e.getMessage());
            return new ExtractionResult(null, "FAILED", e.getMessage());
        }
    }

    private String extractPdf(byte[] bytes) throws Exception {
        try (PDDocument document = PDDocument.load(new ByteArrayInputStream(bytes))) {
            PDFTextStripper stripper = new PDFTextStripper();
            return stripper.getText(document);
        }
    }

    private boolean isPdf(String fileName, String contentType, byte[] bytes) {
        if (StringUtils.hasText(contentType) && contentType.toLowerCase().contains("pdf")) {
            return true;
        }
        if (StringUtils.hasText(fileName) && fileName.toLowerCase().endsWith(".pdf")) {
            return true;
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

    private String trim(String value) {
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
}
