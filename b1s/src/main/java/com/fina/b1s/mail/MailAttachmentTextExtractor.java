package com.fina.b1s.mail;

public interface MailAttachmentTextExtractor {

    ExtractionResult extract(String fileName, String contentType, byte[] bytes);

    record ExtractionResult(
            String text,
            String status,
            String errorMessage
    ) {
    }
}
