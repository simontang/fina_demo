package com.fina.b1s.mail;

import com.fina.b1s.document.DocumentParseClient;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class MailAttachmentTextExtractorImplTest {

    @Test
    void pdfAttachmentsUseDocumentService() {
        DocumentParseClient documentParseClient = mock(DocumentParseClient.class);
        MailAttachmentTextExtractorImpl extractor = new MailAttachmentTextExtractorImpl(documentParseClient);
        byte[] bytes = "%PDF-1.7".getBytes();

        String markdown = "# Purchase Order\r\n\r\n| Item | Qty |\n| --- | --- |\n| Office  Laptops | 2 |";
        when(documentParseClient.parse("po.pdf", "application/pdf", bytes))
                .thenReturn(new DocumentParseClient.ParseResult(
                        markdown,
                        "EXTRACTED",
                        "mineru",
                        "asset_1",
                        "run_1",
                        null
                ));

        MailAttachmentTextExtractor.ExtractionResult result = extractor.extract("po.pdf", "application/pdf", bytes);

        assertEquals("DOCUMENT_SERVICE_EXTRACTED", result.status());
        assertEquals("# Purchase Order\n\n| Item | Qty |\n| --- | --- |\n| Office  Laptops | 2 |", result.text());
        assertNull(result.errorMessage());
        verify(documentParseClient).parse("po.pdf", "application/pdf", bytes);
    }

    @Test
    void textAttachmentsStayLocal() {
        DocumentParseClient documentParseClient = mock(DocumentParseClient.class);
        MailAttachmentTextExtractorImpl extractor = new MailAttachmentTextExtractorImpl(documentParseClient);

        MailAttachmentTextExtractor.ExtractionResult result =
                extractor.extract("note.txt", "text/plain", "hello\nworld".getBytes());

        assertEquals("EXTRACTED", result.status());
        assertEquals("hello\nworld", result.text());
        assertNull(result.errorMessage());
        verifyNoInteractions(documentParseClient);
    }
}
