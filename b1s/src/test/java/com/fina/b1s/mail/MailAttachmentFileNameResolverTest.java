package com.fina.b1s.mail;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MailAttachmentFileNameResolverTest {

    @Test
    void decodesMimeEncodedAttachmentFileName() {
        String encoded = "=?UTF-8?b?UE/CoDEwMTExNTA1OMKgLcKgU2VsYXRhbsKgQnVzaW5lc3MucGRm?=";

        String resolved = MailAttachmentFileNameResolver.resolve(
                encoded,
                "application/octet-stream; charset=\"utf-8\"; name=\"" + encoded + "\"",
                "%PDF-1.7".getBytes(StandardCharsets.ISO_8859_1)
        );

        assertEquals("PO 101115058 - Selatan Business.pdf", resolved);
    }

    @Test
    void fallsBackToContentTypeNameWhenPartFileNameIsMissing() {
        String resolved = MailAttachmentFileNameResolver.resolve(
                null,
                "application/pdf; name=\"purchase-order\"",
                "%PDF-1.7".getBytes(StandardCharsets.ISO_8859_1)
        );

        assertEquals("purchase-order.pdf", resolved);
    }

    @Test
    void fallsBackToPdfMagicWhenFileNameAndContentTypeAreGeneric() {
        String resolved = MailAttachmentFileNameResolver.resolve(
                "download",
                "application/octet-stream",
                "%PDF-1.7".getBytes(StandardCharsets.ISO_8859_1)
        );

        assertEquals("download.pdf", resolved);
    }
}
