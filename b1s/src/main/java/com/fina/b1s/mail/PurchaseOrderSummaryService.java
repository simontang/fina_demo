package com.fina.b1s.mail;

public interface PurchaseOrderSummaryService {

    PurchaseOrderSummary summarize(String subject, String bodyText, String attachmentText);

    record PurchaseOrderSummary(
            String summaryText,
            String agentMessage
    ) {
    }
}
