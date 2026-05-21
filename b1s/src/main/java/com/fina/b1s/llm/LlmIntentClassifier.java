package com.fina.b1s.llm;

public interface LlmIntentClassifier {

    Classification classify(String subject, String bodyText);

    record Classification(boolean decisive, boolean orderIntent, String rawResponse, String errorMessage) {
    }
}
