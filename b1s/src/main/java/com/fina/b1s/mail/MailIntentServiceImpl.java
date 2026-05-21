package com.fina.b1s.mail;

import com.fina.b1s.entity.MailMessage;
import com.fina.b1s.llm.LlmIntentClassifier;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class MailIntentServiceImpl implements MailIntentService {

    private static final List<String> ORDER_KEYWORDS = List.of(
            "订购", "采购", "下单", "购买", "要订", "请发货", "按之前的价格", "按照之前的价格",
            "报价", "订单", "下单意向", "采购意向", "要货", "要买");

    private final LlmIntentClassifier llmIntentClassifier;

    @Override
    public boolean isOrderIntent(MailMessage mailMessage) {
        String text = normalize(mailMessage.getSubject()) + "\n" + normalize(mailMessage.getBodyText());
        if (!StringUtils.hasText(text)) {
            return false;
        }
        for (String keyword : ORDER_KEYWORDS) {
            if (text.contains(keyword)) {
                return true;
            }
        }
        boolean quantityIntent = text.matches("(?s).*(\\d+\\s*(件|个|套|台|只|箱)).*")
                && text.matches("(?s).*(要|需|请|买|订|采购|购买).*");
        if (quantityIntent) {
            return true;
        }

        LlmIntentClassifier.Classification classification =
                llmIntentClassifier.classify(mailMessage.getSubject(), mailMessage.getBodyText());
        if (classification.decisive()) {
            return classification.orderIntent();
        }
        return false;
    }

    private String normalize(String text) {
        return text == null ? "" : text.replaceAll("\\s+", " ").trim();
    }
}
