package com.fina.b1s.mail;

import com.fina.b1s.entity.MailMessage;

public interface MailIntentService {

    boolean isOrderIntent(MailMessage mailMessage);
}
