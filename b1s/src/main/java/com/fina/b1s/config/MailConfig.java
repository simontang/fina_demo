package com.fina.b1s.config;

import com.fina.b1s.mail.MailListenerProperties;
import com.fina.b1s.tos.TosProperties;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@EnableConfigurationProperties({MailListenerProperties.class, TosProperties.class})
public class MailConfig {
}
