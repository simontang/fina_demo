package com.fina.b1s.mail;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.concurrent.Executor;

@Configuration
public class MailDispatchConfig {

    @Bean(name = "mailDispatchExecutor")
    public Executor mailDispatchExecutor(MailListenerProperties properties) {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setThreadNamePrefix("mail-dispatch-");
        executor.setCorePoolSize(Math.max(1, properties.dispatchThreadPoolSize()));
        executor.setMaxPoolSize(Math.max(1, properties.dispatchThreadPoolSize()));
        executor.setQueueCapacity(Math.max(1, properties.dispatchQueueCapacity()));
        executor.setWaitForTasksToCompleteOnShutdown(false);
        executor.initialize();
        return executor;
    }
}
