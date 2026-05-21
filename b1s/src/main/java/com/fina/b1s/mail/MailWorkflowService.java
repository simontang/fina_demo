package com.fina.b1s.mail;

import com.fina.b1s.agent.AgentRunResult;
import com.fina.b1s.entity.MailMessage;

public interface MailWorkflowService {

    AgentRunResult dispatchIfOrderIntent(MailMessage mailMessage);

    AgentRunResult dispatch(MailMessage mailMessage);
}
