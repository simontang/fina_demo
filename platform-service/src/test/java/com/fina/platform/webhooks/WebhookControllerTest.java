package com.fina.platform.webhooks;

import com.fina.platform.exception.GlobalExceptionHandler;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.util.List;
import java.util.Map;

import static org.hamcrest.Matchers.containsString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@ExtendWith(MockitoExtension.class)
class WebhookControllerTest {

    @Mock
    private WebhookFacade facade;

    private MockMvc mvc;

    @BeforeEach
    void setUp() {
        mvc = MockMvcBuilders
                .standaloneSetup(new WebhookController(facade))
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
    }

    @Test
    void publishForwardsSvixChannels() throws Exception {
        when(facade.publish(
                eq("job.completed"),
                eq(Map.of("jobId", "1")),
                eq(List.of("vip-customers", "ops"))))
                .thenReturn(Map.of(
                        "messageId", "msg_1",
                        "eventType", "job.completed",
                        "channels", List.of("vip-customers", "ops")));

        mvc.perform(post("/api/v1/webhooks/publish")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "eventType": "job.completed",
                                  "payload": {"jobId": "1"},
                                  "channels": ["vip-customers", "ops"]
                                }
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.messageId").value("msg_1"))
                .andExpect(jsonPath("$.channels[0]").value("vip-customers"))
                .andExpect(jsonPath("$.channels[1]").value("ops"));

        verify(facade).publish(
                eq("job.completed"),
                eq(Map.of("jobId", "1")),
                eq(List.of("vip-customers", "ops")));
    }

    @Test
    void publishRejectsEndpointIdsInsteadOfSilentlyBroadcasting() throws Exception {
        mvc.perform(post("/api/v1/webhooks/publish")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "eventType": "job.completed",
                                  "payload": {"jobId": "1"},
                                  "endpointIds": ["ep_1"]
                                }
                                """))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("BAD_REQUEST"))
                .andExpect(jsonPath("$.message", containsString("use channels")));

        verifyNoInteractions(facade);
    }

    @Test
    void publishRejectsConflictingLegacyTopicField() throws Exception {
        mvc.perform(post("/api/v1/webhooks/publish")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "eventType": "job.completed",
                                  "topic": "job.started",
                                  "payload": {"jobId": "1"}
                                }
                                """))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.message", containsString("eventType")));

        verifyNoInteractions(facade);
    }
}
