import {
  AgentThreadProvider,
  ChatUIContextProvider,
  Chating,
  ColumnLayout,
  SideAppViewBrowser,
  useAgentChat,
  AxiomLatticeProvider,
  LatticeChatShellContextProvider,
  ConversationContextProvider,
  AssistantContextProvider,
  useAssistantContext,
  useConversationContext,
  LatticeChat,
} from "@axiom-lattice/react-sdk";
import { forwardRef, useEffect, useImperativeHandle } from "react";

import { TOKEN_KEY } from "../../authProvider";
import { getBaseAPIPath } from "../../getBaseAPIPath";

function normalizeApiBaseUrl(raw?: string): string {
  const v = String(raw || "").trim().replace(/\/$/, "");
  if (!v) return "http://localhost:5702/api";
  if (v.endsWith("/api")) return v;
  return `${v}/api`;
}

const apiUrl = getBaseAPIPath();

export interface ChatBotRef {
  sendMessage: (message: string) => void;
}

export const ChatBotCompnent = forwardRef<
  ChatBotRef,
  {
    id: string;
    threadId: string;
    name: string;
  }
>(({ id, threadId, name }, ref) => {

  
  return (
    <LatticeChatShellContextProvider
      initialConfig={{
        baseURL: getBaseAPIPath(),
        apiKey: localStorage.getItem(TOKEN_KEY) || "",
        transport: "sse",
        enableThreadCreation: true,
        enableThreadList: true,
      }}
    > <AssistantContextProvider autoLoad={true} initialAssistantId={id}>
        <ConversationContextProvider>
          <AxiomLatticeProvider
            config={{
              baseURL: apiUrl,
              apiKey: localStorage.getItem(TOKEN_KEY) || "",
              assistantId: id || "",
              transport: "sse",

            }}
          >
            <ChatContent name={name} threadId={threadId} ref={ref} />
          </AxiomLatticeProvider>

        </ConversationContextProvider>
      </AssistantContextProvider>
    </LatticeChatShellContextProvider>
  );
});

const ChatContent = forwardRef<ChatBotRef, { name: string, threadId: string }>(({ name, threadId }, ref) => {
  const { assistantId, thread, createThread } = useConversationContext();
  const { currentAssistant } = useAssistantContext();
  const { sendMessage } = useAgentChat();
  useImperativeHandle(ref, () => ({
    sendMessage: (message: string) => {
      sendMessage({ input: { message } });
    },
  }));

  useEffect(() => {
    if (!thread) {
      createThread(threadId);
    }
  }, [thread]);

  return thread ? (

    <AgentThreadProvider
      assistantId={assistantId || ""}
      threadId={thread?.id || threadId}
      options={{
        streaming: true,
        enableReturnStateWhenStreamCompleted: true,
        enableResumeStream: true,
      }}
    >
      <LatticeChat
        thread_id={thread?.id}
        assistant_id={assistantId || ""}
        name={currentAssistant?.name}
        description={currentAssistant?.description}
      //header={<LatticeShellHeader />}
      />
    </AgentThreadProvider>
  ) : null;
});

