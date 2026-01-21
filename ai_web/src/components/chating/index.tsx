import {
  AgentThreadProvider,
  ChatUIContextProvider,
  Chating,
  ColumnLayout,
  SideAppViewBrowser,
  useAgentChat,
  AxiomLatticeProvider,
} from "@axiom-lattice/react-sdk";
import { forwardRef, useImperativeHandle } from "react";

import { TOKEN_KEY } from "../../authProvider";

function normalizeApiBaseUrl(raw?: string): string {
  const v = String(raw || "").trim().replace(/\/$/, "");
  if (!v) return "http://localhost:6203/api";
  if (v.endsWith("/api")) return v;
  return `${v}/api`;
}

const apiUrl = normalizeApiBaseUrl(import.meta.env.VITE_API_URL as string | undefined);

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
    <AxiomLatticeProvider
      config={{
        baseURL: apiUrl,
        apiKey: localStorage.getItem(TOKEN_KEY) || "",
        assistantId: id,
        transport: "sse",
      }}
    >
      <AgentThreadProvider
        assistantId={id}
        threadId={threadId}
        options={{
          streaming: true,
          enableReturnStateWhenStreamCompleted: true,
          enableResumeStream: true,
        }}
      >
        <ChatContent name={name} ref={ref} />
      </AgentThreadProvider>
    </AxiomLatticeProvider>
  );
});

const ChatContent = forwardRef<ChatBotRef, { name: string }>(({ name }, ref) => {
  const { sendMessage } = useAgentChat();
  useImperativeHandle(ref, () => ({
    sendMessage: (message: string) => {
      sendMessage({ input: { message } });
    },
  }));

  return (
    <ChatUIContextProvider>
      <ColumnLayout left={<Chating name={name} />} right={<SideAppViewBrowser />} />
    </ChatUIContextProvider>
  );
});

