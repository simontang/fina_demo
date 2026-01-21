import { LatticeChatShellContextProvider, AssistantFlow } from "@axiom-lattice/react-sdk";
import { TOKEN_KEY } from "../../../authProvider";

const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:6203/api";

export const AgentOverview = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <LatticeChatShellContextProvider
        initialConfig={{
          baseURL: apiUrl,
          apiKey: localStorage.getItem(TOKEN_KEY) || "",
          transport: "sse",
        }}
      >
        <AssistantFlow />
      </LatticeChatShellContextProvider>
    </div>
  );
};
