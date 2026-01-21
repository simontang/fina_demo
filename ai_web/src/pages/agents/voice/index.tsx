import { LatticeChatShell } from "@axiom-lattice/react-sdk";
import { getBaseAPIPath } from "../../../getBaseAPIPath";
import { TOKEN_KEY } from "../../../authProvider";

export const VoiceAgentList = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <LatticeChatShell initialConfig={{
        baseURL: getBaseAPIPath(),
        apiKey: localStorage.getItem(TOKEN_KEY) || "",
        transport: "sse",
        enableThreadCreation: true,
        enableThreadList: true,
        assistantId: "voice_agent",
        showSideMenu: false
      }} />
    </div>
  );
};
