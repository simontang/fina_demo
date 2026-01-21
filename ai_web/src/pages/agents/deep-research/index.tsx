import { LatticeChatShell } from "@axiom-lattice/react-sdk";
import { getBaseAPIPath } from "../../../getBaseAPIPath";
import { TOKEN_KEY } from "../../../authProvider";

export const DeepResearchAgentList = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <LatticeChatShell initialConfig={{
        baseURL: getBaseAPIPath(),
        apiKey: localStorage.getItem(TOKEN_KEY) || "",
        transport: "sse",
        enableThreadCreation: true,
        enableThreadList: true,
        assistantId: "deep_research_agent",
        showSideMenu: false
      }} />

    </div>
  );
};
