import { LatticeChatShell } from "@axiom-lattice/react-sdk";
import { TOKEN_KEY } from "../../../authProvider";
import { getBaseAPIPath } from "../../../getBaseAPIPath";

export const ComputerUseAgentList = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <LatticeChatShell initialConfig={{
        baseURL: getBaseAPIPath(),
        apiKey: localStorage.getItem(TOKEN_KEY) || "",
        transport: "sse",
        enableThreadCreation: true,
        enableThreadList: true,
        assistantId: "sandbox_agent",
        showSideMenu: false,
        globalSharedSandboxURL: "https://demo.alphafina.cn/sandbox/global",

      }} />
    </div>
  );
};
