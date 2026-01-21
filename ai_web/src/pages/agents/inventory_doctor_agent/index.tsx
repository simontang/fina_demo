import { LatticeChatShell } from "@axiom-lattice/react-sdk";
import { TOKEN_KEY } from "../../../authProvider";
import { getBaseAPIPath } from "../../../getBaseAPIPath";

export const InventoryDoctorAgentList = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <LatticeChatShell initialConfig={{
        baseURL: getBaseAPIPath(),
        apiKey: localStorage.getItem(TOKEN_KEY) || "",
        transport: "sse",
        enableThreadCreation: true,
        enableThreadList: true,
        assistantId: "inventory_doctor_agent",
        showSideMenu: false
      }} />
    </div>
  );
};
