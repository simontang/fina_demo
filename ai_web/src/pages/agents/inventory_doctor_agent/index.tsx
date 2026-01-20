import { ChatBotCompnent } from "../../../components/chating";

export const InventoryDoctorAgentList = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <ChatBotCompnent
        id="inventory_doctor_agent"
        threadId="inventory_doctor_agent_thread_1"
        name="Inventory Doctor Agent"
      />
    </div>
  );
};
