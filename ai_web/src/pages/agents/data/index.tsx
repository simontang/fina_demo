import { ChatBotCompnent } from "../../../components/chating";

export const DataAgentList = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <ChatBotCompnent
        id="data_agent"
        threadId="data_agent_thread_1"
        name="Data Agent"
      />
    </div>
  );
};
