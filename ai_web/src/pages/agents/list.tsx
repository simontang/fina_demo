import { ChatBotCompnent } from "../../components/chating";

export const AgentList = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <ChatBotCompnent
        id="Research_data_agent"
        threadId="1"
        name="Research Data Agent"
      />
    </div>
  );
};
