import { ChatBotCompnent } from "../../../components/chating";

export const DataAgentList = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <ChatBotCompnent
        id="data_agent"
        threadId="1"
        name="Data Agent"
      />
    </div>
  );
};
