import { ChatBotCompnent } from "../../../components/chating";

export const DeepResearchNewChat = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <ChatBotCompnent
        id="deep_research_agent"
        threadId="new"
        name="New Chat"
      />
    </div>
  );
};
