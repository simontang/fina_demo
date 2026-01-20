import { ChatBotCompnent } from "../../../components/chating";

export const DeepResearchAgentList = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <ChatBotCompnent
        id="deep_research_agent"
        threadId="deep_research_agent_thread_1"
        name="Deep Research Agent"
      />
    </div>
  );
};
