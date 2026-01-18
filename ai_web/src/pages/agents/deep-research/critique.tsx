import { ChatBotCompnent } from "../../../components/chating";

export const DeepResearchCritique = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <ChatBotCompnent
        id="critique-agent"
        threadId="1"
        name="Critique Agent"
      />
    </div>
  );
};
