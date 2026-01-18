import { ChatBotCompnent } from "../../../components/chating";

export const VoiceAgentList = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <ChatBotCompnent
        id="voice_agent"
        threadId="1"
        name="Voice Agent"
      />
    </div>
  );
};
