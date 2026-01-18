import { ChatBotCompnent } from "../../../components/chating";

export const ExceptionHandlerAgentList = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <ChatBotCompnent
        id="exception_handler_agent"
        threadId="1"
        name="异常 Handler Agent"
      />
    </div>
  );
};
