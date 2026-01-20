import { AxiomLatticeProvider, LatticeChatShellContextProvider, SkillFlow } from "@axiom-lattice/react-sdk";
import { TOKEN_KEY } from "../../../authProvider";
const apiUrl = "http://localhost:6203";
export const Skills = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>

      <LatticeChatShellContextProvider initialConfig={{
        baseURL: apiUrl,
        apiKey: localStorage.getItem(TOKEN_KEY) || "",
        transport: "sse",
      }} >
        <SkillFlow />
      </LatticeChatShellContextProvider>
    </div>
  );
};
