import { AxiomLatticeProvider, LatticeChatShell } from "@axiom-lattice/react-sdk";
import { useRef } from "react";
import { getBaseAPIPath } from "../../../getBaseAPIPath";
import { TOKEN_KEY } from "../../../authProvider";
import { getCurrentTenant } from "../../../utils/sessionStorage";
import Logo from "../../../components/Logo";
import ProjectTemplatesPortal from "../../../components/agent-templates/ProjectTemplatesPortal";
import { getAppConfig } from "../../../config";

const baseURL = getBaseAPIPath();

export const DataAgentList = () => {
  const tenant = getCurrentTenant();
  const shellRootRef = useRef<HTMLDivElement>(null);

  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <AxiomLatticeProvider
        config={{
          baseURL,
          apiKey: localStorage.getItem(TOKEN_KEY) || "",
          assistantId: "",
          transport: "sse",
          headers: tenant?.id ? { "x-tenant-id": tenant.id } : undefined,
        }}
      >
        <div ref={shellRootRef} style={{ height: "100%", width: "100%" }}>
          <LatticeChatShell
            initialConfig={{
              baseURL,
              enableSkillSlot: false,
              enableDatabaseSlot: false,
              resourceFolders: [
                { name: "tmp", displayName: "Working Directory", allowUpload: true },
                { name: "artifacts", displayName: "artifacts", allowUpload: false },
              ],
              enableWorkspace: true,
              enableThreadCreation: true,
              enableThreadList: true,
              globalSharedSandboxURL: "https://demo.alphafina.cn/sandbox/global",
              sidebarMode: "expanded",
              sidebarDefaultExpanded: true,
              sidebarShowToggle: true,
              sidebarShowNewAnalysis: false,
              sidebarLogoText: getAppConfig().appName,
              sidebarLogoIcon: <Logo width={28} height={28} />,
              assistantId: "new-data-agent",
              showSideMenu: false,
            }}
          />
        </div>
        <ProjectTemplatesPortal shellRootRef={shellRootRef} />
      </AxiomLatticeProvider>
    </div>
  );
};
