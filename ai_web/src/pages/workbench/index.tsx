import { Button } from "antd";
import { ArrowLeftOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router";
import { LatticeChatShell } from "@axiom-lattice/react-sdk";
import { getBaseAPIPath } from "../../getBaseAPIPath";

export const AgentWorkbench = () => {
  const navigate = useNavigate();

  return (
    <div style={{ width: "100vw", height: "100vh", overflow: "hidden", display: "flex", flexDirection: "column" }}>
      <div style={{
        height: "64px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 24px",
        borderBottom: "1px solid #e8e8e8",
        backgroundColor: "#fff"
      }}>
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={() => navigate(-1)}
        >
          返回
        </Button>
        <div style={{ fontSize: "16px", fontWeight: 600 }}>Agent Studio</div>
        <div style={{ width: "80px" }}></div>
      </div>

      <div style={{ flex: 1, overflow: "hidden" }}>
        <LatticeChatShell
          initialConfig={{
            resourceFolders: [
              { name: "/", displayName: "Root", allowUpload: true },
              { name: "assets", displayName: "Assets", allowUpload: true },
              { name: "metrics", displayName: "Metrics", allowUpload: true },
              { name: "outputs", displayName: "Outputs", allowUpload: false },
            ],
            enableWorkspace: true,
            enableThreadCreation: true,
            enableThreadList: true,
            baseURL: getBaseAPIPath(),
            globalSharedSandboxURL: "https://demo.alphafina.cn/sandbox/global"
          }}
        />
      </div>
    </div>
  );
};
