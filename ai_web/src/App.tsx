import { Authenticated, Refine } from "@refinedev/core";
import {
  ErrorComponent,
  ThemedLayout,
  useNotificationProvider,
} from "@refinedev/antd";
import "@refinedev/antd/dist/reset.css";

import { BarChartOutlined, FolderOutlined, RobotOutlined } from "@ant-design/icons";
import routerProvider, {
  CatchAllNavigate,
  DocumentTitleHandler,
  NavigateToResource,
  UnsavedChangesNotifier,
} from "@refinedev/react-router";
import { App as AntdApp } from "antd";
import { BrowserRouter, Outlet, Route, Routes } from "react-router";
import { AxiomLatticeProvider, AuthProvider as AxiomAuthProvider, useAuth } from "@axiom-lattice/react-sdk";

import { createAuthenticatedDataProvider } from "./authProvider";
import { axiomAuthProvider } from "./authAdapter";
import { CustomSiderWrapper } from "./components/custom-sider-wrapper";
import { Header } from "./components/header";
import { ColorModeContextProvider } from "./contexts/color-mode";
import { AgentList } from "./pages/agents";
import { LoginPage, RegisterPage, TenantSelectPage } from "./pages/auth";
import { DataAgentList } from "./pages/agents/data";
import { DeepResearchAgentList } from "./pages/agents/deep-research";
import { DeepResearchCritique } from "./pages/agents/deep-research/critique";
import { DeepResearchNewChat } from "./pages/agents/deep-research/new";
import { DeepResearchResearch } from "./pages/agents/deep-research/research";
import { ComputerUseAgentList } from "./pages/agents/computer_use_agent";
import { AgentOverview } from "./pages/agents/overview";
import { VoiceAgentRtc } from "./pages/agents/voice/rtc";
import { InventoryAllocation } from "./pages/prediction/inventory";
import { RFMEngine } from "./pages/prediction/rfm";
import { SalesForecast } from "./pages/prediction/sales-forecast";
import { Segmentation } from "./pages/prediction/segmentation";
import { Datasets } from "./pages/assets/datasets";
import { DatasetDetail } from "./pages/assets/datasets/detail";
import { Models } from "./pages/assets/models";
import { Skills } from "./pages/assets/skills";
import { AgentWorkbench } from "./pages/workbench";
import { getCurrentTenant } from "./utils/sessionStorage";
import { getBaseAPIPath } from "./getBaseAPIPath";

// Get API base URL
const baseURL = getBaseAPIPath();

// Refine App Content - rendered when user is authenticated
const RefineAppContent: React.FC = () => {
  const dataProvider = createAuthenticatedDataProvider();

  return (
    <Refine
      dataProvider={dataProvider}
      notificationProvider={useNotificationProvider}
      routerProvider={routerProvider}
      authProvider={axiomAuthProvider}
      resources={[
        {
          name: "agent-center",
          meta: {
            label: "Agent Center",
            icon: <RobotOutlined />,
          },
        },
        {
          name: "agent-overview",
          list: "/agents/overview",
          meta: {
            label: "Overview",
            parent: "agent-center",
          },
        },
        {
          name: "deep-research-agent",
          list: "/agents/deep-research",
          meta: {
            label: "Deep Research Agent",
            parent: "agent-center",
          },
        },
        {
          name: "data-agent",
          list: "/agents/data",
          meta: {
            label: "Data Agent",
            parent: "agent-center",
          },
        },
        {
          name: "voice-agent",
          list: "/agents/voice",
          meta: {
            label: "Voice Agent",
            parent: "agent-center",
          },
        },
        {
          name: "computer-use-agent",
          list: "/agents/computer_use_agent",
          meta: {
            label: "Computer Use Agent",
            parent: "agent-center",
          },
        },
        {
          name: "prediction-center",
          meta: {
            label: "Prediction Center",
            icon: <BarChartOutlined />,
          },
        },
        {
          name: "segmentation",
          list: "/prediction/segmentation",
          meta: {
            label: "Segmentation",
            parent: "prediction-center",
          },
        },
        {
          name: "sales-forecast",
          list: "/prediction/sales-forecast",
          meta: {
            label: "Sales Forecast",
            parent: "prediction-center",
          },
        },
        {
          name: "inventory-allocation",
          list: "/prediction/inventory",
          meta: {
            label: "Inventory Allocation",
            parent: "prediction-center",
          },
        },
        {
          name: "rfm-engine",
          list: "/prediction/rfm",
          meta: {
            label: "RFM Engine",
            parent: "prediction-center",
          },
        },
        {
          name: "asset-center",
          meta: {
            label: "Asset Center",
            icon: <FolderOutlined />,
          },
        },
        {
          name: "datasets",
          list: "/assets/datasets",
          show: "/assets/datasets/:id",
          meta: {
            label: "Datasets",
            parent: "asset-center",
          },
        },
        {
          name: "models",
          list: "/assets/models",
          meta: {
            label: "Models",
            parent: "asset-center",
          },
        },
        {
          name: "skills",
          list: "/assets/skills",
          meta: {
            label: "Skills",
            parent: "asset-center",
          },
        },
      ]}
      options={{
        syncWithLocation: true,
        warnWhenUnsavedChanges: true,
      }}
    >
      <Routes>
        <Route
          element={
            <Authenticated
              key="authenticated-inner"
              fallback={<CatchAllNavigate to="/login" />}
            >
              <ThemedLayout
                initialSiderCollapsed={false}
                Title={() => (
                  <RobotOutlined
                    style={{ fontSize: "24px", color: "inherit" }}
                  />
                )}
                Header={Header}
                Sider={(props) => <CustomSiderWrapper {...props} fixed />}
              >
                <Outlet />
              </ThemedLayout>
            </Authenticated>
          }
        >
          <Route
            index
            element={<NavigateToResource resource="data-agent" />}
          />
          <Route path="/agents">
            <Route index element={<AgentList />} />
            <Route path="overview" element={<AgentOverview />} />
            <Route path="deep-research">
              <Route index element={<DeepResearchAgentList />} />
              <Route path="new" element={<DeepResearchNewChat />} />
              <Route path="critique" element={<DeepResearchCritique />} />
              <Route path="research" element={<DeepResearchResearch />} />
            </Route>
            <Route path="data" element={<DataAgentList />} />
            <Route path="voice" element={<VoiceAgentRtc />} />
            <Route path="computer_use_agent" element={<ComputerUseAgentList />} />
          </Route>
          <Route path="/prediction">
            <Route path="segmentation" element={<Segmentation />} />
            <Route path="sales-forecast" element={<SalesForecast />} />
            <Route path="inventory" element={<InventoryAllocation />} />
            <Route path="rfm" element={<RFMEngine />} />
          </Route>
          <Route path="/assets">
            <Route path="datasets">
              <Route index element={<Datasets />} />
              <Route path=":id" element={<DatasetDetail />} />
            </Route>
            <Route path="models" element={<Models />} />
            <Route path="skills" element={<Skills />} />
          </Route>
          <Route path="*" element={<ErrorComponent />} />
        </Route>

        <Route
          path="/workbench"
          element={
            <Authenticated
              key="authenticated-workbench"
              fallback={<CatchAllNavigate to="/login" />}
            >
              <AgentWorkbench />
            </Authenticated>
          }
        />

        <Route path="*" element={<ErrorComponent />} />
      </Routes>

      <UnsavedChangesNotifier />
      <DocumentTitleHandler />
    </Refine>
  );
};

// App Content with auth logic
const AppContent: React.FC = () => {
  const { isAuthenticated, tenants, isLoading } = useAuth();
  const currentTenant = getCurrentTenant();

  if (isLoading) {
    return (
      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        加载中...
      </div>
    );
  }

  // Not authenticated - show auth routes
  if (!isAuthenticated) {
    return (
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="*" element={<CatchAllNavigate to="/login" />} />
      </Routes>
    );
  }

  // Authenticated but needs tenant selection (multi-tenant users)
  if (tenants.length > 1 && !currentTenant) {
    return (
      <Routes>
        <Route path="/tenant-select" element={<TenantSelectPage />} />
        <Route path="*" element={<CatchAllNavigate to="/tenant-select" />} />
      </Routes>
    );
  }

  // Authenticated and ready - show Refine app
  return <RefineAppContent />;
};

function App() {
  return (
    <BrowserRouter basename="/admin">
      <ColorModeContextProvider>
        <AntdApp>
          <AxiomLatticeProvider
            config={{
              baseURL: baseURL,
              apiKey: "",
              assistantId: "",
              transport: "sse",
            }}
          >
            <AxiomAuthProvider baseURL={baseURL}>
              <AppContent />
            </AxiomAuthProvider>
          </AxiomLatticeProvider>
        </AntdApp>
      </ColorModeContextProvider>
    </BrowserRouter>
  );
}

export default App;
