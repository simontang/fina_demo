import { Authenticated, Refine } from "@refinedev/core";

import {
  ErrorComponent,
  ThemedLayout,
  ThemedSider,
  useNotificationProvider,
} from "@refinedev/antd";
import "@refinedev/antd/dist/reset.css";

import routerProvider, {
  CatchAllNavigate,
  DocumentTitleHandler,
  NavigateToResource,
  UnsavedChangesNotifier,
} from "@refinedev/react-router";
import dataProvider from "@refinedev/simple-rest";
import { App as AntdApp } from "antd";
import { BrowserRouter, Outlet, Route, Routes } from "react-router";
import { authProvider, createAuthenticatedDataProvider } from "./authProvider";
import { Header } from "./components/header";
import { CustomSiderWrapper } from "./components/custom-sider-wrapper";
import { ColorModeContextProvider } from "./contexts/color-mode";
import { AgentList } from "./pages/agents";
import { Login } from "./pages/login";
import { RobotOutlined, BarChartOutlined, FolderOutlined } from "@ant-design/icons";
// Agent Center pages
import { DeepResearchAgentList } from "./pages/agents/deep-research";
import { DeepResearchNewChat } from "./pages/agents/deep-research/new";
import { DeepResearchCritique } from "./pages/agents/deep-research/critique";
import { DeepResearchResearch } from "./pages/agents/deep-research/research";
import { DataAgentList } from "./pages/agents/data";
import { VoiceAgentList } from "./pages/agents/voice";
import { ExceptionHandlerAgentList } from "./pages/agents/exception-handler";
// Prediction Center pages
import { Segmentation } from "./pages/prediction/segmentation";
import { SalesForecast } from "./pages/prediction/sales-forecast";
import { InventoryAllocation } from "./pages/prediction/inventory";
import { RFMEngine } from "./pages/prediction/rfm";
// Asset Center pages
import { Datasets } from "./pages/assets/datasets";
import { DatasetDetail } from "./pages/assets/datasets/detail";
import { Skills } from "./pages/assets/skills";
import { Models } from "./pages/assets/models";
const apiUrl = import.meta.env.VITE_API_URL;

function App() {
  const dataProvider = createAuthenticatedDataProvider();

  return (
    <BrowserRouter basename="/admin">
      <ColorModeContextProvider>
        <AntdApp>
          <Refine
            dataProvider={dataProvider}
            notificationProvider={useNotificationProvider}
            routerProvider={routerProvider}
            // authProvider={authProvider}
            resources={[
              // Agent Center
              {
                name: "agent-center",
                meta: {
                  label: "Agent Center",
                  icon: <RobotOutlined />,
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
                name: "exception-handler-agent",
                list: "/agents/exception-handler",
                meta: {
                  label: "Exception Handler Agent",
                  parent: "agent-center",
                },
              },
              // Prediction Center
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
              // Asset Center
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
                  element={<NavigateToResource resource="deep-research-agent" />}
                />
                {/* Agent Center Routes */}
                <Route path="/agents">
                  <Route index element={<AgentList />} />
                  <Route path="deep-research">
                    <Route index element={<DeepResearchAgentList />} />
                    <Route path="new" element={<DeepResearchNewChat />} />
                    <Route path="critique" element={<DeepResearchCritique />} />
                    <Route path="research" element={<DeepResearchResearch />} />
                  </Route>
                  <Route path="data" element={<DataAgentList />} />
                  <Route path="voice" element={<VoiceAgentList />} />
                  <Route path="exception-handler" element={<ExceptionHandlerAgentList />} />
                </Route>
                {/* Prediction Center Routes */}
                <Route path="/prediction">
                  <Route path="segmentation" element={<Segmentation />} />
                  <Route path="sales-forecast" element={<SalesForecast />} />
                  <Route path="inventory" element={<InventoryAllocation />} />
                  <Route path="rfm" element={<RFMEngine />} />
                </Route>
                {/* Asset Center Routes */}
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
                element={
                  <Authenticated
                    key="authenticated-outer"
                    fallback={<Outlet />}
                  >
                    <NavigateToResource />
                  </Authenticated>
                }
              >
                <Route path="/login" element={<Login />} />
                {/* <Route path="/register" element={<Register />} />
                    <Route
                      path="/forgot-password"
                      element={<ForgotPassword />}
                    /> */}
              </Route>
            </Routes>

            <UnsavedChangesNotifier />
            <DocumentTitleHandler />
          </Refine>
        </AntdApp>
      </ColorModeContextProvider>
    </BrowserRouter>
  );
}

export default App;
