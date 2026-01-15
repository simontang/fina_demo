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
import { ColorModeContextProvider } from "./contexts/color-mode";
import { AgentList } from "./pages/agents";
import { Login } from "./pages/login";
import { RobotOutlined } from "@ant-design/icons";
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
              {
                name: "agents",
                list: "/agents",
                show: "/agents/show/:id",
                meta: {
                  canDelete: false,
                  canCreate: false,
                  label: "Agent Center",
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
                      initialSiderCollapsed
                      Title={() => (
                        <RobotOutlined
                          style={{ fontSize: "24px", color: "inherit" }}
                        />
                      )}
                      Header={Header}
                      Sider={(props) => <ThemedSider {...props} fixed />}
                    >
                      <Outlet />
                    </ThemedLayout>
                  </Authenticated>
                }
              >
                <Route
                  index
                  element={<NavigateToResource resource="agents" />}
                />
                <Route path="/agents">
                  <Route index element={<AgentList />} />
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
