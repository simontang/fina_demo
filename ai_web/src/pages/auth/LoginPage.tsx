import React from "react";
import { ConfigProvider } from "antd";
import {
  AuthProvider,
  AxiomLatticeProvider,
  LoginPage as AxiomLoginPage,
  RegisterForm,
  TenantSelector,
  useAuth,
} from "@axiom-lattice/react-sdk";
import { LatticeChatShell } from "@axiom-lattice/react-sdk";
import { generateTheme } from "../../theme";
import "antd/dist/reset.css";
import { getBaseAPIPath } from "../../getBaseAPIPath";
import Logo from "../../components/Logo";
import { useNavigate } from "react-router";
import { setToken, setUser, clearAuth } from "../../utils/sessionStorage";
import { getAppConfig } from "../../config";

// Create theme with custom primary color
const customTheme = generateTheme("#5b50c6");

// 动态适配网关地址
const GATEWAY_URL = getBaseAPIPath();

function AppWithAuth() {
  return (
    <AuthProvider
      baseURL={GATEWAY_URL}
      onLoginSuccess={(user, tenants) => {
        // Sync to sessionStorage so Refine's authProvider.check() passes and avoids redirect loop to /login
        setToken("axiom-session");
        setUser({
          id: user.id,
          name: user.name,
          email: user.email,
        });
      }}
      onTenantSelected={(tenant) => {
        console.log("Selected tenant:", tenant.name);
      }}
      onLogout={() => {
        console.log("Logged out");
      }}
    >
      <AppContent />
    </AuthProvider>
  );
}

function AppContent() {
  const navigate = useNavigate();

  const {
    isAuthenticated,
    user,
    tenants,
    currentTenant,
    selectTenant,
    logout
  } = useAuth();

  const [showRegister, setShowRegister] = React.useState(false);

  if (!isAuthenticated) {
    if (showRegister) {
      return (
        <div style={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#ffffff"
        }}>
          <RegisterForm
            onSuccess={() => setShowRegister(false)}
            onCancel={() => setShowRegister(false)}
            footer={
              <p style={{ textAlign: "center", marginTop: "20px" }}>
                Already have an account?{" "}
                <button
                  onClick={() => setShowRegister(false)}
                  style={{
                    background: "none",
                    border: "none",
                    color: "#1890ff",
                    cursor: "pointer",
                    textDecoration: "underline"
                  }}
                >
                  Sign in
                </button>
              </p>
            }
          />
        </div>
      );
    }

    return (
      <div style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#ffffff"
      }}>
        <AxiomLoginPage
          title={getAppConfig().appName}
          logo={<Logo />}
          footer={
            <p style={{ textAlign: "center", marginTop: "20px" }}>
              Don&apos;t have an account?{" "}
              <button
                onClick={() => setShowRegister(true)}
                style={{
                  background: "none",
                  border: "none",
                  color: "#1890ff",
                  cursor: "pointer",
                  textDecoration: "underline"
                }}
              >
                Create one
              </button>
            </p>
          }
        />
      </div>
    );
  }

  if (tenants.length === 0) {
    return (
      <div style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#ffffff",
        flexDirection: "column",
        gap: "20px"
      }}>
        <div style={{ textAlign: "center" }}>
          <h2>Welcome, {user?.name}!</h2>
          <p style={{ color: "#666", marginTop: "10px" }}>
            You don&apos;t have access to any tenants yet.
          </p>
          <p style={{ color: "#999", fontSize: "14px", marginTop: "5px" }}>
            Please contact your administrator to be assigned to a tenant.
          </p>
        </div>
        <button
          onClick={() => {
            clearAuth();
            logout();
          }}
          style={{
            padding: "10px 20px",
            background: "#f0f0f0",
            border: "none",
            borderRadius: "6px",
            cursor: "pointer"
          }}
        >
          Logout
        </button>
      </div>
    );
  }

  if (!currentTenant && tenants.length > 0) {
    return (
      <div style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#ffffff"
      }}>
        <TenantSelector
          tenants={tenants.map(t => t.tenant).filter(Boolean) as any}
          onSelect={async (selectedTenant) => {
            await selectTenant(selectedTenant.id);
            navigate("/agents/data");
          }}
          title="Select Your Tenant"
          description={`Welcome, ${user?.name}! Choose a tenant to continue`}
        />
      </div>
    );
  }


}



export const LoginPage: React.FC = () => {



  return <ConfigProvider theme={customTheme}>
    <AppWithAuth />
  </ConfigProvider>
};
