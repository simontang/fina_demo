import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { TenantSelector, useAuth } from "@axiom-lattice/react-sdk";
import { Tenant } from "@axiom-lattice/protocols";
import { Spin } from "antd";
import { setCurrentTenant } from "../../utils/sessionStorage";
import { AuthLogo, AuthTitle } from "./components";

export const TenantSelectPage: React.FC = () => {
  const navigate = useNavigate();
  const { tenants, fetchUserTenants, isLoading } = useAuth();
  const [tenantList, setTenantList] = useState<Tenant[]>([]);

  useEffect(() => {
    fetchUserTenants();
  }, [fetchUserTenants]);

  useEffect(() => {
    if (tenants && tenants.length > 0) {
      // Extract tenant info from UserTenantInfo
      const list = tenants.map((t) => ({
        id: t.tenantId,
        name: t.tenant?.name || t.tenantId,
        description: t.tenant?.description || "",
        createdAt: t.tenant?.createdAt || new Date().toISOString(),
        updatedAt: t.tenant?.updatedAt || new Date().toISOString(),
      })) as Tenant[];
      setTenantList(list);
    }
  }, [tenants]);

  const handleSelect = (tenant: Tenant) => {
    setCurrentTenant(tenant);
    navigate("/agents/data");
  };

  if (isLoading) {
    return (
      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
        }}
      >
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
        padding: 24,
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 640,
          padding: "40px 32px",
          background: "#fff",
          borderRadius: 12,
          boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
        }}
      >
        <AuthLogo />
        <AuthTitle />
        <TenantSelector
          tenants={tenantList}
          onSelect={handleSelect}
          isLoading={isLoading}
          title="选择租户"
          description="请选择一个租户继续访问"
        />
      </div>
    </div>
  );
};
