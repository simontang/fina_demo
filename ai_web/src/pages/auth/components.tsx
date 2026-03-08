import React from "react";
import { RobotOutlined } from "@ant-design/icons";

export const AuthLogo: React.FC = () => (
  <div style={{ 
    display: "flex", 
    alignItems: "center", 
    justifyContent: "center",
    marginBottom: 24 
  }}>
    <RobotOutlined style={{ fontSize: 48, color: "#1890ff" }} />
  </div>
);

export const AuthTitle: React.FC = () => (
  <div style={{ textAlign: "center", marginBottom: 8 }}>
    <h1 style={{ fontSize: 24, fontWeight: 600, margin: 0 }}>
      FULI Agent Center
    </h1>
    <p style={{ color: "#666", margin: "8px 0 0 0" }}>
      数据智能分析平台
    </p>
  </div>
);
