import {
  AppstoreOutlined,
  BarChartOutlined,
  DashboardOutlined,
  DatabaseOutlined,
  FolderOpenOutlined,
  FolderOutlined,
  InboxOutlined,
  PieChartOutlined,
  RobotOutlined,
  SearchOutlined,
  ShoppingOutlined,
  SoundOutlined,
  ToolOutlined,
  UserOutlined,
  WarningOutlined
} from "@ant-design/icons";
import type { RefineThemedLayoutSiderProps } from "@refinedev/antd";
import { ThemedSider } from "@refinedev/antd";
import type { MenuProps } from "antd";
import { Menu } from "antd";
import React from "react";
import { useLocation, useNavigate } from "react-router";

type MenuItem = Required<MenuProps>["items"][number];

function getItem(
  label: React.ReactNode,
  key: React.Key,
  icon?: React.ReactNode,
  children?: MenuItem[]
): MenuItem {
  return {
    key,
    icon,
    children,
    label,
  } as MenuItem;
}

export const CustomSiderWrapper: React.FC<RefineThemedLayoutSiderProps> = (
  props
) => {
  const navigate = useNavigate();
  const location = useLocation();
  // We don't use useMenu here because we fully customize the menu.
  // const { menuItems } = useMenu();

  // Build custom menu items
  const customMenuItems: MenuItem[] = [
    getItem("Agent Center", "agent-center", <RobotOutlined />, [
      getItem("Overview", "agent-overview", <DashboardOutlined />),
      getItem("Deep Research Agent", "deep-research-agent", <SearchOutlined />),
      getItem("Data Agent", "data-agent", <DatabaseOutlined />),
      getItem("Voice Agent", "voice-agent", <SoundOutlined />),
      getItem("Inventory Doctor Agent", "inventory-doctor-agent", <WarningOutlined />),
    ]),
    getItem("Prediction Center", "prediction-center", <BarChartOutlined />, [
      getItem("Segmentation", "segmentation", <UserOutlined />),
      getItem("Sales Forecast", "sales-forecast", <ShoppingOutlined />),
      getItem("Inventory Allocation", "inventory-allocation", <InboxOutlined />),
      getItem("RFM Engine", "rfm-engine", <PieChartOutlined />),
    ]),
    getItem("Asset Center", "asset-center", <FolderOutlined />, [
      getItem("Datasets", "datasets", <FolderOpenOutlined />),
      getItem("Models", "models", <AppstoreOutlined />),
      getItem("Skills", "skills", <ToolOutlined />),
    ]),
  ];

  const pathMap: Record<string, string> = {
    "agent-overview": "/agents/overview",
    "deep-research-agent": "/agents/deep-research",
    "data-agent": "/agents/data",
    "voice-agent": "/agents/voice",
    "voice-agent-chat": "/agents/voice",
    "voice-agent-rtc": "/agents/voice/rtc",
    "inventory-doctor-agent": "/agents/inventory_doctor_agent",
    segmentation: "/prediction/segmentation",
    "sales-forecast": "/prediction/sales-forecast",
    "inventory-allocation": "/prediction/inventory",
    "rfm-engine": "/prediction/rfm",
    datasets: "/assets/datasets",
    models: "/assets/models",
    skills: "/assets/skills",
  };

  const getSelectedKeys = (): string[] => {
    const path = location.pathname;
    const pathToKey: Record<string, string> = {
      "/admin/agents/overview": "agent-overview",
      "/admin/agents/deep-research": "deep-research-agent",
      "/admin/agents/data": "data-agent",
      "/admin/agents/voice/rtc": "voice-agent-rtc",
      "/admin/agents/voice": "voice-agent-chat",
      "/admin/agents/inventory_doctor_agent": "inventory-doctor-agent",
      "/admin/prediction/segmentation": "segmentation",
      "/admin/prediction/sales-forecast": "sales-forecast",
      "/admin/prediction/inventory": "inventory-allocation",
      "/admin/prediction/rfm": "rfm-engine",
      "/admin/assets/datasets": "datasets",
      "/admin/assets/models": "models",
      "/admin/assets/skills": "skills",
    };

    for (const [keyPath, key] of Object.entries(pathToKey)) {
      if (path.startsWith(keyPath)) {
        return [key];
      }
    }
    return [];
  };

  return (
    <ThemedSider
      {...props}
      render={({ items, logout, collapsed }) => {
        const onClick: MenuProps["onClick"] = (e) => {
          const path = pathMap[e.key as string];
          if (path) {
            navigate(path);
          }
        };

        return (
          <>
            <Menu
              mode="inline"
              selectedKeys={getSelectedKeys()}
              defaultOpenKeys={collapsed ? [] : ["agent-center", "prediction-center", "asset-center"]}
              items={customMenuItems}
              onClick={onClick}
              style={{
                background: "transparent",
                borderRight: "none"
              }}
              triggerSubMenuAction="click"
            />
            {logout}
          </>
        );
      }}
    />
  );
};
