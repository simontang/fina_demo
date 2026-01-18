import React from "react";
import { ThemedSider } from "@refinedev/antd";
import type { RefineThemedLayoutSiderProps } from "@refinedev/antd";
import { Menu } from "antd";
import { useNavigate, useLocation } from "react-router";
import {
  RobotOutlined,
  BarChartOutlined,
  FolderOutlined,
  SearchOutlined,
  DatabaseOutlined,
  SoundOutlined,
  WarningOutlined,
  UserOutlined,
  ShoppingOutlined,
  InboxOutlined,
  PieChartOutlined,
  FolderOpenOutlined,
  ToolOutlined,
  AppstoreOutlined,
} from "@ant-design/icons";
import type { MenuProps } from "antd";

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
  // 不再使用 useMenu，因为我们完全自定义菜单
  // const { menuItems } = useMenu();

  // 构建自定义菜单项
  const customMenuItems: MenuItem[] = [
    getItem("Agent Center", "agent-center", <RobotOutlined />, [
      getItem("Deep Research Agent", "deep-research-agent", <SearchOutlined />),
      getItem("Data Agent", "data-agent", <DatabaseOutlined />),
      getItem("Voice Agent", "voice-agent", <SoundOutlined />),
      getItem("Exception Handler Agent", "exception-handler-agent", <WarningOutlined />),
    ]),
    getItem("Prediction Center", "prediction-center", <BarChartOutlined />, [
      getItem("Segmentation", "segmentation", <UserOutlined />),
      getItem("Sales Forecast", "sales-forecast", <ShoppingOutlined />),
      getItem("Inventory Allocation", "inventory-allocation", <InboxOutlined />),
      getItem("RFM 引擎", "rfm-engine", <PieChartOutlined />),
    ]),
    getItem("Asset Center", "asset-center", <FolderOutlined />, [
      getItem("Datasets", "datasets", <FolderOpenOutlined />),
      getItem("Models", "models", <AppstoreOutlined />),
      getItem("Skills", "skills", <ToolOutlined />),
    ]),
  ];

  const pathMap: Record<string, string> = {
    "deep-research-agent": "/agents/deep-research",
    "data-agent": "/agents/data",
    "voice-agent": "/agents/voice",
    "exception-handler-agent": "/agents/exception-handler",
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
      "/admin/agents/deep-research": "deep-research-agent",
      "/admin/agents/data": "data-agent",
      "/admin/agents/voice": "voice-agent",
      "/admin/agents/exception-handler": "exception-handler-agent",
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
      render={({ items, dashboard, logout, collapsed }) => {
        const onClick: MenuProps["onClick"] = (e) => {
          const path = pathMap[e.key as string];
          if (path) {
            navigate(path);
          }
        };

        return (
          <>
            {dashboard}
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
