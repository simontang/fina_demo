import React, { useState } from "react";
import { Menu, MenuProps, Layout } from "antd";
import { useNavigate, useLocation } from "react-router";
import type { RefineThemedLayoutSiderProps } from "@refinedev/antd";
import {
  RobotOutlined,
  BarChartOutlined,
  FolderOutlined,
  SearchOutlined,
  DatabaseOutlined,
  SoundOutlined,
  WarningOutlined,
  MessageOutlined,
  FileTextOutlined,
  UserOutlined,
  ShoppingOutlined,
  InboxOutlined,
  FolderOpenOutlined,
  ToolOutlined,
} from "@ant-design/icons";

const { Sider: AntdSider } = Layout;

type MenuItem = Required<MenuProps>["items"][number];

function getItem(
  label: React.ReactNode,
  key: React.Key,
  icon?: React.ReactNode,
  children?: MenuItem[],
  type?: "group"
): MenuItem {
  return {
    key,
    icon,
    children,
    label,
    type,
  } as MenuItem;
}

export const CustomSider: React.FC<RefineThemedLayoutSiderProps> = ({
  siderItemsAreCollapsed,
}) => {
  const collapsed = Boolean(siderItemsAreCollapsed);
  const navigate = useNavigate();
  const location = useLocation();
  const [openKeys, setOpenKeys] = useState<string[]>([
    "agent-center",
    "deep-research-agent",
  ]);

  const onOpenChange = (keys: string[]) => {
    setOpenKeys(keys);
  };

  const onClick: MenuProps["onClick"] = (e) => {
    const pathMap: Record<string, string> = {
      "deep-research-agent": "/agents/deep-research",
      "deep-research-new-chat": "/agents/deep-research/new",
      "critique-agent": "/agents/deep-research/critique",
      "research-agent": "/agents/deep-research/research",
      "data-agent": "/agents/data",
      "voice-agent": "/agents/voice",
      "exception-handler-agent": "/agents/exception-handler",
      segmentation: "/prediction/segmentation",
      "sales-forecast": "/prediction/sales-forecast",
      "inventory-allocation": "/prediction/inventory",
      datasets: "/assets/datasets",
      skills: "/assets/skills",
    };

    const path = pathMap[e.key as string];
    if (path) {
      navigate(path);
    }
  };

  // 根据当前路径确定选中的菜单项
  const getSelectedKeys = (): string[] => {
    const path = location.pathname;
    const pathToKey: Record<string, string> = {
      "/admin/agents/deep-research": "deep-research-agent",
      "/admin/agents/deep-research/new": "deep-research-new-chat",
      "/admin/agents/deep-research/critique": "critique-agent",
      "/admin/agents/deep-research/research": "research-agent",
      "/admin/agents/data": "data-agent",
      "/admin/agents/voice": "voice-agent",
      "/admin/agents/exception-handler": "exception-handler-agent",
      "/admin/prediction/segmentation": "segmentation",
      "/admin/prediction/sales-forecast": "sales-forecast",
      "/admin/prediction/inventory": "inventory-allocation",
      "/admin/assets/datasets": "datasets",
      "/admin/assets/skills": "skills",
    };

    for (const [keyPath, key] of Object.entries(pathToKey)) {
      if (path.startsWith(keyPath)) {
        return [key];
      }
    }
    return [];
  };

  const items: MenuItem[] = [
    getItem("Agent Center", "agent-center", <RobotOutlined />, [
      getItem("Deep Research Agent", "deep-research-agent", <SearchOutlined />, [
        getItem("New Chat", "deep-research-new-chat", <MessageOutlined />),
        getItem("Critique Agent", "critique-agent", <FileTextOutlined />),
        getItem("Research Agent", "research-agent", <SearchOutlined />),
      ]),
      getItem("Data Agent", "data-agent", <DatabaseOutlined />),
      getItem("Voice Agent", "voice-agent", <SoundOutlined />),
      getItem("异常 Handler Agent", "exception-handler-agent", <WarningOutlined />),
    ]),
    getItem("Prediction Center", "prediction-center", <BarChartOutlined />, [
      getItem("Segmentation", "segmentation", <UserOutlined />),
      getItem("销量预测", "sales-forecast", <ShoppingOutlined />),
      getItem("库存分配", "inventory-allocation", <InboxOutlined />),
    ]),
    getItem("Asset Center", "asset-center", <FolderOutlined />, [
      getItem("数据集", "datasets", <FolderOpenOutlined />),
      getItem("Skills", "skills", <ToolOutlined />),
    ]),
  ];

  return (
    <AntdSider
      trigger={null}
      collapsible
      collapsed={collapsed}
      width={256}
      style={{
        overflow: "auto",
        height: "100vh",
      }}
    >
      <div style={{ padding: "16px" }}>
        <Menu
          mode="inline"
          theme="light"
          items={items}
          openKeys={collapsed ? [] : openKeys}
          onOpenChange={onOpenChange}
          selectedKeys={getSelectedKeys()}
          onClick={onClick}
          style={{ height: "100%", borderRight: 0 }}
        />
      </div>
    </AntdSider>
  );
};
