import React, { useState, useCallback } from "react";
import { Button, Spin, Typography } from "antd";
import { CloseOutlined } from "@ant-design/icons";
import {
  AgentConversations,
  LatticeChat,
  AssistantContextProvider,
  ConversationContextProvider,
  useChatUIContext,
  useConversationContext,
  getElement,
} from "@axiom-lattice/react-sdk";

const { Text } = Typography;

export interface TaskAgentTab {
  key: string;
  label: string;
  icon: React.ReactNode;
  componentKey: string;
  data?: Record<string, unknown>;
}

export interface TaskAgentLayoutProps {
  assistantId: string;
  tabs: TaskAgentTab[];
}

export const TaskAgentLayout: React.FC<TaskAgentLayoutProps> = ({ assistantId, tabs }) => {
  return (
    <AssistantContextProvider autoLoad={true} initialAssistantId={assistantId}>
      <ConversationContextProvider>
        <TaskAgentInner assistantId={assistantId} tabs={tabs} />
      </ConversationContextProvider>
    </AssistantContextProvider>
  );
};

const TaskAgentInner: React.FC<TaskAgentLayoutProps> = ({ assistantId, tabs }) => {
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const { thread } = useConversationContext();
  const { menuCollapsed } = useChatUIContext();

  const toggleTab = useCallback(
    (key: string) => {
      setActiveKey((prev) => (prev === key ? null : key));
    },
    []
  );

  if (!thread) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
        <Spin tip="Creating conversation..." />
      </div>
    );
  }

  return (
    <div style={{ height: "100%", width: "100%", display: "flex", overflow: "hidden" }}>
      <div style={{ flex: 1, minWidth: 0, height: "100%" }}>
        <LatticeChat
          assistant_id={assistantId}
          thread_id={thread.id}
          showProjectSelector={false}
          showAgentSlot={false}
          menu={menuCollapsed ? undefined : <AgentConversations />}
          headerRight={
            <>
              {tabs.map((tab) => (
                <Button
                  key={tab.key}
                  type="text"
                  size="small"
                  icon={tab.icon}
                  onClick={() => toggleTab(tab.key)}
                  style={{
                    borderRadius: 8,
                    color: activeKey === tab.key ? "#6366f1" : undefined,
                    background: activeKey === tab.key ? "rgba(99,102,241,0.08)" : undefined,
                  }}
                  aria-pressed={activeKey === tab.key}
                >
                  {tab.label}
                </Button>
              ))}
            </>
          }
        />
      </div>

      {activeKey !== null && (
        <Panel
          activeTab={tabs.find((t) => t.key === activeKey)!}
          onClose={() => setActiveKey(null)}
        />
      )}
    </div>
  );
};

const Panel: React.FC<{
  activeTab: TaskAgentTab;
  onClose: () => void;
}> = ({ activeTab, onClose }) => {
  const elementMeta = getElement(activeTab.componentKey);
  const SidePanel = elementMeta?.side_app_view;

  return (
    <div
      style={{
        width: 480,
        height: "100%",
        borderLeft: "1px solid #e8e8e8",
        background: "#fff",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "8px 12px",
        borderBottom: "1px solid #e8e8e8",
        flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {activeTab.icon}
          <Text strong style={{ fontSize: 14 }}>{activeTab.label}</Text>
        </div>
        <Button type="text" size="small" icon={<CloseOutlined />} onClick={onClose} />
      </div>
      <div style={{ flex: 1, overflow: "auto" }}>
        {SidePanel ? (
          <SidePanel component_key={activeTab.componentKey} data={activeTab.data || {}} />
        ) : (
          <div style={{ padding: 16, color: "#999" }}>Panel not loaded</div>
        )}
      </div>
    </div>
  );
};
