import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
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
import {
  TableDetailWorkbenchContext,
  type TableDetailWorkbenchContextValue,
} from "../../ui_lattices/task_agents/shared/tableDetailWorkbenchContext";
import { TABLE_DETAIL_WORKBENCH_LAYOUT } from "../../ui_lattices/task_agents/shared/tableDetailWorkbenchLayout";

const { Text } = Typography;

export interface TaskAgentTab {
  key: string;
  label: string;
  icon: React.ReactNode;
  componentKey: string;
  data?: Record<string, unknown>;
  panelLayout?: "table-detail";
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

      {activeKey !== null && tabs.some((tab) => tab.key === activeKey) && (
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
  const panelRef = useRef<HTMLDivElement>(null);
  const dragStartRef = useRef<{ x: number; width: number } | null>(null);
  const tableDetailPanel = activeTab.panelLayout === "table-detail";
  const [detailExpanded, setDetailExpandedState] = useState(false);
  const [panelWidth, setPanelWidth] = useState<number>(TABLE_DETAIL_WORKBENCH_LAYOUT.collapsedPanelWidth);
  const [dragging, setDragging] = useState(false);

  const clampPanelWidth = useCallback((width: number) => {
    const containerWidth = panelRef.current?.parentElement?.clientWidth ?? window.innerWidth;
    const maxWidth = Math.max(
      TABLE_DETAIL_WORKBENCH_LAYOUT.minPanelWidth,
      containerWidth - TABLE_DETAIL_WORKBENCH_LAYOUT.minMainWidth,
    );
    return Math.min(Math.max(width, TABLE_DETAIL_WORKBENCH_LAYOUT.minPanelWidth), maxWidth);
  }, []);

  const setDetailExpanded = useCallback((expanded: boolean) => {
    if (!tableDetailPanel) return;

    setDetailExpandedState(expanded);
    setPanelWidth((currentWidth) => (
      clampPanelWidth(
        expanded
          ? Math.max(currentWidth, TABLE_DETAIL_WORKBENCH_LAYOUT.expandedPanelWidth)
          : TABLE_DETAIL_WORKBENCH_LAYOUT.collapsedPanelWidth,
      )
    ));
  }, [clampPanelWidth, tableDetailPanel]);

  useEffect(() => {
    setDetailExpandedState(false);
    setPanelWidth(clampPanelWidth(TABLE_DETAIL_WORKBENCH_LAYOUT.collapsedPanelWidth));
  }, [activeTab.key, clampPanelWidth]);

  useEffect(() => {
    const container = panelRef.current?.parentElement;
    if (!container) return undefined;

    const observer = new ResizeObserver(() => {
      setPanelWidth((currentWidth) => clampPanelWidth(currentWidth));
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [clampPanelWidth]);

  const onResizePointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    dragStartRef.current = { x: event.clientX, width: panelWidth };
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
  }, [panelWidth]);

  const onResizePointerMove = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const dragStart = dragStartRef.current;
    if (!dragStart) return;
    setPanelWidth(clampPanelWidth(dragStart.width + dragStart.x - event.clientX));
  }, [clampPanelWidth]);

  const onResizePointerEnd = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (!dragStartRef.current) return;
    dragStartRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    setDragging(false);
  }, []);

  const tableDetailContext = useMemo<TableDetailWorkbenchContextValue>(() => ({
    detailExpanded: tableDetailPanel && detailExpanded,
    setDetailExpanded,
  }), [detailExpanded, setDetailExpanded, tableDetailPanel]);

  return (
    <div
      ref={panelRef}
      style={{
        width: panelWidth,
        height: "100%",
        borderLeft: "1px solid #e8e8e8",
        background: "#fff",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        position: "relative",
        transition: dragging ? undefined : TABLE_DETAIL_WORKBENCH_LAYOUT.panelTransition,
      }}
    >
      {tableDetailPanel && detailExpanded ? (
        <div
          role="separator"
          aria-label="Resize table detail workbench"
          aria-orientation="vertical"
          onPointerDown={onResizePointerDown}
          onPointerMove={onResizePointerMove}
          onPointerUp={onResizePointerEnd}
          onPointerCancel={onResizePointerEnd}
          style={{
            position: "absolute",
            top: 0,
            bottom: 0,
            left: -4,
            width: 8,
            cursor: "col-resize",
            touchAction: "none",
            zIndex: 1,
          }}
        />
      ) : null}
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
      <div style={{ flex: 1, minHeight: 0, overflow: tableDetailPanel ? "hidden" : "auto" }}>
        <TableDetailWorkbenchContext.Provider value={tableDetailContext}>
          {SidePanel ? (
            <SidePanel component_key={activeTab.componentKey} data={activeTab.data || {}} />
          ) : (
            <div style={{ padding: 16, color: "#999" }}>Panel not loaded</div>
          )}
        </TableDetailWorkbenchContext.Provider>
      </div>
    </div>
  );
};
