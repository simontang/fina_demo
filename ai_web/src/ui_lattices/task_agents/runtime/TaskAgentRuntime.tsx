import { Card, Tag, Timeline, Typography } from "antd";
import { ClockCircleOutlined, RocketOutlined } from "@ant-design/icons";
import type { ElementProps } from "@axiom-lattice/react-sdk";

const { Text, Paragraph } = Typography;

interface RuntimeEvent {
  label: string;
  status: "completed" | "in_progress" | string;
  time: string;
}

interface RuntimeData {
  activeRuntime: string;
  currentState: string;
  selectedArtifact: string;
  nextAction: string;
  events: RuntimeEvent[];
}

const mockRuntimeData: RuntimeData = {
  activeRuntime: "TASK-AUDIENCE-DR-001",
  currentState: "Segment 已物化，等待下游 policy checks",
  selectedArtifact: "120-day dormant high-value members",
  nextAction: "切换到 Artifact mode 审核产物",
  events: [
    { label: "委派 Segment Discovery", status: "completed", time: "2026-07-09 10:00:32" },
    { label: "生成规则和 SQL condition", status: "completed", time: "2026-07-09 10:01:15" },
    { label: "物化 segment artifact", status: "in_progress", time: "2026-07-09 10:02:08" },
  ],
};

export const TaskAgentRuntimeCard: React.FC<ElementProps> = ({ data }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 8, padding: 4 }}>
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <RocketOutlined style={{ color: "#6366f1" }} />
      <Text strong style={{ fontSize: 13 }}>Runtime</Text>
      <Tag color="processing" style={{ marginLeft: "auto" }}>Active</Tag>
    </div>
    <Text style={{ fontFamily: "monospace", fontSize: 12, color: "#888" }}>
      {data?.activeRuntime || mockRuntimeData.activeRuntime}
    </Text>
    <Text style={{ fontSize: 12 }}>{data?.currentState || mockRuntimeData.currentState}</Text>
  </div>
);

export const TaskAgentRuntimePanel: React.FC<ElementProps> = ({ data }) => {
  const runtime = (data?.activeRuntime ? data : mockRuntimeData) as RuntimeData;

  return (
    <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 16, height: "100%", overflow: "auto" }}>
      <Card
        size="small"
        title={
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <RocketOutlined style={{ color: "#6366f1" }} />
            <Text strong>Runtime</Text>
          </div>
        }
        style={{ borderRadius: 12 }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div>
            <Text type="secondary" style={{ fontSize: 12 }}>Active Runtime</Text>
            <Paragraph strong style={{ margin: 0, fontFamily: "monospace", fontSize: 13 }}>
              {runtime.activeRuntime}
            </Paragraph>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: 12 }}>Current State</Text>
            <Paragraph style={{ margin: 0, fontSize: 13 }}>{runtime.currentState}</Paragraph>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: 12 }}>Selected Artifact</Text>
            <Paragraph strong style={{ margin: 0, fontSize: 13 }}>
              {runtime.selectedArtifact}
            </Paragraph>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: 12 }}>Next Action</Text>
            <Paragraph style={{ margin: 0, fontSize: 13 }}>{runtime.nextAction}</Paragraph>
          </div>
        </div>
      </Card>

      <Card
        size="small"
        title={
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <ClockCircleOutlined style={{ color: "#6366f1" }} />
            <Text strong>Recent Events</Text>
          </div>
        }
        style={{ borderRadius: 12 }}
      >
        <Timeline>
          {(runtime.events || []).map((event, index) => (
            <Timeline.Item
              key={index}
              color={event.status === "completed" ? "green" : event.status === "in_progress" ? "blue" : "gray"}
            >
              <Text strong style={{ fontSize: 13 }}>{event.label}</Text>
              <br />
              <Text type="secondary" style={{ fontSize: 11 }}>{event.time}</Text>
            </Timeline.Item>
          ))}
        </Timeline>
      </Card>
    </div>
  );
};
