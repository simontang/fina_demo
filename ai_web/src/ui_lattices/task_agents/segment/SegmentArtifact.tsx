import { Card, Spin, Table, Tag, Typography } from "antd";
import type { TableColumnsType } from "antd";
import { FileTextOutlined } from "@ant-design/icons";
import type { ElementProps } from "@axiom-lattice/react-sdk";
import { SegmentDetail } from "./SegmentDetail";
import { useSegmentSummary, useSegmentWorkbench } from "./useSegmentWorkbench";
import type { SegmentDefinitionVO } from "./types";

const { Text } = Typography;

const columns: TableColumnsType<SegmentDefinitionVO> = [
  {
    title: "Name",
    dataIndex: "name",
    key: "name",
    ellipsis: true,
    render: (name: string) => <Text strong style={{ fontSize: 13 }}>{name}</Text>,
  },
  {
    title: "Status",
    dataIndex: "status",
    key: "status",
    width: 80,
    render: (status: number) => (
      <Tag color={status === 1 ? "green" : "default"}>{status === 1 ? "active" : "disabled"}</Tag>
    ),
  },
  {
    title: "Created",
    dataIndex: "createdAt",
    key: "createdAt",
    width: 160,
    render: (date: string) => (
      <Text type="secondary" style={{ fontSize: 12 }}>{date?.slice(0, 19)?.replace("T", " ")}</Text>
    ),
  },
];

export const SegmentArtifactCard: React.FC<ElementProps> = () => {
  const segments = useSegmentSummary();
  const first = segments[0];

  if (!first) {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: 4 }}>
        <FileTextOutlined style={{ color: "#6366f1" }} />
        <Text strong style={{ fontSize: 13 }}>Segment Artifact</Text>
        <Tag style={{ marginLeft: "auto" }}>0 segments</Tag>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, padding: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <FileTextOutlined style={{ color: "#6366f1" }} />
        <Text strong style={{ fontSize: 13 }}>{first.name}</Text>
      </div>
      <Text type="secondary" style={{ fontSize: 12 }}>
        Status: {first.status === 1 ? "active" : "disabled"}
      </Text>
    </div>
  );
};

export const SegmentArtifactPanel: React.FC<ElementProps> = ({ data }) => {
  const workbench = useSegmentWorkbench(data?.selectedSegmentKey);

  return (
    <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 16, height: "100%", overflow: "auto" }}>
      <Card size="small" title={<Text strong>Segment Artifact List</Text>} style={{ borderRadius: 12 }} styles={{ body: { padding: 0 } }}>
        {workbench.loading ? (
          <div style={{ padding: 24, textAlign: "center" }}><Spin /></div>
        ) : workbench.error ? (
          <div style={{ padding: 24, textAlign: "center", color: "#ef4444" }}>{workbench.error}</div>
        ) : (
          <Table<SegmentDefinitionVO>
            columns={columns}
            dataSource={workbench.segments}
            size="small"
            pagination={false}
            rowKey="id"
            onRow={(record) => ({
              onClick: () => workbench.selectSegment(record),
              style: {
                cursor: "pointer",
                background: record.id === workbench.selectedId ? "#f0f0ff" : undefined,
              },
            })}
          />
        )}
      </Card>

      {workbench.selectedSegment ? (
        <SegmentDetail
          segment={workbench.selectedSegment}
          segmentData={workbench.segmentData}
          rows={workbench.rows}
          dataLoading={workbench.dataLoading}
          processing={workbench.processing}
          onDelete={workbench.deleteSelectedSegment}
          onProcess={workbench.processSelectedSegment}
        />
      ) : null}
    </div>
  );
};
