import { Tag, Typography } from "antd";
import type { TableColumnsType } from "antd";
import { FileTextOutlined } from "@ant-design/icons";
import type { ElementProps } from "@axiom-lattice/react-sdk";
import { TableDetailWorkbench } from "../shared/TableDetailWorkbench";
import { SegmentDetail } from "./SegmentDetail";
import { useSegmentSummary, useSegmentWorkbench } from "./useSegmentWorkbench";
import type { SegmentDefinitionVO } from "./types";

const { Text } = Typography;

const columns: TableColumnsType<SegmentDefinitionVO> = [
  {
    title: "Name",
    dataIndex: "name",
    key: "name",
    width: 210,
    ellipsis: true,
    render: (name: string) => <Text strong ellipsis style={{ display: "block", fontSize: 13 }}>{name}</Text>,
  },
  {
    title: "Status",
    dataIndex: "status",
    key: "status",
    width: 88,
    render: (status: number) => (
      <Tag color={status === 1 ? "green" : "default"}>{status === 1 ? "active" : "disabled"}</Tag>
    ),
  },
  {
    title: "Created",
    dataIndex: "createdAt",
    key: "createdAt",
    width: 138,
    ellipsis: true,
    render: (date: string) => (
      <Text type="secondary" style={{ fontSize: 11 }}>{date?.slice(0, 16).replace("T", " ")}</Text>
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
    <TableDetailWorkbench
      title="Segment Artifact List"
      records={workbench.segments}
      columns={columns}
      getRecordId={(segment) => segment.id}
      selectedId={workbench.selectedId}
      loading={workbench.loading}
      error={workbench.error}
      emptyDescription="No segments"
      onSelect={workbench.selectSegment}
      onCloseDetail={workbench.clearSelection}
      renderDetail={() => workbench.selectedSegment ? (
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
    />
  );
};
