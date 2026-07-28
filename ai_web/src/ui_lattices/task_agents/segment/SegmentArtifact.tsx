import { Alert, Button, Empty, Spin, Tag, Typography } from "antd";
import { FileTextOutlined } from "@ant-design/icons";
import type { ElementProps } from "@axiom-lattice/react-sdk";
import { ArtifactSelector } from "../shared/ArtifactSelector";
import { SegmentDetail } from "./SegmentDetail";
import { useSegmentSummary, useSegmentWorkbench } from "./useSegmentWorkbench";
import type { SegmentDefinitionVO } from "./types";

const { Text } = Typography;
const getSegmentId = (segment: SegmentDefinitionVO) => segment.id;
const getSegmentName = (segment: SegmentDefinitionVO) => segment.name;

function formatSegmentCreatedAt(value: string): string {
  return value?.slice(0, 16).replace("T", " ") || "-";
}

function renderSegmentOption(segment: SegmentDefinitionVO) {
  const statusLabel = segment.status === 1 ? "active" : "disabled";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 0, padding: "2px 0" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        <Text
          strong
          ellipsis={{ tooltip: segment.name }}
          style={{ display: "block", flex: 1, minWidth: 0, fontSize: 13 }}
        >
          {segment.name}
        </Text>
        <Tag color={segment.status === 1 ? "green" : "default"} style={{ flexShrink: 0, marginInlineEnd: 0 }}>
          {statusLabel}
        </Tag>
      </div>
      <Text type="secondary" style={{ fontSize: 11 }}>
        Created {formatSegmentCreatedAt(segment.createdAt)}
      </Text>
    </div>
  );
}

function renderSegmentSummary(segment: SegmentDefinitionVO) {
  const statusLabel = segment.status === 1 ? "active" : "disabled";

  return (
    <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: "4px 8px" }}>
      <Tag color={segment.status === 1 ? "green" : "default"} style={{ marginInlineEnd: 0 }}>
        {statusLabel}
      </Tag>
      <Text type="secondary" style={{ fontSize: 12 }}>
        Created {formatSegmentCreatedAt(segment.createdAt)}
      </Text>
    </div>
  );
}

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

  if (workbench.loading && !workbench.selectedSegment) {
    return (
      <div style={{ height: "100%", display: "grid", placeItems: "center" }}>
        <Spin tip="Loading segments..." />
      </div>
    );
  }

  if (workbench.error && !workbench.selectedSegment) {
    return (
      <div style={{ padding: 16 }}>
        <Alert
          type="error"
          showIcon
          message="Segment artifacts unavailable"
          description={workbench.error}
          action={<Button size="small" onClick={workbench.retryList}>Retry</Button>}
        />
      </div>
    );
  }

  if (!workbench.loading && !workbench.error && workbench.overallTotal === 0) {
    return (
      <div style={{ height: "100%", display: "grid", placeItems: "center", padding: 16 }}>
        <Empty description="No segment artifacts" />
      </div>
    );
  }

  return (
    <div
      style={{
        boxSizing: "border-box",
        display: "flex",
        flexDirection: "column",
        height: "100%",
        minHeight: 0,
        overflow: "hidden",
        padding: 16,
      }}
    >
      <ArtifactSelector
        ariaLabel="Select segment artifact"
        placeholder="Select a segment artifact"
        items={workbench.segments}
        selectedItem={workbench.selectedSegment}
        getId={getSegmentId}
        getName={getSegmentName}
        renderOption={renderSegmentOption}
        renderSummary={renderSegmentSummary}
        searchValue={workbench.searchValue}
        searchActive={Boolean(workbench.query)}
        overallTotal={workbench.overallTotal}
        matchedTotal={workbench.matchedTotal}
        loading={workbench.loading}
        loadingMore={workbench.loadingMore}
        error={workbench.error}
        onSearch={workbench.setSearchValue}
        onSelect={workbench.selectSegment}
        onLoadMore={workbench.loadMore}
        onRetry={workbench.retryList}
      />
      <div style={{ flex: 1, minHeight: 0, overflowY: "auto", paddingTop: 12 }}>
        {workbench.selectedSegment ? (
          <SegmentDetail
            segment={workbench.selectedSegment}
            segmentData={workbench.segmentData}
            rows={workbench.rows}
            dataLoading={workbench.dataLoading}
            dataError={workbench.dataError}
            processing={workbench.processing}
            onDelete={workbench.deleteSelectedSegment}
            onProcess={workbench.processSelectedSegment}
            onRetryData={workbench.retrySegmentData}
          />
        ) : null}
      </div>
    </div>
  );
};
