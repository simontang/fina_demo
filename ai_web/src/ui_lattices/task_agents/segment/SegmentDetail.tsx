import { useMemo } from "react";
import { Button, Card, Col, Popconfirm, Row, Spin, Table, Tabs, Tag, Typography } from "antd";
import type { TableColumnsType } from "antd";
import { WarningOutlined } from "@ant-design/icons";
import type { SegmentDataRow, SegmentDataVO, SegmentDefinitionVO } from "./types";

const { Text, Paragraph } = Typography;

interface SegmentPreviewRow extends SegmentDataRow {
  _key: number;
}

interface SegmentDetailProps {
  segment: SegmentDefinitionVO;
  segmentData: SegmentDataVO | null;
  rows: SegmentDataRow[];
  dataLoading: boolean;
  processing: boolean;
  onDelete: () => Promise<void>;
  onProcess: () => Promise<void>;
}

export function SegmentDetail({
  segment,
  segmentData,
  rows,
  dataLoading,
  processing,
  onDelete,
  onProcess,
}: SegmentDetailProps) {
  const previewRows = useMemo<SegmentPreviewRow[]>(
    () => rows.map((row, index) => ({ ...row, _key: index })),
    [rows],
  );
  const dataColumns = useMemo<TableColumnsType<SegmentPreviewRow>>(
    () => rows.length > 0
      ? Object.keys(rows[0]).map((key) => ({ title: key, dataIndex: key, key, ellipsis: true }))
      : [],
    [rows],
  );

  return (
    <Card
      size="small"
      title={<Text strong>{segment.name}</Text>}
      extra={
        <Popconfirm title="Delete this segment?" onConfirm={onDelete} okText="Delete" okButtonProps={{ danger: true }}>
          <Button type="text" size="small" danger icon={<WarningOutlined />}>Delete</Button>
        </Popconfirm>
      }
      style={{ borderRadius: 12, flex: 1 }}
    >
      <Tabs
        defaultActiveKey="overview"
        size="small"
        items={[
          {
            key: "overview",
            label: "Overview",
            children: (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <div>
                  <Text type="secondary" style={{ fontSize: 11 }}>Segment ID</Text>
                  <Paragraph strong style={{ margin: 0, fontFamily: "monospace", fontSize: 12 }}>{segment.id}</Paragraph>
                </div>
                <div>
                  <Text type="secondary" style={{ fontSize: 11 }}>Datasource ID</Text>
                  <Paragraph style={{ margin: 0, fontSize: 13 }}>{segment.datasourceId}</Paragraph>
                </div>
                <div>
                  <Text type="secondary" style={{ fontSize: 11 }}>Status</Text>
                  <Paragraph style={{ margin: 0 }}>
                    <Tag color={segment.status === 1 ? "green" : "default"}>
                      {segment.status === 1 ? "active" : "disabled"}
                    </Tag>
                  </Paragraph>
                </div>
                {segment.description ? (
                  <div>
                    <Text type="secondary" style={{ fontSize: 11 }}>Description</Text>
                    <Paragraph style={{ margin: 0, fontSize: 13 }}>{segment.description}</Paragraph>
                  </div>
                ) : null}
                <div>
                  <Text type="secondary" style={{ fontSize: 11 }}>SQL</Text>
                  <pre style={{ margin: 0, fontSize: 11, background: "#f5f5f5", padding: 8, borderRadius: 6, overflow: "auto", maxHeight: 120 }}>
                    {segment.querySql}
                  </pre>
                </div>
              </div>
            ),
          },
          {
            key: "detail",
            label: "Detail",
            children: (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <Text strong style={{ fontSize: 13 }}>Segment Data</Text>
                  <Button size="small" type="primary" loading={processing} onClick={onProcess}>
                    Process
                  </Button>
                </div>
                {dataLoading ? (
                  <div style={{ padding: 24, textAlign: "center" }}><Spin /></div>
                ) : segmentData ? (
                  <>
                    <Row gutter={12}>
                      <Col span={8}>
                        <Card size="small" style={{ borderRadius: 8, textAlign: "center" }}>
                          <Text type="secondary" style={{ fontSize: 11 }}>Row Count</Text>
                          <Paragraph strong style={{ margin: 0, fontSize: 18 }}>{segmentData.rowCount}</Paragraph>
                        </Card>
                      </Col>
                      <Col span={16}>
                        <Card size="small" style={{ borderRadius: 8, textAlign: "center" }}>
                          <Text type="secondary" style={{ fontSize: 11 }}>Run ID</Text>
                          <Paragraph style={{ margin: 0, fontSize: 11, fontFamily: "monospace" }}>{segmentData.runId}</Paragraph>
                        </Card>
                      </Col>
                    </Row>
                    {previewRows.length > 0 ? (
                      <Card size="small" title="Preview" style={{ borderRadius: 8 }} styles={{ body: { padding: 0 } }}>
                        <Table<SegmentPreviewRow>
                          size="small"
                          pagination={false}
                          dataSource={previewRows}
                          columns={dataColumns}
                          rowKey="_key"
                          scroll={{ x: "max-content" }}
                        />
                      </Card>
                    ) : null}
                  </>
                ) : (
                  <div style={{ padding: 24, textAlign: "center", color: "#999" }}>
                    No data yet. Click "Process" to execute the segment SQL.
                  </div>
                )}
              </div>
            ),
          },
        ]}
      />
    </Card>
  );
}
