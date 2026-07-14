import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Card,
  Table,
  Tabs,
  Tag,
  Typography,
  Timeline,
  Statistic,
  Row,
  Col,
  Button,
  Modal,
  Spin,
  message,
  Popconfirm,
} from "antd";
import {
  RocketOutlined,
  ClockCircleOutlined,
  WarningOutlined,
  FileTextOutlined,
  UserSwitchOutlined,
  RiseOutlined,
  FallOutlined,
  TeamOutlined,
  GiftOutlined,
  ShoppingCartOutlined,
  StarOutlined,
} from "@ant-design/icons";
import { regsiterElement, type ElementProps, useApi } from "@axiom-lattice/react-sdk";
import { TaskAgentLayout, type TaskAgentTab } from "../pages/task-agents/TaskAgentLayout";

const { Text, Paragraph } = Typography;

// ============================================================
// task_agent_runtime
// ============================================================

const mockRuntimeData = {
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

const TaskAgentRuntimeCard: React.FC<ElementProps> = ({ data }) => (
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

const TaskAgentRuntimePanel: React.FC<ElementProps> = ({ data, context }) => {
  const runtime = data?.activeRuntime ? data : mockRuntimeData;

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
          {(runtime.events || []).map((event: any, i: number) => (
            <Timeline.Item
              key={i}
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

// ============================================================
// segment_artifact_workbench
// ============================================================

interface SegmentDefinitionVO {
  id: number;
  tenantId: string;
  name: string;
  description: string;
  datasourceId: number;
  querySql: string;
  status: number;
  createdAt: string;
  updatedAt: string;
}

interface SegmentDataVO {
  id: number;
  tenantId: string;
  definitionId: number;
  runId: string;
  dataJson: string;
  rowCount: number;
  createdAt: string;
  updatedAt: string;
}

interface CdpApiResponse<T> {
  code: number;
  message: string;
  data: T;
}

interface SegmentDataPage {
  items: SegmentDataVO[];
  total: number;
  page: number;
  pageSize: number;
}

function unwrapCdpResponse<T>(response: CdpApiResponse<T>): T {
  if (response.code !== 200) {
    throw new Error(response.message || "CDP request failed");
  }
  return response.data;
}

function getErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

const API_BASE = "/api/cdp";

const traceEvents = [
  { actor: "Audience Agent", target: "Segment Discovery", status: "completed", summary: "委派 segment 发现任务", evidence: "task-delegation-001", auditId: "AUD-20260709-001" },
  { actor: "Segment Discovery", target: "Rule Engine", status: "completed", summary: "生成行为规则和 SQL 条件", evidence: "rule-gen-042", auditId: "AUD-20260709-002" },
  { actor: "Rule Engine", target: "Artifact Store", status: "completed", summary: "物化 segment artifact", evidence: "artifact-128", auditId: "AUD-20260709-003" },
  { actor: "Audience Agent", target: "Policy Engine", status: "in_progress", summary: "等待下游 policy checks", evidence: "policy-queue-007", auditId: "AUD-20260709-004" },
];

const SegmentArtifactCard: React.FC<ElementProps> = ({ data }) => {
  const [segments, setSegments] = useState<SegmentDefinitionVO[]>([]);
  const { get } = useApi();

  useEffect(() => {
    let active = true;

    void get<CdpApiResponse<SegmentDefinitionVO[]>>(`${API_BASE}/segment-definitions`)
      .then((response) => {
        const list = unwrapCdpResponse(response);
        if (active) setSegments(Array.isArray(list) ? list : []);
      })
      .catch(() => {});

    return () => {
      active = false;
    };
  }, [get]);

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

const SegmentArtifactPanel: React.FC<ElementProps> = ({ data }) => {
  const initialKey = data?.selectedSegmentKey || null;
  const { get, post, del } = useApi();
  const [segments, setSegments] = useState<SegmentDefinitionVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(
    initialKey ? Number(initialKey) : null
  );
  const [segmentData, setSegmentData] = useState<SegmentDataVO | null>(null);
  const [dataLoading, setDataLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const segmentListRequestId = useRef(0);
  const segmentDataRequestId = useRef(0);

  const loadSegments = useCallback(async () => {
    const requestId = ++segmentListRequestId.current;
    setLoading(true);
    setError(null);
    try {
      const response = await get<CdpApiResponse<SegmentDefinitionVO[]>>(`${API_BASE}/segment-definitions`);
      const list = unwrapCdpResponse(response);
      if (requestId !== segmentListRequestId.current) return;
      setSegments(Array.isArray(list) ? list : []);
    } catch (error: unknown) {
      if (requestId !== segmentListRequestId.current) return;
      setError(getErrorMessage(error, "Failed to load segments"));
    } finally {
      if (requestId === segmentListRequestId.current) setLoading(false);
    }
  }, [get]);

  useEffect(() => {
    void loadSegments();
    return () => {
      segmentListRequestId.current += 1;
    };
  }, [loadSegments]);

  const loadSegmentData = useCallback(async (definitionId: number) => {
    const requestId = ++segmentDataRequestId.current;
    setDataLoading(true);
    try {
      const response = await get<CdpApiResponse<SegmentDataPage>>(
        `${API_BASE}/segment-data?definitionId=${definitionId}&pageSize=1`,
      );
      const { items } = unwrapCdpResponse(response);
      if (requestId !== segmentDataRequestId.current) return;
      const latest: SegmentDataVO | null = Array.isArray(items) ? items[0] ?? null : null;
      setSegmentData(latest);
    } catch {
      if (requestId === segmentDataRequestId.current) setSegmentData(null);
    } finally {
      if (requestId === segmentDataRequestId.current) setDataLoading(false);
    }
  }, [get]);

  useEffect(() => {
    if (selectedId) {
      void loadSegmentData(selectedId);
    } else {
      segmentDataRequestId.current += 1;
      setSegmentData(null);
    }
    return () => {
      segmentDataRequestId.current += 1;
    };
  }, [selectedId, loadSegmentData]);

  const selectedSegment = segments.find((s) => s.id === selectedId);

  const handleDelete = async () => {
    if (!selectedId) return;
    try {
      const response = await del<CdpApiResponse<null>>(`${API_BASE}/segment-definitions/${selectedId}`);
      unwrapCdpResponse(response);
      message.success("Segment deleted");
      setSelectedId(null);
      void loadSegments();
    } catch (error: unknown) {
      message.error(getErrorMessage(error, "Delete failed"));
    }
  };

  const handleProcess = async () => {
    if (!selectedId) return;
    setProcessing(true);
    try {
      const response = await post<CdpApiResponse<SegmentDataVO>>(
        `${API_BASE}/segment-definitions/${selectedId}/process`,
        {},
      );
      unwrapCdpResponse(response);
      message.success("Process completed");
      void loadSegmentData(selectedId);
    } catch (error: unknown) {
      message.error(getErrorMessage(error, "Process failed"));
    } finally {
      setProcessing(false);
    }
  };

  const parseDataJson = (dataJson: string): Array<Record<string, unknown>> => {
    try {
      const parsed = JSON.parse(dataJson);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  };

  const rows = segmentData ? parseDataJson(segmentData.dataJson) : [];
  const dataColumns = rows.length > 0
    ? Object.keys(rows[0]).map((k) => ({ title: k, dataIndex: k, key: k, ellipsis: true }))
    : [];

  const columns: any = [
    { title: "Name", dataIndex: "name", key: "name", ellipsis: true,
      render: (n: string) => <Text strong style={{ fontSize: 13 }}>{n}</Text> },
    { title: "Status", dataIndex: "status", key: "status", width: 80,
      render: (s: number) => <Tag color={s === 1 ? "green" : "default"}>{s === 1 ? "active" : "disabled"}</Tag> },
    { title: "Created", dataIndex: "createdAt", key: "createdAt", width: 130,
      render: (d: string) => <Text type="secondary" style={{ fontSize: 12 }}>{d?.slice(0, 10)}</Text> }
  ];

  return (
    <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 16, height: "100%", overflow: "auto" }}>
      <Card size="small" title={<Text strong>Segment Artifact List</Text>} style={{ borderRadius: 12 }} styles={{ body: { padding: 0 } }}>
        {loading ? (
          <div style={{ padding: 24, textAlign: "center" }}><Spin /></div>
        ) : error ? (
          <div style={{ padding: 24, textAlign: "center", color: "#ef4444" }}>{error}</div>
        ) : (
          <Table<SegmentDefinitionVO>
            columns={columns}
            dataSource={segments}
            size="small"
            pagination={false}
            rowKey="id"
            onRow={(record) => ({
              onClick: () => setSelectedId(record.id),
              style: { cursor: "pointer", background: record.id === selectedId ? "#f0f0ff" : undefined },
            })}
          />
        )}
      </Card>

      {selectedSegment && (
        <Card
          size="small"
          title={<Text strong>{selectedSegment.name}</Text>}
          extra={
            <Popconfirm title="Delete this segment?" onConfirm={handleDelete} okText="Delete" okButtonProps={{ danger: true }}>
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
                      <Paragraph strong style={{ margin: 0, fontFamily: "monospace", fontSize: 12 }}>{selectedSegment.id}</Paragraph>
                    </div>
                    <div>
                      <Text type="secondary" style={{ fontSize: 11 }}>Datasource ID</Text>
                      <Paragraph style={{ margin: 0, fontSize: 13 }}>{selectedSegment.datasourceId}</Paragraph>
                    </div>
                    <div>
                      <Text type="secondary" style={{ fontSize: 11 }}>Status</Text>
                      <Paragraph style={{ margin: 0 }}>
                        <Tag color={selectedSegment.status === 1 ? "green" : "default"}>
                          {selectedSegment.status === 1 ? "active" : "disabled"}
                        </Tag>
                      </Paragraph>
                    </div>
                    {selectedSegment.description && (
                      <div>
                        <Text type="secondary" style={{ fontSize: 11 }}>Description</Text>
                        <Paragraph style={{ margin: 0, fontSize: 13 }}>{selectedSegment.description}</Paragraph>
                      </div>
                    )}
                    <div>
                      <Text type="secondary" style={{ fontSize: 11 }}>SQL</Text>
                      <pre style={{ margin: 0, fontSize: 11, background: "#f5f5f5", padding: 8, borderRadius: 6, overflow: "auto", maxHeight: 120 }}>
                        {selectedSegment.querySql}
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
                      <Button size="small" type="primary" loading={processing} onClick={handleProcess}>
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
                        {rows.length > 0 && (
                          <Card size="small" title="Preview" style={{ borderRadius: 8 }} styles={{ body: { padding: 0 } }}>
                            <Table
                              size="small" pagination={false}
                              dataSource={rows.map((r, i) => ({ ...r, _key: i }))}
                              columns={dataColumns.map((c) => ({ ...c, ellipsis: true }))}
                              rowKey="_key"
                              scroll={{ x: "max-content" }}
                            />
                          </Card>
                        )}
                      </>
                    ) : (
                      <div style={{ padding: 24, textAlign: "center", color: "#999" }}>
                        No data yet. Click "Process" to execute the segment SQL.
                      </div>
                    )}
                  </div>
                ),
              },
              // {
              //   key: "trace",
              //   label: "Agent Trace",
              //   children: (
              //     <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              //       <Text type="secondary" style={{ fontSize: 12 }}>
              //         Full trace timeline — actor / target / status / summary / evidence / auditId
              //       </Text>
              //       <Timeline
              //         items={traceEvents.map((e) => ({
              //           color: e.status === "completed" ? "green" : e.status === "in_progress" ? "blue" : "gray",
              //           children: (
              //             <div>
              //               <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
              //                 <Text strong style={{ fontSize: 12 }}>{e.actor}</Text>
              //                 <Text type="secondary" style={{ fontSize: 11 }}>→</Text>
              //                 <Text strong style={{ fontSize: 12 }}>{e.target}</Text>
              //                 <Tag color={e.status === "completed" ? "green" : e.status === "in_progress" ? "blue" : "default"} style={{ fontSize: 10, lineHeight: "16px" }}>{e.status}</Tag>
              //               </div>
              //               <Text style={{ fontSize: 11 }}>{e.summary}</Text>
              //               <div style={{ display: "flex", gap: 12, marginTop: 2 }}>
              //                 <Text type="secondary" style={{ fontSize: 10 }}>evidence: {e.evidence}</Text>
              //                 <Text type="secondary" style={{ fontSize: 10, fontFamily: "monospace" }}>auditId: {e.auditId}</Text>
              //               </div>
              //             </div>
              //           ),
              //         }))}
              //       />
              //     </div>
              //   ),
              // },
            ]}
          />
        </Card>
      )}
    </div>
  );
};

// ============================================================
// churn_scoring_list
// ============================================================

interface ChurnRecord {
  key: string;
  customer: string;
  segment: string;
  churnScore: number;
  risk: "high" | "medium" | "low";
  lastActivity: string;
  predictedRevenue: number;
}

const churnData: ChurnRecord[] = [
  { key: "1", customer: "Acme Corp", segment: "Enterprise", churnScore: 0.92, risk: "high", lastActivity: "3 days ago", predictedRevenue: 450000 },
  { key: "2", customer: "Globex Inc", segment: "Mid-Market", churnScore: 0.78, risk: "high", lastActivity: "1 week ago", predictedRevenue: 280000 },
  { key: "3", customer: "Initech LLC", segment: "SMB", churnScore: 0.65, risk: "medium", lastActivity: "2 days ago", predictedRevenue: 95000 },
  { key: "4", customer: "Umbrella Co", segment: "Enterprise", churnScore: 0.45, risk: "medium", lastActivity: "5 days ago", predictedRevenue: 620000 },
  { key: "5", customer: "Stark Industries", segment: "Enterprise", churnScore: 0.15, risk: "low", lastActivity: "Today", predictedRevenue: 890000 },
];

const riskColor: Record<string, string> = { high: "#ef4444", medium: "#f59e0b", low: "#22c55e" };

const ChurnListCard: React.FC<ElementProps> = ({ data }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 8, padding: 4 }}>
    <FallOutlined style={{ color: "#ef4444" }} />
    <Text strong style={{ fontSize: 13 }}>Churn Scoring List</Text>
    <Tag style={{ marginLeft: "auto" }}>{churnData.length} records</Tag>
  </div>
);

const ChurnListPanel: React.FC<ElementProps> = ({ data }) => (
  <div style={{ padding: 16, height: "100%", overflow: "auto" }}>
    <Table<ChurnRecord>
      size="small" pagination={false} dataSource={churnData} rowKey="key"
      columns={[
        { title: "Customer", dataIndex: "customer", key: "customer", ellipsis: true },
        { title: "Segment", dataIndex: "segment", key: "segment", width: 110 },
        { title: "Churn Score", dataIndex: "churnScore", key: "churnScore", width: 100, render: (v: number) => <Text style={{ color: v > 0.7 ? "#ef4444" : v > 0.4 ? "#f59e0b" : "#22c55e", fontWeight: 600 }}>{(v * 100).toFixed(0)}%</Text> },
        { title: "Risk", dataIndex: "risk", key: "risk", width: 80, render: (r: string) => <Tag color={r === "high" ? "red" : r === "medium" ? "orange" : "green"}>{r}</Tag> },
        { title: "Last Activity", dataIndex: "lastActivity", key: "lastActivity", width: 120 },
        { title: "Predicted Revenue", dataIndex: "predictedRevenue", key: "predictedRevenue", width: 140, render: (v: number) => `$${v.toLocaleString()}` },
      ]}
    />
  </div>
);

// ============================================================
// churn_scoring_dashboard
// ============================================================

const ChurnDashboardCard: React.FC<ElementProps> = ({ data }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 8, padding: 4 }}>
    <RiseOutlined style={{ color: "#ef4444" }} />
    <Text strong style={{ fontSize: 13 }}>Churn Dashboard</Text>
    <Text type="secondary" style={{ fontSize: 12, marginLeft: "auto" }}>4.2% churn rate</Text>
  </div>
);

const ChurnDashboardPanel: React.FC<ElementProps> = ({ data }) => (
  <div style={{ padding: 16 }}>
    <Row gutter={[12, 12]}>
      <Col span={24}>
        <Card size="small" style={{ borderRadius: 12 }}>
          <Statistic title="At-Risk Revenue" value={730000} prefix="$" suffix={<Text type="danger">35% of total</Text>} valueStyle={{ color: "#ef4444" }} />
        </Card>
      </Col>
      <Col span={12}>
        <Card size="small" style={{ borderRadius: 12 }}>
          <Statistic title="Churn Rate (30d)" value={4.2} suffix="%" prefix={<RiseOutlined style={{ color: "#ef4444" }} />} />
        </Card>
      </Col>
      <Col span={12}>
        <Card size="small" style={{ borderRadius: 12 }}>
          <Statistic title="Retention (90d)" value={82.5} suffix="%" prefix={<UserSwitchOutlined style={{ color: "#22c55e" }} />} />
        </Card>
      </Col>
    </Row>
  </div>
);

// ============================================================
// product_nba_list
// ============================================================

interface NBARecommendation {
  key: string;
  customer: string;
  recommendedProduct: string;
  category: string;
  confidence: number;
  estimatedRevenue: number;
  reason: string;
}

const nbaData: NBARecommendation[] = [
  { key: "1", customer: "Acme Corp", recommendedProduct: "AI Analytics Suite", category: "Analytics", confidence: 0.94, estimatedRevenue: 120000, reason: "High data volume + expiring contract" },
  { key: "2", customer: "Globex Inc", recommendedProduct: "Cloud Migration Bundle", category: "Infrastructure", confidence: 0.87, estimatedRevenue: 85000, reason: "Legacy infra + growth pattern" },
  { key: "3", customer: "Initech LLC", recommendedProduct: "Security Compliance Pack", category: "Security", confidence: 0.82, estimatedRevenue: 45000, reason: "Regulatory audit due Q3" },
  { key: "4", customer: "Umbrella Co", recommendedProduct: "Premium Support Tier", category: "Support", confidence: 0.91, estimatedRevenue: 95000, reason: "High ticket volume + SLA needs" },
  { key: "5", customer: "Stark Industries", recommendedProduct: "Enterprise API Gateway", category: "Platform", confidence: 0.76, estimatedRevenue: 210000, reason: "API traffic surge + scaling needs" },
];

const NBAListCard: React.FC<ElementProps> = ({ data }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 8, padding: 4 }}>
    <GiftOutlined style={{ color: "#6366f1" }} />
    <Text strong style={{ fontSize: 13 }}>Product NBA List</Text>
    <Tag style={{ marginLeft: "auto" }}>{nbaData.length} recommendations</Tag>
  </div>
);

const NBAListPanel: React.FC<ElementProps> = ({ data }) => (
  <div style={{ padding: 16, height: "100%", overflow: "auto" }}>
    <Table<NBARecommendation>
      size="small" pagination={false} dataSource={nbaData} rowKey="key"
      columns={[
        { title: "Customer", dataIndex: "customer", key: "customer", ellipsis: true },
        { title: "Recommended Product", dataIndex: "recommendedProduct", key: "recommendedProduct", ellipsis: true },
        { title: "Category", dataIndex: "category", key: "category", width: 90, render: (c: string) => <Tag>{c}</Tag> },
        { title: "Confidence", dataIndex: "confidence", key: "confidence", width: 90, render: (v: number) => <Text style={{ color: v > 0.85 ? "#22c55e" : v > 0.7 ? "#f59e0b" : "#ef4444", fontWeight: 600 }}>{(v * 100).toFixed(0)}%</Text> },
        { title: "Est. Revenue", dataIndex: "estimatedRevenue", key: "estimatedRevenue", width: 120, render: (v: number) => `$${v.toLocaleString()}` },
        { title: "Reason", dataIndex: "reason", key: "reason", ellipsis: true },
      ]}
    />
  </div>
);

// ============================================================
// product_nba_dashboard
// ============================================================

const NBADashboardCard: React.FC<ElementProps> = ({ data }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 8, padding: 4 }}>
    <ShoppingCartOutlined style={{ color: "#6366f1" }} />
    <Text strong style={{ fontSize: 13 }}>NBA Dashboard</Text>
    <Text type="secondary" style={{ fontSize: 12, marginLeft: "auto" }}>86% avg confidence</Text>
  </div>
);

const NBADashboardPanel: React.FC<ElementProps> = ({ data }) => (
  <div style={{ padding: 16 }}>
    <Row gutter={[12, 12]}>
      <Col span={24}>
        <Card size="small" style={{ borderRadius: 12 }}>
          <Statistic title="Total Pipeline Revenue" value={555000} prefix="$" valueStyle={{ color: "#6366f1" }} />
        </Card>
      </Col>
      <Col span={8}>
        <Card size="small" style={{ borderRadius: 12 }}>
          <Statistic title="Recommendations" value={nbaData.length} prefix={<GiftOutlined />} />
        </Card>
      </Col>
      <Col span={8}>
        <Card size="small" style={{ borderRadius: 12 }}>
          <Statistic title="Avg Confidence" value={86} suffix="%" prefix={<StarOutlined style={{ color: "#22c55e" }} />} />
        </Card>
      </Col>
      <Col span={8}>
        <Card size="small" style={{ borderRadius: 12 }}>
          <Statistic title="Categories" value={5} prefix={<ShoppingCartOutlined style={{ color: "#6366f1" }} />} />
        </Card>
      </Col>
    </Row>
  </div>
);

// ============================================================
// Register all elements
// ============================================================

regsiterElement("task_agent_runtime", {
  card_view: TaskAgentRuntimeCard,
  side_app_view: TaskAgentRuntimePanel,
});

regsiterElement("segment_artifact_workbench", {
  card_view: SegmentArtifactCard,
  side_app_view: SegmentArtifactPanel,
});

regsiterElement("churn_scoring_list", {
  card_view: ChurnListCard,
  side_app_view: ChurnListPanel,
});

regsiterElement("churn_scoring_dashboard", {
  card_view: ChurnDashboardCard,
  side_app_view: ChurnDashboardPanel,
});

regsiterElement("product_nba_list", {
  card_view: NBAListCard,
  side_app_view: NBAListPanel,
});

regsiterElement("product_nba_dashboard", {
  card_view: NBADashboardCard,
  side_app_view: NBADashboardPanel,
});

// ============================================================
// Workspace menu elements (full-page task agent views)
// ============================================================

const segmentTabs: TaskAgentTab[] = [
  // { key: "runtime", label: "Runtime", icon: <RocketOutlined />, componentKey: "task_agent_runtime" },
  { key: "artifact", label: "Artifact", icon: <FileTextOutlined />, componentKey: "segment_artifact_workbench" },
];

regsiterElement("task_agent_segment", {
  card_view: () => null,
  side_app_view: () => (
    <TaskAgentLayout assistantId="task-audience-discovery" tabs={segmentTabs} />
  ),
});
