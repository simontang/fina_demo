import { Card, Col, Row, Statistic, Table, Tag, Typography } from "antd";
import type { TableColumnsType } from "antd";
import { FallOutlined, RiseOutlined, UserSwitchOutlined } from "@ant-design/icons";
import type { ElementProps } from "@axiom-lattice/react-sdk";

const { Text } = Typography;

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

const columns: TableColumnsType<ChurnRecord> = [
  { title: "Customer", dataIndex: "customer", key: "customer", ellipsis: true },
  { title: "Segment", dataIndex: "segment", key: "segment", width: 110 },
  {
    title: "Churn Score",
    dataIndex: "churnScore",
    key: "churnScore",
    width: 100,
    render: (value: number) => (
      <Text style={{ color: value > 0.7 ? "#ef4444" : value > 0.4 ? "#f59e0b" : "#22c55e", fontWeight: 600 }}>
        {(value * 100).toFixed(0)}%
      </Text>
    ),
  },
  {
    title: "Risk",
    dataIndex: "risk",
    key: "risk",
    width: 80,
    render: (risk: ChurnRecord["risk"]) => (
      <Tag color={risk === "high" ? "red" : risk === "medium" ? "orange" : "green"}>{risk}</Tag>
    ),
  },
  { title: "Last Activity", dataIndex: "lastActivity", key: "lastActivity", width: 120 },
  {
    title: "Predicted Revenue",
    dataIndex: "predictedRevenue",
    key: "predictedRevenue",
    width: 140,
    render: (value: number) => `$${value.toLocaleString()}`,
  },
];

export const ChurnListCard: React.FC<ElementProps> = () => (
  <div style={{ display: "flex", alignItems: "center", gap: 8, padding: 4 }}>
    <FallOutlined style={{ color: "#ef4444" }} />
    <Text strong style={{ fontSize: 13 }}>Churn Scoring List</Text>
    <Tag style={{ marginLeft: "auto" }}>{churnData.length} records</Tag>
  </div>
);

export const ChurnListPanel: React.FC<ElementProps> = () => (
  <div style={{ padding: 16, height: "100%", overflow: "auto" }}>
    <Table<ChurnRecord>
      size="small"
      pagination={false}
      dataSource={churnData}
      rowKey="key"
      columns={columns}
    />
  </div>
);

export const ChurnDashboardCard: React.FC<ElementProps> = () => (
  <div style={{ display: "flex", alignItems: "center", gap: 8, padding: 4 }}>
    <RiseOutlined style={{ color: "#ef4444" }} />
    <Text strong style={{ fontSize: 13 }}>Churn Dashboard</Text>
    <Text type="secondary" style={{ fontSize: 12, marginLeft: "auto" }}>4.2% churn rate</Text>
  </div>
);

export const ChurnDashboardPanel: React.FC<ElementProps> = () => (
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
