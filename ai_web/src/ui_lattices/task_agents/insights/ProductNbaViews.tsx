import { Card, Col, Row, Statistic, Table, Tag, Typography } from "antd";
import type { TableColumnsType } from "antd";
import { GiftOutlined, ShoppingCartOutlined, StarOutlined } from "@ant-design/icons";
import type { ElementProps } from "@axiom-lattice/react-sdk";

const { Text } = Typography;

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

const columns: TableColumnsType<NBARecommendation> = [
  { title: "Customer", dataIndex: "customer", key: "customer", ellipsis: true },
  { title: "Recommended Product", dataIndex: "recommendedProduct", key: "recommendedProduct", ellipsis: true },
  { title: "Category", dataIndex: "category", key: "category", width: 90, render: (category: string) => <Tag>{category}</Tag> },
  {
    title: "Confidence",
    dataIndex: "confidence",
    key: "confidence",
    width: 90,
    render: (value: number) => (
      <Text style={{ color: value > 0.85 ? "#22c55e" : value > 0.7 ? "#f59e0b" : "#ef4444", fontWeight: 600 }}>
        {(value * 100).toFixed(0)}%
      </Text>
    ),
  },
  { title: "Est. Revenue", dataIndex: "estimatedRevenue", key: "estimatedRevenue", width: 120, render: (value: number) => `$${value.toLocaleString()}` },
  { title: "Reason", dataIndex: "reason", key: "reason", ellipsis: true },
];

export const NBAListCard: React.FC<ElementProps> = () => (
  <div style={{ display: "flex", alignItems: "center", gap: 8, padding: 4 }}>
    <GiftOutlined style={{ color: "#6366f1" }} />
    <Text strong style={{ fontSize: 13 }}>Product NBA List</Text>
    <Tag style={{ marginLeft: "auto" }}>{nbaData.length} recommendations</Tag>
  </div>
);

export const NBAListPanel: React.FC<ElementProps> = () => (
  <div style={{ padding: 16, height: "100%", overflow: "auto" }}>
    <Table<NBARecommendation>
      size="small"
      pagination={false}
      dataSource={nbaData}
      rowKey="key"
      columns={columns}
    />
  </div>
);

export const NBADashboardCard: React.FC<ElementProps> = () => (
  <div style={{ display: "flex", alignItems: "center", gap: 8, padding: 4 }}>
    <ShoppingCartOutlined style={{ color: "#6366f1" }} />
    <Text strong style={{ fontSize: 13 }}>NBA Dashboard</Text>
    <Text type="secondary" style={{ fontSize: 12, marginLeft: "auto" }}>86% avg confidence</Text>
  </div>
);

export const NBADashboardPanel: React.FC<ElementProps> = () => (
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
