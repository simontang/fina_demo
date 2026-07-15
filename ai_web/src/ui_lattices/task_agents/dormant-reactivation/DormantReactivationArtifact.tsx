import { Card, Spin, Table, Tag, Typography } from "antd";
import type { TableColumnsType } from "antd";
import { RocketOutlined } from "@ant-design/icons";
import type { ElementProps } from "@axiom-lattice/react-sdk";
import { CampaignDetail } from "./CampaignDetail";
import { campaignStatusConfig } from "./types";
import type { MarketingCampaignVO } from "./types";
import { useCampaignSummary, useCampaignWorkbench } from "./useCampaignWorkbench";

const { Text } = Typography;

const columns: TableColumnsType<MarketingCampaignVO> = [
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
    width: 100,
    render: (status: string) => {
      const config = campaignStatusConfig[status] || campaignStatusConfig.draft;
      return <Tag color={config.color}>{config.label}</Tag>;
    },
  },
  { title: "Type", dataIndex: "type", key: "type", width: 110, render: (type: string) => <Tag>{type}</Tag> },
  {
    title: "Start",
    dataIndex: "startTime",
    key: "startTime",
    width: 130,
    render: (date: string) => <Text type="secondary" style={{ fontSize: 12 }}>{date?.slice(0, 10)}</Text>,
  },
  {
    title: "End",
    dataIndex: "endTime",
    key: "endTime",
    width: 130,
    render: (date: string) => <Text type="secondary" style={{ fontSize: 12 }}>{date?.slice(0, 10)}</Text>,
  },
];

export const DormantReactivationArtifactCard: React.FC<ElementProps> = () => {
  const campaigns = useCampaignSummary();
  const first = campaigns[0];

  if (!first) {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: 4 }}>
        <RocketOutlined style={{ color: "#6366f1" }} />
        <Text strong style={{ fontSize: 13 }}>Dormant Reactivation</Text>
        <Tag style={{ marginLeft: "auto" }}>0 campaigns</Tag>
      </div>
    );
  }

  const status = campaignStatusConfig[first.status] || campaignStatusConfig.draft;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, padding: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <RocketOutlined style={{ color: "#6366f1" }} />
        <Text strong style={{ fontSize: 13 }}>{first.name}</Text>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <Tag color={status.color}>{status.label}</Tag>
        <Text type="secondary" style={{ fontSize: 12 }}>
          {campaigns.length} campaign{campaigns.length !== 1 ? "s" : ""}
        </Text>
      </div>
    </div>
  );
};

export const DormantReactivationArtifactPanel: React.FC<ElementProps> = ({ data }) => {
  const workbench = useCampaignWorkbench(data?.selectedCampaignKey);

  return (
    <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 16, height: "100%", overflow: "auto" }}>
      <Card size="small" title={<Text strong>Reactivation Campaigns</Text>} style={{ borderRadius: 12 }} styles={{ body: { padding: 0 } }}>
        {workbench.loading ? (
          <div style={{ padding: 24, textAlign: "center" }}><Spin /></div>
        ) : workbench.error ? (
          <div style={{ padding: 24, textAlign: "center", color: "#ef4444" }}>{workbench.error}</div>
        ) : (
          <Table<MarketingCampaignVO>
            columns={columns}
            dataSource={workbench.campaigns}
            size="small"
            pagination={false}
            rowKey="id"
            onRow={(record) => ({
              onClick: () => workbench.selectCampaign(record.id),
              style: {
                cursor: "pointer",
                background: record.id === workbench.selectedId ? "#f0f0ff" : undefined,
              },
            })}
          />
        )}
      </Card>

      {workbench.selectedCampaign ? (
        <CampaignDetail
          campaign={workbench.selectedCampaign}
          actionLoading={workbench.actionLoading}
          onStart={workbench.startSelectedCampaign}
          onStop={workbench.stopSelectedCampaign}
          onDelete={workbench.deleteSelectedCampaign}
        />
      ) : null}
    </div>
  );
};
