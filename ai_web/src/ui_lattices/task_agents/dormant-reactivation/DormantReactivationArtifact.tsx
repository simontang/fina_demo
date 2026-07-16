import { Alert, Card, Spin, Table, Tag, Typography } from "antd";
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
    title: "Campaign",
    dataIndex: "name",
    key: "name",
    width: 220,
    ellipsis: true,
    render: (name: string, campaign) => (
      <div style={{ minWidth: 0 }}>
        <Text strong ellipsis style={{ display: "block", fontSize: 13 }}>{name}</Text>
        <Text type="secondary" style={{ fontSize: 11 }}>#{campaign.id} / {campaign.type}</Text>
      </div>
    ),
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
  {
    title: "Main Segment",
    dataIndex: "mainSegmentDataId",
    key: "mainSegmentDataId",
    width: 115,
    render: (id: number | null) => id == null ? <Text type="secondary">-</Text> : <Text code>#{id}</Text>,
  },
  {
    title: "Schedule",
    key: "schedule",
    width: 205,
    render: (_, campaign) => (
      <div>
        <Text type="secondary" style={{ display: "block", fontSize: 11 }}>{campaign.startTime}</Text>
        <Text type="secondary" style={{ display: "block", fontSize: 11 }}>{campaign.endTime}</Text>
      </div>
    ),
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
      <Card size="small" title={<Text strong>Reactivation Campaigns</Text>} style={{ borderRadius: 8 }} styles={{ body: { padding: 0 } }}>
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
            scroll={{ x: 640 }}
            onRow={(record) => ({
              onClick: () => workbench.selectCampaign(record),
              onKeyDown: (event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  workbench.selectCampaign(record);
                }
              },
              tabIndex: 0,
              "aria-selected": record.id === workbench.selectedId,
              style: {
                cursor: "pointer",
                background: record.id === workbench.selectedId ? "#f0f0ff" : undefined,
              },
            })}
          />
        )}
      </Card>

      {workbench.detailLoading ? (
        <div style={{ minHeight: 180, display: "grid", placeItems: "center" }}><Spin tip="Loading campaign detail..." /></div>
      ) : workbench.detailError ? (
        <Alert type="error" showIcon message="Campaign detail unavailable" description={workbench.detailError} />
      ) : workbench.selectedCampaign ? (
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
