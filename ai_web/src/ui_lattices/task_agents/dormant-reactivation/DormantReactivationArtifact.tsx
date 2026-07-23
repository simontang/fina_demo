import { Tag, Typography } from "antd";
import type { TableColumnsType } from "antd";
import { RocketOutlined } from "@ant-design/icons";
import type { ElementProps } from "@axiom-lattice/react-sdk";
import { TableDetailWorkbench } from "../shared/TableDetailWorkbench";
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
    width: 210,
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
    width: 96,
    render: (status: string) => {
      const config = campaignStatusConfig[status] || campaignStatusConfig.draft;
      return <Tag color={config.color}>{config.label}</Tag>;
    },
  },
  {
    title: "Start",
    dataIndex: "startTime",
    key: "startTime",
    width: 138,
    ellipsis: true,
    render: (value: string) => <Text type="secondary" style={{ fontSize: 11 }}>{value?.slice(0, 16).replace("T", " ")}</Text>,
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
    <TableDetailWorkbench
      title="Reactivation Campaigns"
      records={workbench.campaigns}
      columns={columns}
      getRecordId={(campaign) => campaign.id}
      selectedId={workbench.selectedId}
      loading={workbench.loading}
      error={workbench.error}
      emptyDescription="No reactivation campaigns"
      detailLoading={workbench.detailLoading}
      detailError={workbench.detailError}
      onSelect={workbench.selectCampaign}
      onCloseDetail={workbench.clearSelection}
      renderDetail={() => workbench.selectedCampaign ? (
        <CampaignDetail
          campaign={workbench.selectedCampaign}
          actionLoading={workbench.actionLoading}
          onStart={workbench.startSelectedCampaign}
          onStop={workbench.stopSelectedCampaign}
          onDelete={workbench.deleteSelectedCampaign}
        />
      ) : null}
    />
  );
};
