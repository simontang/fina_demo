import { Button, Card, Popconfirm, Tabs, Typography } from "antd";
import { PauseCircleOutlined, PlayCircleOutlined, WarningOutlined } from "@ant-design/icons";
import { CampaignOverviewTab, CampaignStatisticsTab, CampaignStrategyTab } from "./CampaignDetailTabs";
import type { MarketingCampaignVO } from "./types";

const { Text } = Typography;

interface CampaignDetailProps {
  campaign: MarketingCampaignVO;
  actionLoading: boolean;
  onStart: () => Promise<void>;
  onStop: () => Promise<void>;
  onDelete: () => Promise<void>;
}

export function CampaignDetail({ campaign, actionLoading, onStart, onStop, onDelete }: CampaignDetailProps) {
  return (
    <Card
      size="small"
      title={<Text strong>{campaign.name}</Text>}
      extra={
        <div style={{ display: "flex", gap: 4 }}>
          {campaign.status === "draft" || campaign.status === "scheduled" ? (
            <Button
              type="primary"
              size="small"
              icon={<PlayCircleOutlined />}
              loading={actionLoading}
              onClick={onStart}
            >
              Start
            </Button>
          ) : null}
          {campaign.status === "running" || campaign.status === "scheduled" ? (
            <Button
              size="small"
              icon={<PauseCircleOutlined />}
              loading={actionLoading}
              onClick={onStop}
            >
              Stop
            </Button>
          ) : null}
          <Popconfirm title="Delete this campaign?" onConfirm={onDelete} okText="Delete" okButtonProps={{ danger: true }}>
            <Button type="text" size="small" danger icon={<WarningOutlined />}>Delete</Button>
          </Popconfirm>
        </div>
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
            children: <CampaignOverviewTab campaign={campaign} />,
          },
          { key: "strategy", label: "Strategy", children: <CampaignStrategyTab campaign={campaign} /> },
          { key: "statistics", label: "Statistics", children: <CampaignStatisticsTab campaign={campaign} /> },
        ]}
      />
    </Card>
  );
}
