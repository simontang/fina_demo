import { Alert, Button, Empty, Spin, Tag, Typography } from "antd";
import { RocketOutlined } from "@ant-design/icons";
import type { ElementProps } from "@axiom-lattice/react-sdk";
import { ArtifactSelector } from "../shared/ArtifactSelector";
import { CampaignDetail } from "./CampaignDetail";
import { campaignStatusConfig } from "./types";
import type { MarketingCampaignVO } from "./types";
import { useCampaignSummary, useCampaignWorkbench } from "./useCampaignWorkbench";

const { Text } = Typography;
const getCampaignId = (campaign: MarketingCampaignVO) => campaign.id;
const getCampaignName = (campaign: MarketingCampaignVO) => campaign.name;

function displayValue(value: string | null | undefined): string {
  return value?.trim() || "-";
}

function getCampaignStatus(campaign: MarketingCampaignVO) {
  return campaignStatusConfig[campaign.status] || campaignStatusConfig.draft;
}

function renderCampaignOption(campaign: MarketingCampaignVO) {
  const status = getCampaignStatus(campaign);
  const identity = `#${campaign.id} / ${displayValue(campaign.type)}`;
  const mainSegment = campaign.mainSegmentDataId == null ? "Main Segment -" : `Main Segment #${campaign.mainSegmentDataId}`;
  const schedule = `${displayValue(campaign.startTime)} -> ${displayValue(campaign.endTime)}`;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3, minWidth: 0, padding: "2px 0" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        <Text
          strong
          ellipsis={{ tooltip: campaign.name }}
          style={{ display: "block", flex: 1, minWidth: 0, fontSize: 13 }}
        >
          {campaign.name}
        </Text>
        <Tag color={status.color} style={{ flexShrink: 0, marginInlineEnd: 0 }}>{status.label}</Tag>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
        <Text type="secondary" ellipsis={{ tooltip: identity }} style={{ minWidth: 0, fontSize: 11 }}>
          {identity}
        </Text>
        <Text type="secondary" ellipsis={{ tooltip: mainSegment }} style={{ minWidth: 0, fontSize: 11 }}>
          {mainSegment}
        </Text>
      </div>
      <Text type="secondary" ellipsis={{ tooltip: schedule }} style={{ display: "block", fontSize: 11 }}>
        {schedule}
      </Text>
    </div>
  );
}

function renderCampaignSummary(campaign: MarketingCampaignVO) {
  const status = getCampaignStatus(campaign);
  const identity = `#${campaign.id} / ${displayValue(campaign.type)}`;
  const mainSegment = campaign.mainSegmentDataId == null ? "Main Segment -" : `Main Segment #${campaign.mainSegmentDataId}`;
  const schedule = `${displayValue(campaign.startTime)} -> ${displayValue(campaign.endTime)}`;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
      <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: "4px 8px", minWidth: 0 }}>
        <Tag color={status.color} style={{ marginInlineEnd: 0 }}>{status.label}</Tag>
        <Text type="secondary" ellipsis={{ tooltip: identity }} style={{ minWidth: 0, fontSize: 11 }}>
          {identity}
        </Text>
        <Text type="secondary" ellipsis={{ tooltip: mainSegment }} style={{ minWidth: 0, fontSize: 11 }}>
          {mainSegment}
        </Text>
      </div>
      <Text type="secondary" ellipsis={{ tooltip: schedule }} style={{ display: "block", fontSize: 11 }}>
        {schedule}
      </Text>
    </div>
  );
}

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

  if (workbench.loading && !workbench.selectedOption) {
    return (
      <div style={{ height: "100%", display: "grid", placeItems: "center" }}>
        <Spin tip="Loading campaigns..." />
      </div>
    );
  }

  if (workbench.error && !workbench.selectedOption) {
    return (
      <div style={{ padding: 16 }}>
        <Alert
          type="error"
          showIcon
          message="Reactivation campaigns unavailable"
          description={workbench.error}
          action={<Button size="small" onClick={workbench.retryList}>Retry</Button>}
        />
      </div>
    );
  }

  if (!workbench.loading && !workbench.error && workbench.overallTotal === 0) {
    return (
      <div style={{ height: "100%", display: "grid", placeItems: "center", padding: 16 }}>
        <Empty description="No reactivation campaigns" />
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
        ariaLabel="Select reactivation campaign"
        placeholder="Select a reactivation campaign"
        items={workbench.campaigns}
        selectedItem={workbench.selectedOption}
        getId={getCampaignId}
        getName={getCampaignName}
        renderOption={renderCampaignOption}
        renderSummary={renderCampaignSummary}
        searchValue={workbench.searchValue}
        searchActive={Boolean(workbench.query)}
        overallTotal={workbench.overallTotal}
        matchedTotal={workbench.matchedTotal}
        loading={workbench.loading}
        loadingMore={workbench.loadingMore}
        error={workbench.error}
        onSearch={workbench.setSearchValue}
        onSelect={workbench.selectCampaign}
        onLoadMore={workbench.loadMore}
        onRetry={workbench.retryList}
      />
      <div style={{ flex: 1, minHeight: 0, overflowY: "auto", paddingTop: 12 }}>
        {workbench.detailLoading ? (
          <div style={{ minHeight: 160, display: "grid", placeItems: "center" }}><Spin /></div>
        ) : workbench.detailError ? (
          <Alert
            type="error"
            showIcon
            message="Campaign detail unavailable"
            description={workbench.detailError}
            action={<Button size="small" onClick={workbench.retryCampaignDetail}>Retry</Button>}
          />
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
    </div>
  );
};
