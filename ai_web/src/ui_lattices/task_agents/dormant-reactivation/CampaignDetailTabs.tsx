import { Col, Empty, Row, Statistic, Typography } from "antd";
import { OverflowSafeTag } from "../shared/OverflowSafeTag";
import { campaignStatusConfig } from "./types";
import type { CampaignPresentation, MarketingCampaignVO } from "./types";

const { Text, Paragraph } = Typography;

const statisticSectionStyle: React.CSSProperties = {
  padding: "12px 0",
  borderBottom: "1px solid #f0f0f0",
};

function countValue(value: number | undefined): number {
  return value ?? 0;
}

export function CampaignOverviewTab({ campaign }: { campaign: MarketingCampaignVO }) {
  const status = campaignStatusConfig[campaign.status] || campaignStatusConfig.draft;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div>
        <Text type="secondary" style={{ fontSize: 11 }}>Campaign ID</Text>
        <Paragraph strong style={{ margin: 0, fontFamily: "monospace", fontSize: 12 }}>{campaign.id}</Paragraph>
      </div>
      <div>
        <Text type="secondary" style={{ fontSize: 11 }}>Type</Text>
        <Paragraph style={{ margin: 0 }}><OverflowSafeTag>{campaign.type}</OverflowSafeTag></Paragraph>
      </div>
      <div>
        <Text type="secondary" style={{ fontSize: 11 }}>Status</Text>
        <Paragraph style={{ margin: 0 }}><OverflowSafeTag color={status.color}>{status.label}</OverflowSafeTag></Paragraph>
      </div>
      {campaign.goal ? (
        <div>
          <Text type="secondary" style={{ fontSize: 11 }}>Goal</Text>
          <Paragraph style={{ margin: 0, fontSize: 13 }}>{campaign.goal}</Paragraph>
        </div>
      ) : null}
      {campaign.description ? (
        <div>
          <Text type="secondary" style={{ fontSize: 11 }}>Description</Text>
          <Paragraph style={{ margin: 0, fontSize: 13 }}>{campaign.description}</Paragraph>
        </div>
      ) : null}
      <Row gutter={[12, 8]}>
        <Col xs={24} sm={12}>
          <Text type="secondary" style={{ fontSize: 11 }}>Start Time</Text>
          <Paragraph style={{ margin: 0, fontSize: 12 }}>{campaign.startTime}</Paragraph>
        </Col>
        <Col xs={24} sm={12}>
          <Text type="secondary" style={{ fontSize: 11 }}>End Time</Text>
          <Paragraph style={{ margin: 0, fontSize: 12 }}>{campaign.endTime}</Paragraph>
        </Col>
      </Row>
      {campaign.mainSegmentDataId != null ? (
        <div>
          <Text type="secondary" style={{ fontSize: 11 }}>Main Segment Data ID</Text>
          <Paragraph strong style={{ margin: 0, fontFamily: "monospace", fontSize: 12 }}>{campaign.mainSegmentDataId}</Paragraph>
        </div>
      ) : null}
      {campaign.actualStartedAt ? (
        <div>
          <Text type="secondary" style={{ fontSize: 11 }}>Actual Started At</Text>
          <Paragraph style={{ margin: 0, fontSize: 12 }}>{campaign.actualStartedAt}</Paragraph>
        </div>
      ) : null}
      {campaign.actualStoppedAt ? (
        <div>
          <Text type="secondary" style={{ fontSize: 11 }}>Actual Stopped At</Text>
          <Paragraph style={{ margin: 0, fontSize: 12 }}>{campaign.actualStoppedAt}</Paragraph>
        </div>
      ) : null}
    </div>
  );
}

function StatisticsSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={statisticSectionStyle}>
      <Text strong style={{ display: "block", marginBottom: 10 }}>{title}</Text>
      {children}
    </section>
  );
}

export function CampaignStatisticsTab({ presentation }: { presentation: CampaignPresentation }) {
  const statistics = presentation.statistics;

  if (!statistics) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No statistics available yet." />;
  }

  const hasSupportedStatistics = Boolean(statistics.audience || statistics.delivery || statistics.conversion);
  if (!hasSupportedStatistics) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No statistics available yet." />;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      {statistics.audience ? (
        <StatisticsSection title="Audience">
          <Row gutter={[12, 12]}>
            <Col span={8}><Statistic title="Target" value={countValue(statistics.audience.targetCount)} valueStyle={{ fontSize: 16 }} /></Col>
            <Col span={8}><Statistic title="Control" value={countValue(statistics.audience.controlCount)} valueStyle={{ fontSize: 16 }} /></Col>
            <Col span={8}><Statistic title="Treatment" value={countValue(statistics.audience.treatmentCount)} valueStyle={{ fontSize: 16 }} /></Col>
          </Row>
        </StatisticsSection>
      ) : null}
      {statistics.delivery ? (
        <StatisticsSection title="Delivery">
          <Row gutter={[12, 12]}>
            <Col span={6}><Statistic title="Sent" value={countValue(statistics.delivery.sent)} valueStyle={{ fontSize: 14 }} /></Col>
            <Col span={6}><Statistic title="Delivered" value={countValue(statistics.delivery.delivered)} valueStyle={{ fontSize: 14 }} /></Col>
            <Col span={6}><Statistic title="Opened" value={countValue(statistics.delivery.opened)} valueStyle={{ fontSize: 14 }} /></Col>
            <Col span={6}><Statistic title="Clicked" value={countValue(statistics.delivery.clicked)} valueStyle={{ fontSize: 14 }} /></Col>
          </Row>
        </StatisticsSection>
      ) : null}
      {statistics.conversion ? (
        <StatisticsSection title="Conversion">
          <Row gutter={[12, 12]}>
            <Col span={12}><Statistic title="Converted" value={countValue(statistics.conversion.converted)} valueStyle={{ fontSize: 14 }} /></Col>
            <Col span={12}>
              <Statistic
                title="Rate"
                value={statistics.conversion.conversionRate != null
                  ? `${(statistics.conversion.conversionRate * 100).toFixed(1)}%`
                  : "N/A"}
                valueStyle={{ fontSize: 14 }}
              />
            </Col>
          </Row>
        </StatisticsSection>
      ) : null}
    </div>
  );
}

export { CampaignStrategyTab } from "./CampaignStrategyTab";
