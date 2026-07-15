import { Card, Col, Row, Statistic, Tag, Typography } from "antd";
import { campaignStatusConfig } from "./types";
import type { JsonValue, MarketingCampaignVO } from "./types";

const { Text, Paragraph } = Typography;

function JsonBlock({ label, data }: { label: string; data?: JsonValue }) {
  if (!data || (typeof data === "object" && Object.keys(data).length === 0)) return null;

  return (
    <div style={{ marginBottom: 12 }}>
      <Text type="secondary" style={{ fontSize: 11 }}>{label}</Text>
      <pre style={{
        margin: "4px 0 0",
        fontSize: 11,
        background: "#f5f5f5",
        padding: 8,
        borderRadius: 6,
        overflow: "auto",
        maxHeight: 200,
      }}>
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
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
        <Paragraph style={{ margin: 0 }}><Tag>{campaign.type}</Tag></Paragraph>
      </div>
      <div>
        <Text type="secondary" style={{ fontSize: 11 }}>Status</Text>
        <Paragraph style={{ margin: 0 }}>
          <Tag color={status.color}>{status.label}</Tag>
        </Paragraph>
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
      <Row gutter={12}>
        <Col span={12}>
          <Text type="secondary" style={{ fontSize: 11 }}>Start Time</Text>
          <Paragraph style={{ margin: 0, fontSize: 12 }}>{campaign.startTime}</Paragraph>
        </Col>
        <Col span={12}>
          <Text type="secondary" style={{ fontSize: 11 }}>End Time</Text>
          <Paragraph style={{ margin: 0, fontSize: 12 }}>{campaign.endTime}</Paragraph>
        </Col>
      </Row>
      {campaign.mainSegmentDataId ? (
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

export function CampaignStrategyTab({ campaign }: { campaign: MarketingCampaignVO }) {
  const hasStrategy = Boolean(
    campaign.segmentationStrategy
    || campaign.controlGroupStrategy
    || campaign.contentChannelStrategy
    || campaign.offerStrategy
    || campaign.waveStrategy
    || campaign.abTestStrategy,
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {hasStrategy ? (
        <>
          <JsonBlock label="Segmentation Strategy" data={campaign.segmentationStrategy} />
          <JsonBlock label="Control Group Strategy" data={campaign.controlGroupStrategy} />
          <JsonBlock label="Content Channel Strategy" data={campaign.contentChannelStrategy} />
          <JsonBlock label="Offer Strategy" data={campaign.offerStrategy} />
          <JsonBlock label="Wave Strategy" data={campaign.waveStrategy} />
          <JsonBlock label="A/B Test Strategy" data={campaign.abTestStrategy} />
        </>
      ) : (
        <div style={{ padding: 24, textAlign: "center", color: "#999" }}>
          No strategy configured yet.
        </div>
      )}
    </div>
  );
}

export function CampaignStatisticsTab({ campaign }: { campaign: MarketingCampaignVO }) {
  const statistics = campaign.statistics;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {statistics ? (
        <>
          {statistics.audience ? (
            <Card size="small" title="Audience" style={{ borderRadius: 8, marginBottom: 8 }}>
              <Row gutter={12}>
                <Col span={8}>
                  <Statistic title="Target" value={statistics.audience.targetCount || 0} valueStyle={{ fontSize: 16 }} />
                </Col>
                <Col span={8}>
                  <Statistic title="Control" value={statistics.audience.controlCount || 0} valueStyle={{ fontSize: 16 }} />
                </Col>
                <Col span={8}>
                  <Statistic title="Treatment" value={statistics.audience.treatmentCount || 0} valueStyle={{ fontSize: 16 }} />
                </Col>
              </Row>
            </Card>
          ) : null}
          {statistics.delivery ? (
            <Card size="small" title="Delivery" style={{ borderRadius: 8, marginBottom: 8 }}>
              <Row gutter={12}>
                <Col span={6}>
                  <Statistic title="Sent" value={statistics.delivery.sent || 0} valueStyle={{ fontSize: 14 }} />
                </Col>
                <Col span={6}>
                  <Statistic title="Delivered" value={statistics.delivery.delivered || 0} valueStyle={{ fontSize: 14 }} />
                </Col>
                <Col span={6}>
                  <Statistic title="Opened" value={statistics.delivery.opened || 0} valueStyle={{ fontSize: 14 }} />
                </Col>
                <Col span={6}>
                  <Statistic title="Clicked" value={statistics.delivery.clicked || 0} valueStyle={{ fontSize: 14 }} />
                </Col>
              </Row>
            </Card>
          ) : null}
          {statistics.conversion ? (
            <Card size="small" title="Conversion" style={{ borderRadius: 8, marginBottom: 8 }}>
              <Row gutter={12}>
                <Col span={12}>
                  <Statistic title="Converted" value={statistics.conversion.converted || 0} valueStyle={{ fontSize: 14 }} />
                </Col>
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
            </Card>
          ) : null}
        </>
      ) : (
        <div style={{ padding: 24, textAlign: "center", color: "#999" }}>
          No statistics available yet.
        </div>
      )}
    </div>
  );
}
